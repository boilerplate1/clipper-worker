"""Analysis half of the reframe engine: cameras, speakers, per-scene strategy.

This is the subset of OpenShorts' ``main.py`` that the reframe engine needs,
kept free of the heavy imports at module scope: torch / ultralytics / mediapipe
are pulled in lazily on first use, so importing this module (and running the
pure state-machine tests) works in a minimal environment. YOLO and BlazeFace
are not thread-safe, so every inference goes through ``DETECT_LOCK``.
"""

import os
import threading

import cv2
import numpy as np

ASPECT_RATIO = 9 / 16

# Consecutive detections a large target move must survive before the camera
# follows it (see SmoothedCameraman.update_target). Env-overridable so the
# damping can be dialled back without a deploy; 1 restores the old behaviour.
JUMP_CONFIRM_FRAMES = max(int(os.environ.get("JUMP_CONFIRM_FRAMES", "3")), 1)


class SmoothedCameraman:
    """
    Handles smooth camera movement.
    Simplified Logic: "Heavy Tripod"
    Only moves if the subject leaves the center safe zone.
    Moves slowly and linearly.
    """

    def __init__(
        self,
        output_width,
        output_height,
        video_width,
        video_height,
        aspect_ratio=ASPECT_RATIO,
        top_pad=0,
        bottom_pad=0,
    ):
        self.output_width = output_width
        self.output_height = output_height
        self.video_width = video_width
        self.video_height = video_height
        self.aspect_ratio = aspect_ratio
        self.top_pad = top_pad
        self.bottom_pad = bottom_pad

        self.current_center_x = video_width / 2
        self.target_center_x = video_width / 2

        self.crop_height = max(100, video_height - top_pad - bottom_pad)
        self.crop_width = int(self.crop_height * aspect_ratio)
        if self.crop_width > video_width:
            self.crop_width = video_width
            self.crop_height = int(self.crop_width / aspect_ratio)

        # As long as the target is within this zone relative to the current center,
        # the camera stays still.
        self.safe_zone_radius = self.crop_width * 0.25

        # A target that teleports further than the safe zone in one detection is
        # far more often a detector error — a second face, a false positive, a
        # box snapping to a different body part — than a person who actually
        # moved that far. Committing to it immediately is what made the camera
        # swing: measured on real user footage, 22% of target updates jumped
        # more than the entire safe zone. So a big move has to REPEAT this many
        # times before the camera follows it; a wrong reading disappears on the
        # next detection and never moves the frame.
        #
        # The cost is latency on a genuinely fast move: at DETECT_STRIDE=4 and
        # 30fps, three confirmations is ~0.4s. That reads as an operator being
        # unhurried, which is the look we want, and it is far cheaper than the
        # whip-panning it replaces.
        self.jump_confirm_frames = JUMP_CONFIRM_FRAMES
        self._pending_target = None
        self._pending_count = 0

    def update_target(self, face_box):
        """Update the target centre from a detection, ignoring lone big jumps."""
        if not face_box:
            return
        x, y, w, h = face_box
        new_center = x + w / 2

        if abs(new_center - self.target_center_x) > self.safe_zone_radius:
            # Same big move as last time? Count it. Otherwise start counting
            # afresh — two contradictory outliers must not confirm each other.
            if (
                self._pending_target is not None
                and abs(new_center - self._pending_target) <= self.safe_zone_radius
            ):
                self._pending_count += 1
            else:
                self._pending_target = new_center
                self._pending_count = 1
            if self._pending_count < self.jump_confirm_frames:
                return  # not convinced yet — hold the frame

        self._pending_target = None
        self._pending_count = 0
        self.target_center_x = new_center

    def get_crop_box(self, force_snap=False):
        """
        Returns the (x1, y1, x2, y2) for the current frame.
        """
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            diff = self.target_center_x - self.current_center_x

            if abs(diff) > self.safe_zone_radius:
                direction = 1 if diff > 0 else -1

                # Large jumps (scene change, fast movement) reframe quickly instead
                # of panning across the whole frame at walking speed.
                if abs(diff) > self.crop_width * 0.5:
                    speed = 15.0
                else:
                    speed = 3.0

                self.current_center_x += direction * speed

                # Check if we overshot (prevent oscillation)
                new_diff = self.target_center_x - self.current_center_x
                if (direction == 1 and new_diff < 0) or (direction == -1 and new_diff > 0):
                    self.current_center_x = self.target_center_x

            # If inside safe zone, DO NOTHING (Stationary Camera)

        half_crop = self.crop_width / 2

        if self.current_center_x - half_crop < 0:
            self.current_center_x = half_crop
        if self.current_center_x + half_crop > self.video_width:
            self.current_center_x = self.video_width - half_crop

        x1 = int(self.current_center_x - half_crop)
        x2 = int(self.current_center_x + half_crop)

        x1 = max(0, x1)
        x2 = min(self.video_width, x2)

        y1 = self.top_pad
        y2 = self.video_height - self.bottom_pad

        return x1, y1, x2, y2


class SpeakerTracker:
    """
    Tracks speakers over time to prevent rapid switching and handle temporary obstructions.
    """

    def __init__(self, stabilization_frames=15, cooldown_frames=30):
        self.active_speaker_id = None
        self.speaker_scores = {}  # {id: score}
        self.last_seen = {}  # {id: frame_number}
        self.locked_counter = 0  # How long we've been locked on current speaker

        self.stabilization_threshold = (
            stabilization_frames  # Frames needed to confirm a new speaker
        )
        self.switch_cooldown = cooldown_frames  # Minimum frames before switching again
        self.last_switch_frame = -1000

        self.next_id = 0
        self.known_faces = []  # [{'id': 0, 'center': x, 'last_frame': 123}]

    def get_target(self, face_candidates, frame_number, width):
        """
        Decides which face to focus on.
        face_candidates: list of {'box': [x,y,w,h], 'score': float}
        """
        current_candidates = []

        for face in face_candidates:
            x, y, w, h = face["box"]
            center_x = x + w / 2

            best_match_id = -1
            min_dist = width * 0.15  # Reduced matching radius to avoid jumping in groups

            for kf in self.known_faces:
                if frame_number - kf["last_frame"] > 30:  # Forgot faces older than 1s (was 2s)
                    continue

                dist = abs(center_x - kf["center"])
                if dist < min_dist:
                    min_dist = dist
                    best_match_id = kf["id"]

            # If no match, assign new ID
            if best_match_id == -1:
                best_match_id = self.next_id
                self.next_id += 1

            self.known_faces = [kf for kf in self.known_faces if kf["id"] != best_match_id]
            self.known_faces.append(
                {"id": best_match_id, "center": center_x, "last_frame": frame_number}
            )

            current_candidates.append(
                {"id": best_match_id, "box": face["box"], "score": face["score"]}
            )

        for pid in list(self.speaker_scores.keys()):
            self.speaker_scores[pid] *= 0.85  # Faster decay (was 0.9)
            if self.speaker_scores[pid] < 0.1:
                del self.speaker_scores[pid]

        for cand in current_candidates:
            pid = cand["id"]
            # Score is purely based on size (proximity) now that we don't have mouth
            raw_score = cand["score"] / (width * width * 0.05)
            self.speaker_scores[pid] = self.speaker_scores.get(pid, 0) + raw_score

        if not current_candidates:
            # If no one found, maintain last active speaker if cooldown allows
            # to avoid black screen or jump to 0,0
            return None

        best_candidate = None
        max_score = -1

        for cand in current_candidates:
            pid = cand["id"]
            total_score = self.speaker_scores.get(pid, 0)

            # Hysteresis: HUGE Bonus for current active speaker
            if pid == self.active_speaker_id:
                total_score *= 3.0  # Sticky factor

            if total_score > max_score:
                max_score = total_score
                best_candidate = cand

        if best_candidate:
            target_id = best_candidate["id"]

            if target_id == self.active_speaker_id:
                self.locked_counter += 1
                return best_candidate["box"]

            # New person. The cooldown must hold whether or not the current
            # speaker happens to be detected in THIS frame.
            #
            # It used to fall through and switch when the active speaker was
            # missing from the candidate list — a blink, a head turn or one
            # motion-blurred frame was enough. That is precisely when the
            # cooldown is needed, so it only ever fired when it wasn't: 3 of 7
            # target switches measured on a 12s clip (25-jul-2026) jumped the
            # cooldown this way, and every jump drags the camera across frame.
            #
            # Returning None holds instead: the caller only calls
            # update_target() on a truthy box, so the camera keeps its current
            # target and finishes whatever move it was making. The hold is
            # bounded by the cooldown itself — once it expires, a speaker who
            # really did leave the shot is switched away from normally.
            if frame_number - self.last_switch_frame < self.switch_cooldown:
                old_cand = next(
                    (c for c in current_candidates if c["id"] == self.active_speaker_id), None
                )
                return old_cand["box"] if old_cand else None

            self.active_speaker_id = target_id
            self.last_switch_frame = frame_number
            self.locked_counter = 0
            return best_candidate["box"]

        return None


# Detectors never need full-resolution frames: MediaPipe returns relative
# coords and YOLO boxes are scaled back up. Running them on a ≤640px copy cuts
# per-frame preprocessing cost hard, which is what dominates CPU-only renders.
DETECT_MAX_WIDTH = 640
# The global MediaPipe graph and YOLO model are NOT thread-safe; clips render
# in parallel, so every inference goes through this lock. Contention is small
# (a few ms per call) — the ffmpeg renders are where the parallel time goes.
DETECT_LOCK = threading.Lock()
# Detect every Nth frame; SmoothedCameraman interpolates between updates.
DETECT_STRIDE = max(int(os.environ.get("DETECT_STRIDE", "4")), 1)
# YOLO fallback (no face found) is far heavier than MediaPipe — extra throttle.
YOLO_FALLBACK_STRIDE = DETECT_STRIDE * 2


# Lazy model handles. Importing this module must not pull torch/ultralytics;
# the pure state-machine tests only ever construct the two classes above.
_face_detection = None
_model = None


def _get_face_detection():
    """The project's face detection model.

    BlazeFace (short-range), bundled as ``assets/blaze_face_short_range.tflite``
    and run through MediaPipe's tasks API. Exposes
    ``process(rgb) -> [{'box': [x, y, w, h], 'score': area}, ...]`` in the
    input frame's pixel coordinates. Loaded once on first use; None when the
    model cannot be loaded, in which case detectors report no faces rather
    than crash.
    """
    global _face_detection
    if _face_detection is None:
        _face_detection = _build_face_detector()
    return _face_detection


class _BlazeFaceDetector:
    """MediaPipe tasks FaceDetector over the bundled short-range model."""

    def __init__(self, model_path):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            min_detection_confidence=0.5,
        )
        self._fd = mp_vision.FaceDetector.create_from_options(options)

    def process(self, rgb):
        import mediapipe as mp

        result = self._fd.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        out = []
        for detection in result.detections:
            box = detection.bounding_box
            out.append(
                {
                    "box": [box.origin_x, box.origin_y, box.width, box.height],
                    "score": box.width * box.height,
                }
            )
        return out


def _build_face_detector():
    import os as _os

    model_path = _os.environ.get("FACE_DETECTOR_MODEL") or _os.path.join(
        _os.path.dirname(_os.path.abspath(__file__)),
        "assets",
        "blaze_face_short_range.tflite",
    )
    try:
        return _BlazeFaceDetector(model_path)
    except Exception as e:
        print(f"   ⚠️ Face detection unavailable ({type(e).__name__}) — no camera tracking")
        return None


def _get_model():
    global _model
    if _model is None:
        try:
            from ultralytics import YOLO

            _model = YOLO("yolov8n.pt")
        except Exception as e:
            # YOLO is a heavy optional fallback; a missing package or a failed
            # first download must degrade to face-only tracking, not kill the
            # render. Cache the failure so we do not re-try every frame.
            print(f"   ⚠️ YOLO unavailable ({type(e).__name__}) — face-only tracking")
            _model = False
    return _model if _model is not False else None


def _detection_frame(frame):
    """Downscaled copy for detectors. Returns (small_frame, scale) with
    scale mapping small-frame pixel coords back to the original frame."""
    h, w = frame.shape[:2]
    if w <= DETECT_MAX_WIDTH:
        return frame, 1.0
    scale = w / DETECT_MAX_WIDTH
    small = cv2.resize(
        frame, (DETECT_MAX_WIDTH, max(int(h / scale), 2)), interpolation=cv2.INTER_AREA
    )
    return small, scale


def detect_face_candidates(frame):
    """
    Returns list of all detected faces using lightweight FaceDetection.
    Boxes are in ORIGINAL frame coordinates (detection runs downscaled).
    """
    small, scale = _detection_frame(frame)
    rgb_frame = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    with DETECT_LOCK:
        detector = _get_face_detection()
        detections = detector.process(rgb_frame) if detector else []

    candidates = []
    for det in detections or []:
        x, y, w, h = det["box"]
        candidates.append(
            {
                "box": [int(x * scale), int(y * scale), int(w * scale), int(h * scale)],
                "score": w * scale * h * scale,  # Area as score
            }
        )

    return candidates


def detect_person_yolo(frame):
    """
    Fallback: Detect largest person using YOLO when face detection fails.
    Returns [x, y, w, h] of the person's 'upper body' approximation, in
    ORIGINAL frame coordinates (inference runs on a downscaled copy).
    """
    small, scale = _detection_frame(frame)
    model = _get_model()
    if model is None:
        return None
    with DETECT_LOCK:
        results = model(small, verbose=False, classes=[0])  # class 0 is person

    if not results:
        return None

    best_box = None
    max_area = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = [int(i * scale) for i in box.xyxy[0]]
            w = x2 - x1
            h = y2 - y1
            area = w * h

            if area > max_area:
                max_area = area
                # Focus on the top 40% of the person (head/chest) for framing
                # This approximates where the face is if we can't detect it directly
                face_h = int(h * 0.4)
                best_box = [x1, y1, w, face_h]

    return best_box


def analyze_scenes_strategy(video_path, scenes):
    """
    Analyzes each scene to determine if it should be TRACK (Single person) or GENERAL (Group/Wide).
    Returns list of strategies corresponding to scenes.
    """
    from tqdm import tqdm

    cap = cv2.VideoCapture(video_path)
    strategies = []

    if not cap.isOpened():
        return ["TRACK"] * len(scenes)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    for start, end in tqdm(scenes, desc="   Analyzing Scenes"):
        s_f, e_f = start.get_frames(), end.get_frames()
        # Sample 5 frames spread across the scene, clamped inside it (the old
        # start+5/end-5 samples landed outside scenes shorter than ~10 frames).
        margin = min(2, max(0, (e_f - s_f - 1) // 2))
        frames_to_check = sorted(
            {int(round(f)) for f in np.linspace(s_f + margin, e_f - 1 - margin, 5)}
        )

        face_counts = []
        for f_idx in frames_to_check:
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Near-black frames (fades, cut-to-black) carry no faces and used
            # to drag single-person scenes into GENERAL. Skip them.
            if frame.mean() < 16:
                continue

            candidates = detect_face_candidates(frame)
            face_counts.append(len(candidates))

        if not face_counts:
            avg_faces = 0
        else:
            avg_faces = sum(face_counts) / len(face_counts)

        # Strategy:
        # 0 faces -> GENERAL (Landscape/B-roll)
        # 1 face -> TRACK
        # > 1.2 faces -> GENERAL (Group)

        if avg_faces > 1.2 or avg_faces < 0.5:
            strategies.append("GENERAL")
        else:
            strategies.append("TRACK")

    cap.release()

    # Hysteresis: a short scene whose two neighbors agree on the opposite
    # strategy is almost always a sampling miss (profile face, insert shot).
    # Each TRACK<->GENERAL flip is a full on-screen layout change, so flapping
    # is worse than an occasional wrong-but-stable choice.
    max_flip_frames = int(2.0 * fps)
    for i in range(1, len(strategies) - 1):
        dur = scenes[i][1].get_frames() - scenes[i][0].get_frames()
        if dur < max_flip_frames and strategies[i - 1] == strategies[i + 1] != strategies[i]:
            strategies[i] = strategies[i - 1]

    return strategies


def detect_scenes(video_path):
    from engine.reframe import scene_detection

    return scene_detection.detect_scenes(video_path)


def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise OSError(f"Could not open video file {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

