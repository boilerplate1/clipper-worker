"""Reframe engine v2, ported from the OpenShorts project.

An alternative verticalization engine: analyze in Python, render natively in
ffmpeg. Pure helpers (sendcmd/concat generation, scene slicing, layout
filtergraphs) stay unit-testable without torch / ultralytics / Gemini.
"""

