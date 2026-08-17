def time_to_seconds(t: str | int | float | None) -> int:
    """Convert 'mm:ss' or 'hh:mm:ss' (or int/float seconds) to integer seconds."""
    if t is None:
        return 0
    if isinstance(t, (int, float)):
        return int(t)
    parts = [int(p) for p in str(t).split(":")]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]

