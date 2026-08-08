"""Pixel-to-real-world scale calibration.

No sensors and no auto-detection of reference objects (stumps, pitch
markings) yet - the user measures one known-length reference visible in
their own frame (e.g. stump height = 71.1cm) and supplies both numbers.
"""

STUMP_HEIGHT_CM = 71.1


def calibrate_scale(pixel_length: float, real_world_cm: float) -> float:
    """
    Given a reference length measured in pixels and its true real-world
    length in cm, return the scale factor in cm-per-pixel.
    """
    if pixel_length <= 0 or real_world_cm <= 0:
        raise ValueError("pixel_length and real_world_cm must both be positive.")
    return real_world_cm / pixel_length
