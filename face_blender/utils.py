"""
Utility functions for FaceBlender.
Handles coordinate system conversions between OpenCV and Blender (OpenGL).
"""

import numpy as np
import mathutils


def opencv_to_blender(R, t):
    """
    Convert OpenCV camera extrinsics (R, t) to a Blender camera world matrix.

    OpenCV coordinate system: right-handed, Y-down, Z-forward (into scene)
    Blender/OpenGL coordinate system: right-handed, Y-up, Z-backward (out of screen)

    The conversion follows the pixel3dmm approach:
      1. Build the 4x4 extrinsic matrix [R | t]
      2. Flip Y and Z axes to convert from OpenCV to OpenGL
      3. Invert to get the camera-to-world matrix

    Args:
        R (np.ndarray): 3x3 rotation matrix from OpenCV solvePnP
        t (np.ndarray): 3x1 translation vector from OpenCV solvePnP

    Returns:
        mathutils.Matrix: 4x4 Blender camera world matrix
    """
    # Build the 4×4 world-to-camera matrix in OpenCV convention
    Rt = np.eye(4, dtype=np.float64)
    Rt[:3, :3] = R
    Rt[:3, 3] = t.flatten()

    # Flip Y and Z to convert from OpenCV (Y-down, Z-forward) to OpenGL/Blender (Y-up, Z-backward)
    Rt[1] *= -1
    Rt[2] *= -1

    # Invert to obtain camera-to-world (i.e., camera matrix_world)
    Rt_inv = np.linalg.inv(Rt)

    # Convert to Blender mathutils.Matrix
    return mathutils.Matrix(Rt_inv.tolist())


def focal_length_px_to_mm(focal_length_px, sensor_width_mm, image_width_px):
    """
    Convert focal length from pixels to millimetres using the sensor width.

    Args:
        focal_length_px (float): Focal length in pixels.
        sensor_width_mm (float): Camera sensor width in mm (Blender default is 36 mm).
        image_width_px (int): Image width in pixels.

    Returns:
        float: Focal length in mm.
    """
    return focal_length_px * sensor_width_mm / image_width_px


def estimate_focal_length(image_width, image_height, scale=1.2):
    """
    Estimate a reasonable default focal length in pixels.

    A common heuristic is focal_length ≈ image_width * scale.

    Args:
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.
        scale (float): Scaling factor (default 1.2 is a reasonable approximation).

    Returns:
        float: Estimated focal length in pixels.
    """
    return max(image_width, image_height) * scale


def build_intrinsic_matrix(focal_length_px, cx, cy):
    """
    Build a 3×3 camera intrinsic matrix K.

    Args:
        focal_length_px (float): Focal length in pixels.
        cx (float): Principal point x (usually image_width / 2).
        cy (float): Principal point y (usually image_height / 2).

    Returns:
        np.ndarray: 3×3 intrinsic matrix.
    """
    return np.array([
        [focal_length_px, 0, cx],
        [0, focal_length_px, cy],
        [0, 0, 1]
    ], dtype=np.float64)


def compute_shift(cx, cy, image_width, image_height):
    """
    Compute Blender camera shift values from principal point offset.

    In Blender's ``SENSOR_FIT_AUTO`` mode (the default), both ``shift_x`` and
    ``shift_y`` are expressed as a fraction of the **sensor width**, which maps
    to ``image_width`` pixels.  Concretely:

    * ``shift_x = (cx - image_width / 2) / image_width``
    * ``shift_y = -(cy - image_height / 2) / image_width``

    The Y axis is negated because Blender's Y points upward while image Y
    points downward.

    This function assumes ``SENSOR_FIT_AUTO`` with ``image_width >= image_height``
    (landscape or square).  For portrait images (``image_height > image_width``)
    Blender switches the reference dimension to ``image_height``; in that case
    both shift values should be divided by ``image_height`` instead.  Since the
    current pipeline always passes the principal point at the image centre
    (``cx = image_width / 2``, ``cy = image_height / 2``), the resulting shift
    is always zero and the orientation does not affect the output.

    Args:
        cx (float): Principal point x in pixels.
        cy (float): Principal point y in pixels.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.

    Returns:
        tuple[float, float]: (shift_x, shift_y) in Blender normalised units.
    """
    # Offset from the image centre
    offset_x = cx - image_width / 2.0
    offset_y = cy - image_height / 2.0

    # Normalise by image_width — Blender uses sensor_width as the reference
    # dimension for both axes in SENSOR_FIT_AUTO mode.
    shift_x = offset_x / image_width
    shift_y = -offset_y / image_width  # Blender Y is flipped relative to image Y

    return shift_x, shift_y
