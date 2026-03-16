"""
Core camera alignment logic for FaceBlender.

Detects 68 facial landmarks in a reference image, matches them to 3D vertex
positions on the loaded head mesh, solves the PnP problem with OpenCV, and
returns the camera extrinsics together with an estimated focal length.
"""

import numpy as np

from . import landmark_mapping as lm_module

# ---------------------------------------------------------------------------
# Optional dependency helpers
# ---------------------------------------------------------------------------

def _require_cv2():
    try:
        import cv2
        return cv2
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required for camera alignment.  "
            "Install it with: pip install opencv-python"
        ) from exc


def _require_mediapipe():
    try:
        import mediapipe as mp
        return mp
    except ImportError as exc:
        raise ImportError(
            "mediapipe is required for face landmark detection.  "
            "Install it with: pip install mediapipe"
        ) from exc


def _try_dlib():
    """Return dlib if installed, otherwise None."""
    try:
        import dlib
        return dlib
    except ImportError:
        return None


def get_image_size(image_path: str) -> tuple[int, int]:
    """Return ``(width, height)`` for an image using OpenCV."""
    cv2 = _require_cv2()
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_height, image_width = image_bgr.shape[:2]
    return image_width, image_height


# ---------------------------------------------------------------------------
# Landmark detection
# ---------------------------------------------------------------------------

def detect_landmarks_mediapipe(image_path: str) -> tuple[list[int], list]:
    """
    Detect 68 facial landmarks using MediaPipe Face Mesh and remap them to the
    standard 68-point dlib convention.

    Args:
        image_path: Absolute path to the reference photo.

    Returns:
        tuple: ``(landmark_indices, points_2d)`` where *landmark_indices* are
        integers 0–67 and *points_2d* is a list of ``[x, y]`` pixel coords.

    Raises:
        RuntimeError: If no face is detected in the image.
    """
    cv2 = _require_cv2()
    mp = _require_mediapipe()

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        raise RuntimeError("No face detected in the image.")

    face_lms = results.multi_face_landmarks[0].landmark

    # Mapping from the 68 dlib landmark indices to MediaPipe face mesh indices.
    # Source: https://github.com/nicehorse06/mediapipe-face-landmark-68-to-mediapipe
    DLIB_TO_MP = [
        # Jaw line (0-16)
        162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
        # Left eyebrow (17-21)
        70, 63, 105, 66, 107,
        # Right eyebrow (22-26)
        336, 296, 334, 293, 301,
        # Nose bridge (27-30)
        168, 6, 197, 195,
        # Nose bottom (31-35)
        5, 4, 75, 97, 2,
        # Left eye (36-41)
        33, 160, 158, 133, 153, 144,
        # Right eye (42-47)
        362, 385, 387, 263, 373, 380,
        # Outer lip (48-59)
        61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91,
        # Inner lip (60-67)
        78, 82, 13, 312, 308, 317, 14, 87,
    ]

    landmark_indices = list(range(68))
    points_2d = []
    for mp_idx in DLIB_TO_MP:
        lm = face_lms[mp_idx]
        points_2d.append([lm.x * w, lm.y * h])

    return landmark_indices, points_2d


def detect_landmarks_dlib(image_path: str, predictor_path: str) -> tuple[list[int], list]:
    """
    Detect 68 facial landmarks using dlib's shape predictor.

    Args:
        image_path: Absolute path to the reference photo.
        predictor_path: Path to dlib's ``shape_predictor_68_face_landmarks.dat``.

    Returns:
        tuple: ``(landmark_indices, points_2d)``

    Raises:
        RuntimeError: If no face is detected in the image.
    """
    dlib = _try_dlib()
    if dlib is None:
        raise ImportError(
            "dlib is not installed. Install it with: pip install dlib  "
            "(or use the mediapipe backend instead)"
        )
    cv2 = _require_cv2()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    faces = detector(image_rgb, 1)
    if len(faces) == 0:
        raise RuntimeError("No face detected in the image.")

    shape = predictor(image_rgb, faces[0])
    points_2d = [[shape.part(i).x, shape.part(i).y] for i in range(68)]
    return list(range(68)), points_2d


def detect_landmarks(
    image_path: str,
    backend: str = "mediapipe",
    dlib_predictor_path: str | None = None,
) -> tuple[list[int], list]:
    """
    Detect 68 facial landmarks using the specified backend.

    Args:
        image_path: Absolute path to the reference photo.
        backend: ``"mediapipe"`` (default) or ``"dlib"``.
        dlib_predictor_path: Required when *backend* is ``"dlib"``.

    Returns:
        tuple: ``(landmark_indices, points_2d)``
    """
    if backend == "dlib":
        if dlib_predictor_path is None:
            raise ValueError("dlib_predictor_path must be specified when using the dlib backend.")
        return detect_landmarks_dlib(image_path, dlib_predictor_path)
    return detect_landmarks_mediapipe(image_path)


# ---------------------------------------------------------------------------
# PnP solving
# ---------------------------------------------------------------------------

def solve_pnp(
    points_3d: list,
    points_2d: list,
    image_width: int,
    image_height: int,
    focal_length_px: float | None = None,
) -> tuple:
    """
    Estimate camera extrinsics (R, t) from 2D–3D point correspondences.

    Uses OpenCV's ``solvePnP`` with the ITERATIVE flag.

    Args:
        points_3d: List of ``[x, y, z]`` 3D world-space coordinates.
        points_2d: List of ``[x, y]`` 2D image-space coordinates (pixels).
        image_width: Image width in pixels.
        image_height: Image height in pixels.
        focal_length_px: Focal length in pixels.  When *None* a heuristic
            estimate (``max(w, h) * 1.2``) is used.

    Returns:
        tuple: ``(R, t, focal_length_px)``
            - ``R`` – 3×3 rotation matrix (np.ndarray)
            - ``t`` – 3×1 translation vector (np.ndarray)
            - ``focal_length_px`` – focal length used for the solve (float)

    Raises:
        RuntimeError: If OpenCV's ``solvePnP`` fails.
        ValueError: If fewer than 6 point correspondences are provided.
    """
    cv2 = _require_cv2()

    if len(points_3d) < 6 or len(points_2d) < 6:
        raise ValueError(
            f"At least 6 point correspondences are required, got {len(points_3d)}."
        )

    pts3d = np.array(points_3d, dtype=np.float64)
    pts2d = np.array(points_2d, dtype=np.float64)

    if focal_length_px is None:
        focal_length_px = max(image_width, image_height) * 1.2

    cx = image_width / 2.0
    cy = image_height / 2.0
    K = np.array([
        [focal_length_px, 0, cx],
        [0, focal_length_px, cy],
        [0, 0, 1],
    ], dtype=np.float64)

    dist_coeffs = np.zeros(4, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        pts3d,
        pts2d,
        K,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("solvePnP failed to find a solution.")

    R, _ = cv2.Rodrigues(rvec)
    return R, tvec, focal_length_px


# ---------------------------------------------------------------------------
# High-level alignment entry point
# ---------------------------------------------------------------------------

def align_camera(
    image_path: str,
    mesh_obj,
    lm_mapping: dict,
    backend: str = "mediapipe",
    dlib_predictor_path: str | None = None,
    focal_length_px: float | None = None,
) -> tuple:
    """
    Full camera alignment pipeline for a single reference image.

    1. Detects 68 facial landmarks in the image.
    2. Retrieves the corresponding 3D vertex positions from *mesh_obj*.
    3. Runs solvePnP to estimate camera extrinsics.

    Args:
        image_path: Absolute path to the reference photo.
        mesh_obj: Blender ``bpy.types.Object`` (mesh) representing the head.
        lm_mapping: ``{landmark_index: vertex_index}`` dict.
        backend: Landmark detection backend (``"mediapipe"`` or ``"dlib"``).
        dlib_predictor_path: Path to dlib predictor dat file (dlib backend only).
        focal_length_px: Optional focal length override in pixels.

    Returns:
        tuple: ``(R, t, focal_length_px, image_width, image_height)``
    """
    image_width, image_height = get_image_size(image_path)

    # Step 1: detect 2D landmarks
    detected_lm_indices, points_2d = detect_landmarks(
        image_path, backend=backend, dlib_predictor_path=dlib_predictor_path
    )

    # Step 2: get 3D positions for the same landmark indices
    available_lm_indices, points_3d = lm_module.get_3d_landmarks(mesh_obj, lm_mapping)

    # Intersect detected and available landmarks to get consistent pairs
    available_set = set(available_lm_indices)
    detected_set = set(detected_lm_indices)
    common_indices = sorted(available_set & detected_set)

    if len(common_indices) < 6:
        raise RuntimeError(
            f"Only {len(common_indices)} common landmarks between detected "
            "and available – at least 6 are needed."
        )

    detected_lookup = {idx: pt for idx, pt in zip(detected_lm_indices, points_2d)}
    available_lookup = {idx: pt for idx, pt in zip(available_lm_indices, points_3d)}

    pts_2d_final = [detected_lookup[i] for i in common_indices]
    pts_3d_final = [available_lookup[i] for i in common_indices]

    # Step 3: solve PnP
    R, t, fl_px = solve_pnp(
        pts_3d_final,
        pts_2d_final,
        image_width,
        image_height,
        focal_length_px=focal_length_px,
    )

    return R, t, fl_px, image_width, image_height
