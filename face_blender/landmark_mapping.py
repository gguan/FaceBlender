"""
Landmark mapping utilities for FaceBlender.

Provides functions to load and query mappings between 68 dlib facial landmark
indices and 3D mesh vertex indices (default: FLAME topology).
"""

import json
import os

# Path to the built-in FLAME landmark mapping JSON file
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_DEFAULT_MAPPING_FILE = os.path.join(_DATA_DIR, "flame_landmark_mapping.json")

# Module-level cache so the file is only read once per session
_mapping_cache: dict | None = None


def _validate_mapping(mapping: dict[int, int]) -> None:
    """Validate landmark-to-vertex mapping entries before use."""
    negative_vertices = {lm_idx: v_idx for lm_idx, v_idx in mapping.items() if v_idx < 0}
    if negative_vertices:
        raise ValueError(
            "Landmark mapping contains negative vertex indices: "
            f"{negative_vertices}"
        )


def load_mapping(json_path: str | None = None) -> dict[int, int]:
    """
    Load a landmark-to-vertex mapping from a JSON file.

    The file must contain a top-level ``"mapping"`` key whose value is an
    object mapping string landmark indices to integer vertex indices, e.g.::

        { "mapping": { "0": 1061, "1": 1177, ... } }

    Args:
        json_path: Path to a custom JSON mapping file.  When *None* the
            built-in FLAME mapping is used.

    Returns:
        dict: ``{landmark_index (int): vertex_index (int)}``

    Raises:
        FileNotFoundError: If *json_path* does not point to an existing file.
        KeyError: If the JSON file does not contain a ``"mapping"`` key.
    """
    global _mapping_cache

    if json_path is None:
        # Return cached built-in mapping when available
        if _mapping_cache is not None:
            return _mapping_cache
        json_path = _DEFAULT_MAPPING_FILE

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Landmark mapping file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if "mapping" not in data:
        raise KeyError(f"JSON file '{json_path}' does not contain a 'mapping' key.")

    mapping = {int(k): int(v) for k, v in data["mapping"].items()}
    _validate_mapping(mapping)

    # Cache the built-in mapping
    if json_path == _DEFAULT_MAPPING_FILE:
        _mapping_cache = mapping

    return mapping


def clear_cache() -> None:
    """Invalidate the built-in mapping cache.

    Call this from the add-on's ``unregister()`` so that a fresh load occurs
    when the add-on is re-enabled (e.g. after the mapping file is updated).
    """
    global _mapping_cache
    _mapping_cache = None


def get_vertex_indices(landmark_indices: list[int], mapping: dict[int, int]) -> list[int]:
    """
    Translate a list of landmark indices into mesh vertex indices.

    Args:
        landmark_indices: 68-point landmark indices (0–67).
        mapping: Landmark-to-vertex mapping as returned by :func:`load_mapping`.

    Returns:
        list[int]: Vertex indices corresponding to *landmark_indices*.

    Raises:
        KeyError: If any landmark index is absent from *mapping*.
    """
    missing = [i for i in landmark_indices if i not in mapping]
    if missing:
        raise KeyError(f"Landmark indices not found in mapping: {missing}")
    return [mapping[i] for i in landmark_indices]


def get_3d_landmarks(mesh_obj, mapping: dict[int, int]) -> tuple[list[int], list[list[float]]]:
    """
    Extract 3D positions of landmark vertices from a Blender mesh object.

    Applies the object's world matrix so the positions are in world space.

    Args:
        mesh_obj: A Blender ``bpy.types.Object`` with ``type == 'MESH'``.
        mapping: Landmark-to-vertex mapping as returned by :func:`load_mapping`.

    Returns:
        tuple: ``(landmark_indices, points_3d)`` where *landmark_indices* is a
        sorted list of landmark indices present in *mapping* and *points_3d* is
        a list of ``[x, y, z]`` world-space coordinates.
    """
    vertices = mesh_obj.data.vertices
    world_matrix = mesh_obj.matrix_world
    max_vertex = len(vertices) - 1

    landmark_indices = []
    points_3d = []

    for lm_idx in sorted(mapping.keys()):
        v_idx = mapping[lm_idx]
        if v_idx < 0:
            raise ValueError(
                f"Landmark {lm_idx} maps to negative vertex index {v_idx}."
            )
        if v_idx > max_vertex:
            # Skip landmarks whose vertex index exceeds the mesh's vertex count
            continue
        co_local = vertices[v_idx].co
        co_world = world_matrix @ co_local
        landmark_indices.append(lm_idx)
        points_3d.append([co_world.x, co_world.y, co_world.z])

    return landmark_indices, points_3d
