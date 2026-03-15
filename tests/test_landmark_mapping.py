import tempfile
import unittest

from face_blender import landmark_mapping


class _DummyCoord:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class _DummyVertex:
    def __init__(self, x, y, z):
        self.co = _DummyCoord(x, y, z)


class _DummyMeshData:
    def __init__(self, vertices):
        self.vertices = vertices


class _DummyMatrix:
    def __matmul__(self, other):
        return other


class _DummyMeshObject:
    def __init__(self, vertices):
        self.data = _DummyMeshData(vertices)
        self.matrix_world = _DummyMatrix()


class LandmarkMappingTests(unittest.TestCase):
    def test_load_mapping_rejects_negative_vertex_indices(self):
        with tempfile.NamedTemporaryFile("w+", suffix=".json", encoding="utf-8") as handle:
            handle.write('{"mapping": {"0": -1, "1": 2}}')
            handle.flush()

            with self.assertRaises(ValueError):
                landmark_mapping.load_mapping(handle.name)

    def test_get_3d_landmarks_rejects_negative_vertex_indices(self):
        mesh = _DummyMeshObject(
            [
                _DummyVertex(0.0, 0.0, 0.0),
                _DummyVertex(1.0, 0.0, 0.0),
            ]
        )

        with self.assertRaises(ValueError):
            landmark_mapping.get_3d_landmarks(mesh, {0: -1})


if __name__ == "__main__":
    unittest.main()
