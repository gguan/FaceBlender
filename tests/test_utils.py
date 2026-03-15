import importlib
import sys
import types
import unittest


class UtilsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.modules.setdefault("mathutils", types.SimpleNamespace(Matrix=lambda rows: rows))
        cls.utils = importlib.import_module("face_blender.utils")

    def test_focal_length_round_trip_for_auto_landscape(self):
        focal_px = self.utils.focal_length_mm_to_px(
            focal_length_mm=50.0,
            sensor_width_mm=36.0,
            sensor_height_mm=24.0,
            image_width_px=1920,
            image_height_px=1080,
            sensor_fit="AUTO",
        )

        focal_mm = self.utils.focal_length_px_to_mm(
            focal_length_px=focal_px,
            sensor_width_mm=36.0,
            sensor_height_mm=24.0,
            image_width_px=1920,
            image_height_px=1080,
            sensor_fit="AUTO",
        )

        self.assertAlmostEqual(focal_mm, 50.0)

    def test_focal_length_round_trip_for_auto_portrait(self):
        focal_px = self.utils.focal_length_mm_to_px(
            focal_length_mm=50.0,
            sensor_width_mm=36.0,
            sensor_height_mm=24.0,
            image_width_px=1080,
            image_height_px=1920,
            sensor_fit="AUTO",
        )

        focal_mm = self.utils.focal_length_px_to_mm(
            focal_length_px=focal_px,
            sensor_width_mm=36.0,
            sensor_height_mm=24.0,
            image_width_px=1080,
            image_height_px=1920,
            sensor_fit="AUTO",
        )

        self.assertAlmostEqual(focal_mm, 50.0)


if __name__ == "__main__":
    unittest.main()
