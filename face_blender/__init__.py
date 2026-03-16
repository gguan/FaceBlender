"""
FaceBlender – Blender Add-on for Photo-to-3D Head Camera Alignment
===================================================================

Inspired by pixel3dmm's accurate camera alignment and landmark functionality,
FaceBlender aligns a Blender camera to a 3D head mesh based on one or more
reference photos.

bl_info fields follow the Blender add-on specification.
"""

bl_info = {
    "name": "FaceBlender",
    "author": "FaceBlender Contributors",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > FaceBlender",
    "description": (
        "Align a Blender camera to a 3D head mesh using reference photos. "
        "Detects facial landmarks, solves PnP, and sets camera parameters automatically."
    ),
    "warning": "Requires opencv-python and mediapipe (or dlib). See README for install instructions.",
    "doc_url": "https://github.com/gguan/FaceBlender",
    "category": "3D View",
}


def register():
    from . import panels, operators

    panels.register()
    operators.register()


def unregister():
    from . import panels, operators, landmark_mapping

    operators.unregister()
    panels.unregister()
    landmark_mapping.clear_cache()


if __name__ == "__main__":
    register()
