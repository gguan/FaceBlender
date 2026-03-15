"""
UI panels and UIList definitions for FaceBlender.

Defines:
  - FACEBLENDER_UL_image_list  – UIList item rendering
  - FACEBLENDER_PT_main_panel  – the main N-panel tab
  - FaceBlenderImageItem       – CollectionProperty item
  - FaceBlenderProperties      – scene-level add-on properties
"""

import bpy
from bpy.props import (
    StringProperty,
    CollectionProperty,
    IntProperty,
    PointerProperty,
    EnumProperty,
)
from bpy.types import PropertyGroup


# ---------------------------------------------------------------------------
# Data properties
# ---------------------------------------------------------------------------

class FaceBlenderImageItem(PropertyGroup):
    """A single entry in the reference-image list."""

    name: StringProperty(
        name="Name",
        description="Display name of the reference image",
        default="",
    )
    filepath: StringProperty(
        name="File Path",
        description="Absolute path to the reference image file",
        default="",
        subtype="FILE_PATH",
    )


class FaceBlenderProperties(PropertyGroup):
    """Scene-level properties for the FaceBlender add-on."""

    head_object: PointerProperty(
        name="Head Mesh",
        description="The 3D head mesh object to align the camera to",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == "MESH",
    )

    image_list: CollectionProperty(
        name="Reference Images",
        description="List of reference photos for camera alignment",
        type=FaceBlenderImageItem,
    )

    image_list_index: IntProperty(
        name="Active Image Index",
        description="Currently selected image in the list",
        default=0,
    )

    landmark_backend: EnumProperty(
        name="Landmark Backend",
        description="Library used for facial landmark detection",
        items=[
            ("mediapipe", "MediaPipe", "Use Google MediaPipe (recommended, easier to install)"),
            ("dlib", "dlib", "Use dlib shape predictor (requires .dat model file)"),
        ],
        default="mediapipe",
    )

    dlib_predictor_path: StringProperty(
        name="dlib Predictor",
        description=(
            "Path to dlib's shape_predictor_68_face_landmarks.dat "
            "(only required when using the dlib backend)"
        ),
        default="",
        subtype="FILE_PATH",
    )

    custom_landmark_mapping: StringProperty(
        name="Custom Mapping",
        description=(
            "Optional path to a custom landmark-to-vertex mapping JSON file. "
            "Leave empty to use the built-in FLAME mapping."
        ),
        default="",
        subtype="FILE_PATH",
    )


# ---------------------------------------------------------------------------
# UIList
# ---------------------------------------------------------------------------

class FACEBLENDER_UL_image_list(bpy.types.UIList):
    """Renders each reference image as a row with a name and a camera-align button."""

    bl_idname = "FACEBLENDER_UL_image_list"

    def draw_item(self, context, layout, data, item, icon, active_data, active_propname, index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            row = layout.row(align=True)
            # Thumbnail icon – use IMAGE if Blender can preview it, else OUTLINER_OB_IMAGE
            row.label(text=item.name, icon="IMAGE_DATA")
            # Camera-align button for this specific image
            op = row.operator(
                "faceblender.align_camera",
                text="",
                icon="CAMERA_DATA",
                emboss=True,
            )
            op.image_index = index
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="", icon="IMAGE_DATA")


# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------

class FACEBLENDER_PT_main_panel(bpy.types.Panel):
    """FaceBlender side panel in the 3D Viewport N-panel."""

    bl_label = "FaceBlender"
    bl_idname = "FACEBLENDER_PT_main_panel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "FaceBlender"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        props = scene.faceblender

        # ---- Head Mesh section ----
        box = layout.box()
        box.label(text="Head Mesh", icon="MESH_MONKEY")
        row = box.row(align=True)
        row.prop(props, "head_object", text="")
        row.operator("faceblender.import_head", text="", icon="IMPORT")

        # ---- Reference Images section ----
        box = layout.box()
        box.label(text="Reference Images", icon="IMAGE_DATA")

        row = box.row()
        row.template_list(
            "FACEBLENDER_UL_image_list",
            "",
            props,
            "image_list",
            props,
            "image_list_index",
            rows=4,
        )

        col = row.column(align=True)
        col.operator("faceblender.add_image", text="", icon="ADD")
        col.operator("faceblender.remove_image", text="", icon="REMOVE")

        # ---- Advanced / Settings section ----
        box = layout.box()
        col = box.column()
        col.label(text="Settings", icon="PREFERENCES")
        col.prop(props, "landmark_backend", text="Backend")

        if props.landmark_backend == "dlib":
            col.prop(props, "dlib_predictor_path", text="Predictor")

        col.prop(props, "custom_landmark_mapping", text="Custom Mapping")


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

UI_CLASSES = [
    FaceBlenderImageItem,
    FaceBlenderProperties,
    FACEBLENDER_UL_image_list,
    FACEBLENDER_PT_main_panel,
]


def register():
    for cls in UI_CLASSES:
        bpy.utils.register_class(cls)

    bpy.types.Scene.faceblender = PointerProperty(type=FaceBlenderProperties)


def unregister():
    del bpy.types.Scene.faceblender

    for cls in reversed(UI_CLASSES):
        bpy.utils.unregister_class(cls)
