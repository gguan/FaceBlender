"""
Blender operators for FaceBlender.

Defines all bpy.types.Operator subclasses used by the add-on:
  - FACEBLENDER_OT_import_head      – import an OBJ as the head mesh
  - FACEBLENDER_OT_add_image        – add a reference photo to the list
  - FACEBLENDER_OT_remove_image     – remove the selected photo
  - FACEBLENDER_OT_align_camera     – align the active camera to a reference photo
"""

import os
import bpy
from bpy.props import StringProperty, IntProperty
from bpy_extras.io_utils import ImportHelper


# ---------------------------------------------------------------------------
# Import head mesh
# ---------------------------------------------------------------------------

class FACEBLENDER_OT_import_head(bpy.types.Operator, ImportHelper):
    """Import an OBJ file as the head mesh"""

    bl_idname = "faceblender.import_head"
    bl_label = "Import OBJ Head Mesh"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ".obj"
    filter_glob: StringProperty(default="*.obj", options={"HIDDEN"})

    def execute(self, context):
        scene = context.scene
        props = scene.faceblender

        filepath = self.filepath
        if not os.path.isfile(filepath):
            self.report({"ERROR"}, f"File not found: {filepath}")
            return {"CANCELLED"}

        # Deselect everything first
        bpy.ops.object.select_all(action="DESELECT")

        # Import OBJ — Blender 4.x uses wm.obj_import; earlier versions use import_scene.obj
        if bpy.app.version >= (4, 0, 0):
            bpy.ops.wm.obj_import(filepath=filepath)
        else:
            bpy.ops.import_scene.obj(filepath=filepath)

        imported = [o for o in context.selected_objects if o.type == "MESH"]
        if not imported:
            self.report({"ERROR"}, "No mesh objects were imported from the OBJ file.")
            return {"CANCELLED"}

        # Assign the first imported mesh as the head object
        props.head_object = imported[0]
        self.report({"INFO"}, f"Imported head mesh: {imported[0].name}")
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Add / Remove reference images
# ---------------------------------------------------------------------------

class FACEBLENDER_OT_add_image(bpy.types.Operator, ImportHelper):
    """Add a reference photo to the image list"""

    bl_idname = "faceblender.add_image"
    bl_label = "Add Reference Image"
    bl_options = {"REGISTER", "UNDO"}

    filename_ext = ""
    filter_glob: StringProperty(
        default="*.jpg;*.jpeg;*.png;*.bmp;*.tiff;*.tif;*.webp",
        options={"HIDDEN"},
    )

    def execute(self, context):
        props = context.scene.faceblender

        filepath = self.filepath
        if not os.path.isfile(filepath):
            self.report({"ERROR"}, f"File not found: {filepath}")
            return {"CANCELLED"}

        # Avoid duplicates
        for item in props.image_list:
            if item.filepath == filepath:
                self.report({"WARNING"}, "Image already in the list.")
                return {"CANCELLED"}

        item = props.image_list.add()
        item.filepath = filepath
        item.name = os.path.basename(filepath)

        # Select the newly added item
        props.image_list_index = len(props.image_list) - 1

        self.report({"INFO"}, f"Added image: {item.name}")
        return {"FINISHED"}


class FACEBLENDER_OT_remove_image(bpy.types.Operator):
    """Remove the selected reference photo from the list"""

    bl_idname = "faceblender.remove_image"
    bl_label = "Remove Reference Image"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        props = context.scene.faceblender
        index = props.image_list_index

        if index < 0 or index >= len(props.image_list):
            self.report({"WARNING"}, "No image selected.")
            return {"CANCELLED"}

        props.image_list.remove(index)
        props.image_list_index = max(0, index - 1)
        return {"FINISHED"}


# ---------------------------------------------------------------------------
# Camera alignment
# ---------------------------------------------------------------------------

class FACEBLENDER_OT_align_camera(bpy.types.Operator):
    """Align the active camera to the specified reference photo"""

    bl_idname = "faceblender.align_camera"
    bl_label = "Align Camera to Image"
    bl_options = {"REGISTER", "UNDO"}

    image_index: IntProperty(
        name="Image Index",
        description="Index of the reference image in the image list",
        default=0,
    )

    def execute(self, context):
        scene = context.scene
        props = scene.faceblender

        # ---- Validate prerequisites ----------------------------------------
        if props.head_object is None:
            self.report({"ERROR"}, "No head mesh selected. Please set the Head Mesh first.")
            return {"CANCELLED"}

        if props.head_object.type != "MESH":
            self.report({"ERROR"}, "The selected head object is not a mesh.")
            return {"CANCELLED"}

        cam_obj = scene.camera
        if cam_obj is None:
            self.report({"ERROR"}, "No active camera in the scene. Please add a camera first.")
            return {"CANCELLED"}

        if self.image_index < 0 or self.image_index >= len(props.image_list):
            self.report({"ERROR"}, "Invalid image index.")
            return {"CANCELLED"}

        image_item = props.image_list[self.image_index]
        image_path = image_item.filepath

        if not os.path.isfile(image_path):
            self.report({"ERROR"}, f"Image file not found: {image_path}")
            return {"CANCELLED"}

        # ---- Load landmark mapping ------------------------------------------
        from .landmark_mapping import load_mapping, get_3d_landmarks
        from .camera_alignment import align_camera
        from .utils import opencv_to_blender, focal_length_px_to_mm, compute_shift

        custom_mapping_path = props.custom_landmark_mapping.strip() or None
        try:
            mapping = load_mapping(custom_mapping_path)
        except (FileNotFoundError, KeyError) as err:
            self.report({"ERROR"}, str(err))
            return {"CANCELLED"}

        # ---- Run alignment --------------------------------------------------
        backend = props.landmark_backend
        dlib_path = props.dlib_predictor_path.strip() or None

        try:
            R, t, focal_px, img_w, img_h = align_camera(
                image_path=image_path,
                mesh_obj=props.head_object,
                landmark_mapping=mapping,
                backend=backend,
                dlib_predictor_path=dlib_path,
            )
        except ImportError as err:
            self.report({"ERROR"}, f"Missing dependency: {err}")
            return {"CANCELLED"}
        except FileNotFoundError as err:
            self.report({"ERROR"}, str(err))
            return {"CANCELLED"}
        except RuntimeError as err:
            self.report({"ERROR"}, str(err))
            return {"CANCELLED"}
        except Exception as err:  # pylint: disable=broad-except
            self.report({"ERROR"}, f"Camera alignment failed: {err}")
            return {"CANCELLED"}

        # ---- Apply extrinsics to camera ------------------------------------
        world_matrix = opencv_to_blender(R, t)
        cam_obj.matrix_world = world_matrix

        # ---- Apply intrinsics to camera ------------------------------------
        cam_data = cam_obj.data
        sensor_width_mm = cam_data.sensor_width  # Blender default 36 mm

        focal_mm = focal_length_px_to_mm(focal_px, sensor_width_mm, img_w)
        cam_data.lens = focal_mm
        cam_data.sensor_fit = "AUTO"

        # Set render resolution to match the reference image so the camera
        # frustum aspect ratio and background image display correctly.
        scene.render.resolution_x = img_w
        scene.render.resolution_y = img_h

        # Principal point shift (only if not centred)
        cx = img_w / 2.0
        cy = img_h / 2.0
        shift_x, shift_y = compute_shift(cx, cy, img_w, img_h)
        cam_data.shift_x = shift_x
        cam_data.shift_y = shift_y

        # ---- Set background image ------------------------------------------
        self._set_background_image(cam_data, image_path, img_w, img_h)

        self.report({"INFO"}, f"Camera aligned to {image_item.name}")
        return {"FINISHED"}

    # ------------------------------------------------------------------
    def _set_background_image(self, cam_data, image_path, img_w, img_h):
        """Assign *image_path* as the camera's background image."""
        cam_data.show_background_images = True

        # Load (or reuse) the image in Blender's data-block list
        existing = bpy.data.images.get(os.path.basename(image_path))
        if existing and os.path.abspath(existing.filepath) == os.path.abspath(image_path):
            bl_image = existing
        else:
            bl_image = bpy.data.images.load(image_path, check_existing=True)

        # Check whether this image already has a background slot
        for bg in cam_data.background_images:
            if bg.image and os.path.abspath(bg.image.filepath) == os.path.abspath(image_path):
                bg.alpha = 0.5
                bg.display_depth = "BACK"
                bg.frame_method = "FIT"
                return

        # Otherwise add a new background image slot
        bg = cam_data.background_images.new()
        bg.image = bl_image
        bg.alpha = 0.5
        bg.display_depth = "BACK"
        bg.frame_method = "FIT"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

OPERATORS = [
    FACEBLENDER_OT_import_head,
    FACEBLENDER_OT_add_image,
    FACEBLENDER_OT_remove_image,
    FACEBLENDER_OT_align_camera,
]


def register():
    for cls in OPERATORS:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(OPERATORS):
        bpy.utils.unregister_class(cls)
