# FaceBlender

A Blender add-on that automatically aligns a camera to a 3D head mesh using
reference photos. Inspired by [pixel3dmm](https://github.com/soubhikmandal/pixel3dmm)'s
accurate camera alignment and facial landmark functionality.

---

## What it does

1. You import a base head OBJ mesh into Blender.
2. You add one or more reference photos to the image list.
3. For each photo, clicking the **camera icon** next to the image automatically:
   - Detects 68 facial landmarks in the photo (via MediaPipe or dlib).
   - Maps those landmarks to 3D vertex positions on your head mesh.
   - Runs OpenCV's `solvePnP` to estimate camera extrinsics using the active
     camera focal length as the solve input (falling back to a heuristic only
     if the camera lens is invalid).
   - Converts the result from OpenCV convention (Y-down, Z-forward) to Blender's
     coordinate system (Y-up, Z-backward) and applies it to the active camera.
   - Displays the photo as a semi-transparent background image in the camera view.

---

## File Structure

```
face_blender/
├── __init__.py                   # Blender add-on registration, bl_info
├── operators.py                  # Blender operators (import, add/remove image, align camera)
├── panels.py                     # UI panels and UIList definitions
├── camera_alignment.py           # Core alignment logic (landmark detection, PnP, coord conversion)
├── landmark_mapping.py           # Landmark-to-vertex mapping utilities
├── utils.py                      # Coordinate conversion and matrix helpers
└── data/
    └── flame_landmark_mapping.json  # Default FLAME 68-landmark vertex indices
requirements.txt                  # Python dependencies
README.md                         # This file
```

---

## Installation

### 1. Install the add-on in Blender

1. Download or clone this repository.
2. In Blender, go to **Edit → Preferences → Add-ons → Install…**
3. Navigate to the repository folder and select the **`face_blender`** directory
   (or zip it first and select the zip).
4. Enable the add-on by ticking the checkbox next to **"FaceBlender"**.

### 2. Install Python dependencies

FaceBlender requires **OpenCV** and **MediaPipe** (or **dlib**) inside Blender's
bundled Python environment.

Open Blender's built-in Python console (**Scripting** workspace) and run:

```python
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install",
                "opencv-python", "mediapipe", "numpy"], check=True)
```

Alternatively, from a terminal locate Blender's Python executable (e.g.,
`<blender>/4.x/python/bin/python3.xx`) and run:

```bash
./python3.xx -m pip install opencv-python mediapipe numpy
```

> **Note:** On some systems pip may not be available in Blender's Python.
> Run `./python3.xx -m ensurepip` first.

#### Using dlib instead of MediaPipe (optional)

```bash
./python3.xx -m pip install dlib
```

You will also need the pre-trained shape predictor model:
[shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)

---

## Usage

1. Open the **3D Viewport** and press **N** to show the side panel.
2. Select the **FaceBlender** tab.

### Step 1 – Set the Head Mesh

- Use the **Head Mesh** search field to pick an existing mesh object, **or**
- Click the **Import** (↓) button to import an OBJ file directly.

### Step 2 – Add Reference Photos

- Click **+** (Add Image) to browse for a JPG/PNG/etc. photo.
- The photo appears in the image list.
- Repeat for each angle you want to align.

### Step 3 – Align the Camera

- Click the **camera icon** 📷 next to any reference photo.
- The add-on will detect landmarks, solve PnP, and update the **active camera**
  in the scene.
- The photo is set as a semi-transparent background image in the camera view.
- Switch to **Camera View** (Numpad 0) to verify the alignment.

### Settings

| Setting | Description |
|---------|-------------|
| **Backend** | `mediapipe` (default, recommended) or `dlib` |
| **dlib Predictor** | Path to `shape_predictor_68_face_landmarks.dat` (dlib only) |
| **Custom Mapping** | Path to a custom JSON landmark mapping file (optional) |

---

## Customising the Landmark Mapping

The built-in mapping (`face_blender/data/flame_landmark_mapping.json`) maps the
68 dlib-convention landmark indices to FLAME mesh vertex indices.

To use a different mesh topology (e.g., BFM), create a JSON file with the
following structure and point the **Custom Mapping** field to it:

```json
{
  "description": "My custom landmark mapping",
  "topology": "MyMesh",
  "mapping": {
    "0": 1234,
    "1": 5678,
    "...": "..."
  }
}
```

Keys are landmark indices (strings `"0"`–`"67"`), values are vertex indices in
your mesh.

---

## Coordinate System Conversion

The conversion between OpenCV and Blender follows the pattern from pixel3dmm's
`opencv_to_opengl()` function:

```python
# Build the 4×4 world-to-camera matrix in OpenCV convention
Rt = np.eye(4)
Rt[:3, :3] = R
Rt[:3, 3] = t

# Flip Y and Z: OpenCV (Y-down, Z-forward) → OpenGL/Blender (Y-up, Z-backward)
Rt[1] *= -1
Rt[2] *= -1

# Invert to get the camera-to-world matrix (Blender's matrix_world)
matrix_world = np.linalg.inv(Rt)
```

See `face_blender/utils.py` for the full implementation.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| *"No face detected in the image"* | Use a clear, front-facing photo with good lighting. Make sure the face occupies a large portion of the frame. |
| *"Missing dependency: OpenCV is required…"* | Follow the dependency installation steps above. |
| *"solvePnP failed to find a solution"* | Ensure the head mesh vertex indices in the landmark mapping actually correspond to the loaded mesh. Try a different photo. |
| *"Only N common landmarks…"* | The landmark mapping references vertex indices that don't exist in your mesh. Update the mapping or use a FLAME-compatible mesh. |
| Camera ends up in the wrong place | Check that the head mesh is not scaled non-uniformly. Apply scale with **Ctrl+A → Scale** before aligning. |

---

## Credits

- **pixel3dmm** – camera alignment approach and coordinate-system conversion
  (https://github.com/soubhikmandal/pixel3dmm)
- **MediaPipe** – face landmark detection (https://mediapipe.dev/)
- **OpenCV** – PnP solving (https://opencv.org/)
- **FLAME** – 3D face model topology (https://flame.is.tue.mpg.de/)
