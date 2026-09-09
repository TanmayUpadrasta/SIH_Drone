## Inputs

The viewer accepts:

- **3D Mesh** — `.obj` / `.ply` file from Role 3
- **Georeference Transform** — `.json` file from Role 4
- **Capture Metadata** — `.json` file from Role 6

## Features

- Interactive 3D model visualization
- Shaded surface view
- Wireframe overlay
- Real-world distance measurement
- Loading of georeferencing transformation
- Loading of capture metadata
- Sample scene for demonstration/testing

## Pipeline

```text
Role 3
3D Reconstruction
      ↓
 .OBJ / .PLY
      ↓
   Role 5
  3D Viewer
      ↑
      │
Role 4 ── Georeference Transform (.json)
Role 6 ── Capture Metadata (.json)
