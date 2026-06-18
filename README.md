# Photo Sorter

A small Windows-friendly desktop tool for organizing photo folders.

Photo Sorter can:

- Sort media by date into `YYYY-MM-DD` folders, with undated files going to `no_date`.
- Sort images by visual category using a stronger CLIP model.
- Always detect people first with an object detector before category sorting.
- Group similar faces into `person_001`, `person_002`, etc. using InsightFace.

The app is intentionally simple: choose a folder, optionally edit the category labels, then run one of the three actions.

## Features

### Sort by date

Moves root-level images and videos into date folders inside the selected folder.

Date detection order:

1. EXIF `DateTimeOriginal`
2. Other EXIF date fields
3. Date-like filename patterns
4. `no_date`

### Sort by category

Moves images into category folders inside the selected folder.

The app first detects people with:

```text
facebook/detr-resnet-50
```

If a person is detected, the file goes to:

```text
people
```

Remaining images are classified with:

```text
openai/clip-vit-large-patch14
```

Default labels:

```text
food,landscape,document
```

Uncertain images go to:

```text
not_classified
```

### Group similar faces

Copies images into:

```text
_face_groups/person_001
_face_groups/person_002
_face_groups/no_face
```

This action copies files instead of moving them.

It uses:

```text
InsightFace buffalo_l
```

Only the detection and recognition modules are loaded, which avoids unnecessary landmark and age/gender modules.

## Install

Python 3.11+ is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run From Source

```powershell
python photo_sorter_app.py
```

The first run may download model weights from Hugging Face and InsightFace.

## Build A Windows Release

Install PyInstaller:

```powershell
python -m pip install pyinstaller
```

Then build:

```powershell
pyinstaller --noconfirm PhotoSorterTool.spec
```

The release folder will be created at:

```text
dist/PhotoSorterTool
```

Keep the full release folder together. Do not distribute only the `.exe` if you bundle models and libraries.

## Bundling Models

This repository does not include model weights.

If you want a fully offline release, place bundled model caches under:

```text
models/huggingface
models/insightface
```

When frozen with PyInstaller, the app looks for these folders next to the bundled application resources.

## Safety Notes

- `Sort by date` moves files.
- `Sort by category` moves files.
- `Group similar faces` copies files.
- Always test on a copied folder before processing important originals.

## License

Choose a license before publishing publicly. MIT is a common option for small utilities.
