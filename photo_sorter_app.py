import os
import re
import shutil
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import END, Button, Entry, Frame, Label, StringVar, Text, Tk, filedialog, messagebox

import numpy as np
from PIL import ExifTags, Image
from sklearn.cluster import DBSCAN


APP_NAME = "Photo Sorter"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

NO_DATE = "no_date"
NOT_CLASSIFIED = "not_classified"
PERSON = "people"

CLIP_MODEL = "openai/clip-vit-large-patch14"
PERSON_DETECTOR_MODEL = "facebook/detr-resnet-50"
FACE_MODEL = "buffalo_l"

LABEL_PROMPTS = {
    "food": "a photo of food, a meal, snacks, or drinks",
    "landscape": "a landscape photo, scenery, nature, city view, or travel place",
    "scenery": "a landscape photo, scenery, nature, city view, or travel place",
    "document": "a photo of a document, paper, receipt, note, book page, or screenshot of text",
    "documents": "a photo of a document, paper, receipt, note, book page, or screenshot of text",
    "pet": "a photo of a pet, cat, dog, or animal",
    "pets": "a photo of a pet, cat, dog, or animal",
    "car": "a photo of a car, vehicle, bus, train, or transportation",
    "vehicle": "a photo of a car, vehicle, bus, train, or transportation",
}

EXIF_TAGS = {name: tag_id for tag_id, name in ExifTags.TAGS.items()}
DATE_TAGS = ("DateTimeOriginal", "DateTimeDigitized", "DateTime")


def resource_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parent


def configure_model_paths() -> None:
    """Use bundled model caches when this app is packaged with PyInstaller."""
    root = resource_root()
    bundled_models = root / "models"
    hf_home = bundled_models / "huggingface"
    insight_home = bundled_models / "insightface"

    if hf_home.exists():
        os.environ.setdefault("HF_HOME", str(hf_home))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(hf_home / "hub"))
    if insight_home.exists():
        os.environ.setdefault("INSIGHTFACE_HOME", str(insight_home))


def safe_folder_name(value: str) -> str:
    value = value.strip()
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", " ", value).strip(" ._")
    return value or NOT_CLASSIFIED


def iter_root_media(folder: Path):
    for path in sorted(folder.iterdir()):
        if path.is_file() and path.suffix.lower() in MEDIA_EXTENSIONS:
            yield path


def is_managed_folder(path: Path) -> bool:
    names = {part.lower() for part in path.parts}
    managed = {PERSON, NOT_CLASSIFIED, "_face_groups", "no_face"}
    managed.update({f"person_{idx:03d}" for idx in range(1, 1000)})
    return bool(names & managed)


def iter_category_images(folder: Path):
    for path in sorted(folder.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_parent = path.parent.relative_to(folder)
        if is_managed_folder(relative_parent):
            continue
        yield path


def iter_images_recursive(folder: Path):
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def parse_exif_datetime(value) -> datetime | None:
    if isinstance(value, bytes):
        value = value.decode("ascii", errors="ignore")
    if not value:
        return None

    text = str(value).strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text[:19], fmt)
        except ValueError:
            continue
        if dt.year not in (0, 1970):
            return dt
    return None


def exif_date(path: Path) -> datetime | None:
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        return None
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            for tag_name in DATE_TAGS:
                tag_id = EXIF_TAGS.get(tag_name)
                dt = parse_exif_datetime(exif.get(tag_id))
                if dt:
                    return dt
    except Exception:
        return None
    return None


def filename_date(path: Path) -> datetime | None:
    patterns = (
        r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)",
        r"([01]\d)[-_]([0-3]\d)[-_](20\d{2})",
    )
    for pattern in patterns:
        match = re.search(pattern, path.stem)
        if not match:
            continue
        parts = match.groups()
        if len(parts[0]) == 4:
            year, month, day = parts
        else:
            month, day, year = parts
        try:
            return datetime(int(year), int(month), int(day))
        except ValueError:
            continue
    return None


def date_folder(path: Path) -> str:
    dt = exif_date(path) or filename_date(path)
    return dt.strftime("%Y-%m-%d") if dt else NO_DATE


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def move_file(path: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = unique_destination(dest_dir / path.name)
    shutil.move(str(path), str(dest_path))
    return dest_path


def read_image_bgr(path: Path):
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            array = np.asarray(rgb)
            return array[:, :, ::-1].copy()
    except Exception:
        return None


class PhotoSorterApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title(APP_NAME)
        self.root.geometry("1040x740")
        self.root.minsize(920, 660)
        self.root.configure(bg="#eef2f7")

        self.folder_var = StringVar()
        self.labels_var = StringVar(value="food,landscape,document")

        self.clip_classifier = None
        self.person_detector = None
        self.face_app = None

        self.title_font = ("Segoe UI", 22, "bold")
        self.subtitle_font = ("Segoe UI", 10)
        self.section_font = ("Segoe UI", 11, "bold")
        self.normal_font = ("Segoe UI", 10)
        self.button_font = ("Segoe UI", 11, "bold")
        self.log_font = ("Consolas", 10)

        self._build_layout()
        configure_model_paths()

    def _build_layout(self):
        header = Frame(self.root, bg="#0f172a")
        header.grid(row=0, column=0, sticky="ew")

        Label(
            header,
            text="Photo Sorter",
            font=self.title_font,
            fg="#ffffff",
            bg="#0f172a",
            anchor="w",
        ).pack(fill="x", padx=22, pady=(18, 4))

        Label(
            header,
            text="Organize a folder by date, by visual category, or by similar faces. Date and category sorting move files. Face grouping copies files.",
            font=self.subtitle_font,
            fg="#cbd5e1",
            bg="#0f172a",
            anchor="w",
        ).pack(fill="x", padx=22, pady=(0, 18))

        body = Frame(self.root, bg="#eef2f7")
        body.grid(row=1, column=0, sticky="nsew", padx=22, pady=18)

        self._folder_section(body).grid(row=0, column=0, sticky="ew")
        self._label_section(body).grid(row=1, column=0, sticky="ew", pady=(16, 0))
        self._actions_section(body).grid(row=2, column=0, sticky="ew", pady=(18, 14))

        Label(body, text="Activity log", font=self.section_font, bg="#eef2f7", fg="#111827").grid(
            row=3, column=0, sticky="w"
        )
        self.log_box = Text(
            body,
            height=22,
            wrap="word",
            font=self.log_font,
            bg="#111827",
            fg="#e5e7eb",
            insertbackground="#ffffff",
            relief="flat",
            padx=12,
            pady=10,
        )
        self.log_box.grid(row=4, column=0, pady=(7, 0), sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(4, weight=1)

    def _folder_section(self, parent: Frame) -> Frame:
        frame = Frame(parent, bg="#ffffff", padx=16, pady=14)
        Label(frame, text="Target folder", font=self.section_font, bg="#ffffff", fg="#111827").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        Entry(frame, textvariable=self.folder_var, font=self.normal_font).grid(
            row=1, column=0, sticky="ew", ipady=6
        )
        Button(frame, text="Browse...", font=self.normal_font, command=self.choose_folder).grid(
            row=1, column=1, padx=(10, 0), ipady=4
        )
        Label(
            frame,
            text="Paste a folder path or choose one. The tool works inside this folder.",
            font=self.subtitle_font,
            bg="#ffffff",
            fg="#64748b",
        ).grid(row=2, column=0, sticky="w", pady=(7, 0))
        frame.columnconfigure(0, weight=1)
        return frame

    def _label_section(self, parent: Frame) -> Frame:
        frame = Frame(parent, bg="#ffffff", padx=16, pady=14)
        Label(frame, text="Category labels", font=self.section_font, bg="#ffffff", fg="#111827").grid(
            row=0, column=0, sticky="w", pady=(0, 8)
        )
        Entry(frame, textvariable=self.labels_var, font=self.normal_font).grid(
            row=1, column=0, sticky="ew", ipady=6
        )
        Label(
            frame,
            text="Comma-separated labels, for example: food,landscape,document. People are detected automatically.",
            font=self.subtitle_font,
            bg="#ffffff",
            fg="#64748b",
        ).grid(row=2, column=0, sticky="w", pady=(7, 0))
        frame.columnconfigure(0, weight=1)
        return frame

    def _actions_section(self, parent: Frame) -> Frame:
        frame = Frame(parent, bg="#eef2f7")
        self._action_button(
            frame,
            0,
            "Sort by date",
            "Moves root-level media to YYYY-MM-DD or no_date.",
            self.sort_by_date,
        )
        self._action_button(
            frame,
            1,
            "Sort by category",
            "Moves images to people, labels, or not_classified.",
            self.sort_by_category,
        )
        self._action_button(
            frame,
            2,
            "Group similar faces",
            "Copies images to _face_groups/person_001.",
            self.group_faces,
        )
        for col in range(3):
            frame.columnconfigure(col, weight=1)
        return frame

    def _action_button(self, parent: Frame, column: int, title: str, detail: str, command):
        card = Frame(parent, bg="#ffffff", padx=14, pady=12)
        card.grid(row=0, column=column, sticky="nsew", padx=6)
        Label(card, text=title, font=self.button_font, bg="#ffffff", fg="#111827").pack(anchor="w")
        Label(
            card,
            text=detail,
            font=self.subtitle_font,
            bg="#ffffff",
            fg="#64748b",
            wraplength=260,
            justify="left",
        ).pack(anchor="w", pady=(4, 10))
        Button(card, text="Run", font=self.button_font, command=lambda: self.run_task(command)).pack(
            fill="x", ipady=6
        )

    def choose_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_var.set(folder)

    def selected_folder(self) -> Path:
        folder = Path(self.folder_var.get().strip().strip('"'))
        if not folder.is_dir():
            raise ValueError(f"Folder does not exist: {folder}")
        return folder

    def log(self, text: str):
        self.root.after(0, self._append_log, text)

    def _append_log(self, text: str):
        self.log_box.insert(END, text + "\n")
        self.log_box.see(END)

    def run_task(self, fn):
        def worker():
            try:
                self.log("=" * 78)
                fn()
                self.log("Done.")
            except Exception:
                error = traceback.format_exc()
                self.log(error)
                self.root.after(0, lambda: messagebox.showerror("Error", error))

        threading.Thread(target=worker, daemon=True).start()

    def sort_by_date(self):
        folder = self.selected_folder()
        files = list(iter_root_media(folder))
        self.log(f"Sort by date: {folder}")
        self.log(f"Found {len(files)} root-level media file(s).")

        moved = 0
        for path in files:
            dest = move_file(path, folder / date_folder(path))
            moved += 1
            self.log(f"{path.name} -> {dest.parent.name}")
        self.log(f"Date sort complete. Moved {moved} file(s).")

    def load_category_models(self):
        if self.clip_classifier is not None and self.person_detector is not None:
            return
        from transformers import pipeline

        if self.person_detector is None:
            self.log(f"Loading person detector: {PERSON_DETECTOR_MODEL}")
            self.person_detector = pipeline("object-detection", model=PERSON_DETECTOR_MODEL)
        if self.clip_classifier is None:
            self.log(f"Loading category model: {CLIP_MODEL}")
            self.clip_classifier = pipeline("zero-shot-image-classification", model=CLIP_MODEL)

    def detect_person(self, path: Path, threshold: float = 0.70) -> tuple[bool, float]:
        result = self.person_detector(str(path))
        best = 0.0
        for item in result:
            if str(item.get("label", "")).lower() == "person":
                best = max(best, float(item.get("score", 0.0)))
        return best >= threshold, best

    def classify_image(self, path: Path, labels: list[tuple[str, str]]) -> tuple[str, str]:
        has_person, score = self.detect_person(path)
        if has_person:
            return PERSON, f"person={score:.3f}"

        prompts = [prompt for _folder, prompt in labels]
        result = self.clip_classifier(str(path), candidate_labels=prompts)
        prompt_to_folder = {prompt: folder for folder, prompt in labels}
        top = result[0]
        second = float(result[1]["score"]) if len(result) > 1 else 0.0
        score = float(top["score"])
        gap = score - second
        folder_label = prompt_to_folder[str(top["label"])]

        if score < 0.55 or gap < 0.12:
            return NOT_CLASSIFIED, f"{folder_label}={score:.3f}, gap={gap:.3f}"
        return folder_label, f"{folder_label}={score:.3f}, gap={gap:.3f}"

    def sort_by_category(self):
        folder = self.selected_folder()
        raw_labels = [safe_folder_name(x).lower() for x in self.labels_var.get().split(",") if x.strip()]
        raw_labels = [x for x in raw_labels if x != PERSON]
        labels = [(label, LABEL_PROMPTS.get(label, label)) for label in raw_labels]
        if not labels:
            raise ValueError("Enter at least one category label, for example: food,landscape,document")

        self.load_category_models()
        files = list(iter_category_images(folder))
        self.log(f"Sort by category: {folder}")
        self.log(f"Categories: {PERSON}, {', '.join(raw_labels)}, {NOT_CLASSIFIED}")
        self.log(f"Found {len(files)} image file(s), including dated subfolders.")

        moved = 0
        for path in files:
            category, reason = self.classify_image(path, labels)
            dest = move_file(path, folder / category)
            moved += 1
            self.log(f"{path.name} -> {dest.parent.name} ({reason})")
        self.log(f"Category sort complete. Moved {moved} file(s).")

    def load_face_model(self):
        if self.face_app is not None:
            return
        from insightface.app import FaceAnalysis

        self.log(f"Loading face model: {FACE_MODEL}")
        app = FaceAnalysis(
            name=FACE_MODEL,
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        self.face_app = app

    @staticmethod
    def largest_face(faces):
        def area(face) -> float:
            x1, y1, x2, y2 = face.bbox
            return float(max(0, x2 - x1) * max(0, y2 - y1))

        return max(faces, key=area)

    def group_faces(self):
        folder = self.selected_folder()
        output = folder / "_face_groups"
        self.load_face_model()

        files = [path for path in iter_images_recursive(folder) if output not in path.parents]
        self.log(f"Group similar faces: {folder}")
        self.log(f"Output folder: {output}")
        self.log(f"Found {len(files)} image file(s).")

        records = []
        no_face = []
        for index, path in enumerate(files, start=1):
            image = read_image_bgr(path)
            if image is None:
                no_face.append(path)
                self.log(f"[{index}/{len(files)}] Read failed: {path.name}")
                continue

            try:
                faces = self.face_app.get(image)
            except Exception as exc:
                no_face.append(path)
                self.log(f"[{index}/{len(files)}] Face analysis failed: {path.name} ({exc})")
                continue
            if not faces:
                no_face.append(path)
                self.log(f"[{index}/{len(files)}] No face: {path.name}")
                continue

            face = self.largest_face(faces)
            embedding = np.asarray(face.embedding, dtype=np.float32)
            norm = np.linalg.norm(embedding)
            if norm == 0:
                no_face.append(path)
                continue
            records.append((path, embedding / norm, len(faces)))
            self.log(f"[{index}/{len(files)}] faces={len(faces)}: {path.name}")

        planned = []
        if records:
            embeddings = np.vstack([embedding for _path, embedding, _count in records])
            clustering = DBSCAN(eps=0.42, min_samples=1, metric="cosine").fit(embeddings)
            label_to_name = {}
            next_person = 1
            for (path, _embedding, _count), label in zip(records, clustering.labels_.tolist()):
                if label not in label_to_name:
                    label_to_name[label] = f"person_{next_person:03d}"
                    next_person += 1
                planned.append((path, label_to_name[label]))

        planned.extend((path, "no_face") for path in no_face)

        copied = 0
        for path, group in planned:
            dest_dir = output / group
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = unique_destination(dest_dir / path.name)
            shutil.copy2(str(path), str(dest_path))
            copied += 1
            self.log(f"{path.name} -> _face_groups/{group}")

        self.log(f"Face grouping complete. Copied {copied} file(s).")


def main():
    root = Tk()
    PhotoSorterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
