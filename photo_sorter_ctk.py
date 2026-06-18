import os
import re
import shutil
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk
import numpy as np
from PIL import ExifTags, Image
from sklearn.cluster import DBSCAN


APP_NAME = "Photo Sorter"
APP_VERSION = "2.0"

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


class ActionCard(ctk.CTkFrame):
    def __init__(self, master, index, title, subtitle, accent, command):
        super().__init__(
            master,
            corner_radius=18,
            fg_color=("#ffffff", "#171923"),
            border_width=1,
            border_color=("#dbe3ef", "#2d3342"),
        )
        self.grid_columnconfigure(0, weight=1)

        badge = ctk.CTkLabel(
            self,
            text=str(index),
            width=36,
            height=36,
            corner_radius=18,
            fg_color=accent,
            text_color="#ffffff",
            font=("Segoe UI", 16, "bold"),
        )
        badge.grid(row=0, column=0, sticky="w", padx=18, pady=(18, 10))

        ctk.CTkLabel(
            self,
            text=title,
            anchor="w",
            font=("Segoe UI", 18, "bold"),
        ).grid(row=1, column=0, sticky="ew", padx=18)

        ctk.CTkLabel(
            self,
            text=subtitle,
            anchor="nw",
            justify="left",
            wraplength=250,
            text_color=("#526173", "#9aa6b8"),
            font=("Segoe UI", 12),
        ).grid(row=2, column=0, sticky="ew", padx=18, pady=(6, 16))

        ctk.CTkButton(
            self,
            text="Run",
            height=38,
            corner_radius=12,
            fg_color=accent,
            hover_color=accent,
            font=("Segoe UI", 13, "bold"),
            command=command,
        ).grid(row=3, column=0, sticky="ew", padx=18, pady=(0, 18))


class PhotoSorterApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title(f"{APP_NAME} {APP_VERSION}")
        self.geometry("1120x760")
        self.minsize(960, 660)

        self.folder_var = ctk.StringVar(value="")
        self.labels_var = ctk.StringVar(value="food,landscape,document")
        self.status_var = ctk.StringVar(value="Ready")

        self.clip_classifier = None
        self.person_detector = None
        self.face_app = None

        self._build_ui()
        configure_model_paths()

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        sidebar = ctk.CTkFrame(self, width=250, corner_radius=0, fg_color=("#edf2f7", "#0f1117"))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)

        ctk.CTkLabel(
            sidebar,
            text="Photo\nSorter",
            justify="left",
            anchor="w",
            font=("Segoe UI", 31, "bold"),
        ).pack(fill="x", padx=24, pady=(28, 8))

        ctk.CTkLabel(
            sidebar,
            text="Local photo organization\nwith AI-assisted sorting.",
            justify="left",
            anchor="w",
            text_color=("#5d6b7a", "#9aa6b8"),
            font=("Segoe UI", 13),
        ).pack(fill="x", padx=24)

        ctk.CTkFrame(sidebar, height=1, fg_color=("#d9e2ec", "#252b36")).pack(fill="x", pady=24)

        self._sidebar_item(sidebar, "Overview", active=True)
        self._sidebar_item(sidebar, "Date sorting")
        self._sidebar_item(sidebar, "Category sorting")
        self._sidebar_item(sidebar, "Face grouping")

        ctk.CTkFrame(sidebar, fg_color="transparent").pack(expand=True, fill="both")

        ctk.CTkLabel(
            sidebar,
            textvariable=self.status_var,
            anchor="w",
            text_color=("#526173", "#a9b4c3"),
            font=("Segoe UI", 12),
        ).pack(fill="x", padx=24, pady=(0, 24))

        main = ctk.CTkFrame(self, corner_radius=0, fg_color=("#f6f8fb", "#111318"))
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(3, weight=1)

        self._build_header(main)
        self._build_folder_panel(main)
        self._build_action_cards(main)
        self._build_log(main)

    def _sidebar_item(self, parent, text, active=False):
        color = ("#dfe8f4", "#1b2230") if active else "transparent"
        label_color = ("#101828", "#f2f5f9") if active else ("#667085", "#7f8a9a")
        ctk.CTkLabel(
            parent,
            text=text,
            anchor="w",
            fg_color=color,
            corner_radius=10,
            text_color=label_color,
            font=("Segoe UI", 13, "bold" if active else "normal"),
            height=38,
        ).pack(fill="x", padx=16, pady=3)

    def _build_header(self, parent):
        header = ctk.CTkFrame(parent, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=28, pady=(26, 12))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="Organize photos without the chaos",
            anchor="w",
            font=("Segoe UI", 28, "bold"),
        ).grid(row=0, column=0, sticky="ew")

        ctk.CTkLabel(
            header,
            text="Sort by date, classify by category, or group similar faces. Everything runs locally.",
            anchor="w",
            text_color=("#5d6b7a", "#9aa6b8"),
            font=("Segoe UI", 14),
        ).grid(row=1, column=0, sticky="ew", pady=(4, 0))

    def _build_folder_panel(self, parent):
        panel = ctk.CTkFrame(
            parent,
            corner_radius=18,
            fg_color=("#ffffff", "#171923"),
            border_width=1,
            border_color=("#dbe3ef", "#2d3342"),
        )
        panel.grid(row=1, column=0, sticky="ew", padx=28, pady=(0, 18))
        panel.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            panel,
            text="Target folder",
            anchor="w",
            font=("Segoe UI", 15, "bold"),
        ).grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 6))

        row = ctk.CTkFrame(panel, fg_color="transparent")
        row.grid(row=1, column=0, sticky="ew", padx=18, pady=(0, 16))
        row.grid_columnconfigure(0, weight=1)

        ctk.CTkEntry(
            row,
            textvariable=self.folder_var,
            placeholder_text="Paste a folder path here, or browse...",
            height=40,
            corner_radius=12,
            font=("Consolas", 12),
        ).grid(row=0, column=0, sticky="ew", padx=(0, 10))

        ctk.CTkButton(
            row,
            text="Browse",
            width=110,
            height=40,
            corner_radius=12,
            command=self.choose_folder,
        ).grid(row=0, column=1)

        ctk.CTkLabel(
            panel,
            text="Category labels",
            anchor="w",
            font=("Segoe UI", 15, "bold"),
        ).grid(row=2, column=0, sticky="ew", padx=18, pady=(0, 6))

        ctk.CTkEntry(
            panel,
            textvariable=self.labels_var,
            height=40,
            corner_radius=12,
            font=("Segoe UI", 13),
        ).grid(row=3, column=0, sticky="ew", padx=18)

        ctk.CTkLabel(
            panel,
            text="Comma-separated labels. People are detected automatically before CLIP classification.",
            anchor="w",
            text_color=("#5d6b7a", "#9aa6b8"),
            font=("Segoe UI", 12),
        ).grid(row=4, column=0, sticky="ew", padx=18, pady=(7, 16))

    def _build_action_cards(self, parent):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.grid(row=2, column=0, sticky="ew", padx=28)
        for column in range(3):
            row.grid_columnconfigure(column, weight=1)

        ActionCard(
            row,
            1,
            "Sort by date",
            "Moves root-level media into YYYY-MM-DD folders. Files without dates go to no_date.",
            "#6c5ce7",
            self.sort_by_date,
        ).grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        ActionCard(
            row,
            2,
            "Sort by category",
            "Moves images into people, your labels, or not_classified. Scans dated subfolders too.",
            "#00a884",
            self.sort_by_category,
        ).grid(row=0, column=1, sticky="nsew", padx=8)

        ActionCard(
            row,
            3,
            "Group faces",
            "Copies images into _face_groups/person_001, person_002, and no_face.",
            "#ff6b4a",
            self.group_faces,
        ).grid(row=0, column=2, sticky="nsew", padx=(8, 0))

    def _build_log(self, parent):
        panel = ctk.CTkFrame(
            parent,
            corner_radius=18,
            fg_color=("#ffffff", "#171923"),
            border_width=1,
            border_color=("#dbe3ef", "#2d3342"),
        )
        panel.grid(row=3, column=0, sticky="nsew", padx=28, pady=18)
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        top = ctk.CTkFrame(panel, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=18, pady=(16, 8))
        top.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            top,
            text="Activity log",
            anchor="w",
            font=("Segoe UI", 15, "bold"),
        ).grid(row=0, column=0, sticky="ew")

        ctk.CTkButton(
            top,
            text="Clear",
            width=80,
            height=30,
            fg_color="transparent",
            border_width=1,
            text_color=("#344054", "#d0d5dd"),
            command=self.clear_log,
        ).grid(row=0, column=1)

        self.log_box = ctk.CTkTextbox(
            panel,
            height=220,
            corner_radius=14,
            font=("Consolas", 12),
            fg_color=("#f8fafc", "#0b0d12"),
            text_color=("#344054", "#b8c0cc"),
            wrap="word",
        )
        self.log_box.grid(row=1, column=0, sticky="nsew", padx=18, pady=(0, 18))
        self.log_box.configure(state="disabled")

    def choose_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_var.set(folder)

    def selected_folder(self) -> Path:
        folder = Path(self.folder_var.get().strip().strip('"'))
        if not folder.is_dir():
            raise ValueError(f"Folder does not exist: {folder}")
        return folder

    def set_status(self, text: str):
        self.after(0, self.status_var.set, text)

    def log(self, text: str):
        self.after(0, self._append_log, text)

    def _append_log(self, text: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{timestamp}  {text}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def run_task(self, label, fn):
        def worker():
            try:
                self.set_status(f"Running: {label}")
                self.log("-" * 78)
                fn()
                self.log("Done.")
                self.set_status("Ready")
            except Exception:
                error = traceback.format_exc()
                self.log(error)
                self.set_status("Error")
                self.after(0, lambda: messagebox.showerror("Error", error))

        threading.Thread(target=worker, daemon=True).start()

    def sort_by_date(self):
        self.run_task("Sort by date", self._sort_by_date)

    def _sort_by_date(self):
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
        self.run_task("Sort by category", self._sort_by_category)

    def _sort_by_category(self):
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
        self.run_task("Group faces", self._group_faces)

    def _group_faces(self):
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
    app = PhotoSorterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
