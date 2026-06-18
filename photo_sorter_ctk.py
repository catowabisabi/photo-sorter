import os
import re
import shutil
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import ExifTags, Image
from sklearn.cluster import DBSCAN


# ── App identity ──────────────────────────────────────────────────────────────
APP_NAME = "Photo Sorter"
APP_VERSION = "2.0"

# ── File types ────────────────────────────────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
MEDIA_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

# ── Folder names ──────────────────────────────────────────────────────────────
NO_DATE = "no_date"
NOT_CLASSIFIED = "not_classified"
PERSON = "people"

# ── Model IDs ─────────────────────────────────────────────────────────────────
CLIP_MODEL = "openai/clip-vit-large-patch14"
PERSON_DETECTOR_MODEL = "facebook/detr-resnet-50"
FACE_MODEL = "buffalo_l"

# ── CLIP label prompts ────────────────────────────────────────────────────────
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

# ── EXIF helpers ──────────────────────────────────────────────────────────────
EXIF_TAGS = {name: tag_id for tag_id, name in ExifTags.TAGS.items()}
DATE_TAGS = ("DateTimeOriginal", "DateTimeDigitized", "DateTime")

# ── Theme colours (dark + light) ──────────────────────────────────────────────
PURPLE = ("#534AB7", "#7F77DD")
TEAL   = ("#0F6E56", "#1D9E75")
CORAL  = ("#993C1D", "#D85A30")
SIDEBAR_BG_DARK  = "#1a1b23"
SIDEBAR_BG_LIGHT = "#f0f0f0"
CARD_BG_DARK  = "#23242f"
CARD_BG_LIGHT = "#ffffff"
LOG_BG_DARK  = "#16171f"
LOG_BG_LIGHT = "#f8f8f8"


# ═════════════════════════════════════════════════════════════════════════════
#  Utility functions  (unchanged logic from original)
# ═════════════════════════════════════════════════════════════════════════════

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


# ═════════════════════════════════════════════════════════════════════════════
#  Custom widgets
# ═════════════════════════════════════════════════════════════════════════════

class TagsEntry(ctk.CTkFrame):
    """Horizontal list of removable tags + inline text input."""

    def __init__(self, master, initial_tags: list[str], **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._tags: list[str] = list(initial_tags)
        self._tag_widgets: list[ctk.CTkFrame] = []
        self._build()

    def _build(self):
        for w in self.winfo_children():
            w.destroy()
        self._tag_widgets.clear()

        wrap = ctk.CTkFrame(self, fg_color="transparent")
        wrap.pack(fill="x")

        for tag in self._tags:
            chip = ctk.CTkFrame(wrap, corner_radius=20,
                                fg_color=("gray85", "gray25"),
                                border_width=1,
                                border_color=("gray70", "gray40"))
            chip.pack(side="left", padx=(0, 6), pady=2)
            ctk.CTkLabel(chip, text=tag, font=("Segoe UI", 12),
                         padx=8, pady=3).pack(side="left")
            ctk.CTkButton(chip, text="✕", width=18, height=18,
                          font=("Segoe UI", 10),
                          fg_color="transparent",
                          hover_color=("gray75", "gray35"),
                          command=lambda t=tag: self._remove(t)).pack(side="left", padx=(0, 4))

        self._entry = ctk.CTkEntry(wrap, placeholder_text="+ add label",
                                   width=110, height=28,
                                   border_width=0,
                                   fg_color="transparent",
                                   font=("Segoe UI", 12))
        self._entry.pack(side="left", padx=4)
        self._entry.bind("<Return>", self._on_enter)

    def _on_enter(self, _event=None):
        text = self._entry.get().strip().lower()
        if text and text not in self._tags:
            self._tags.append(text)
            self._build()

    def _remove(self, tag: str):
        if tag in self._tags:
            self._tags.remove(tag)
            self._build()

    def get_tags(self) -> list[str]:
        return list(self._tags)


class ActionCard(ctk.CTkFrame):
    """A card with icon badge, title, description, and a Run button."""

    ACCENT_COLORS = {
        "purple": ("#EEEDFE", "#3C3489"),
        "teal":   ("#E1F5EE", "#085041"),
        "coral":  ("#FAECE7", "#712B13"),
    }

    def __init__(self, master, title: str, description: str,
                 icon: str, accent: str, command, **kwargs):
        super().__init__(master, corner_radius=12,
                         border_width=1,
                         border_color=("gray80", "gray30"),
                         fg_color=(CARD_BG_LIGHT, CARD_BG_DARK),
                         **kwargs)

        light_bg, dark_bg = self.ACCENT_COLORS[accent]

        # Icon badge
        badge = ctk.CTkFrame(self, width=40, height=40, corner_radius=10,
                             fg_color=(light_bg, dark_bg))
        badge.pack(anchor="w", padx=16, pady=(16, 0))
        badge.pack_propagate(False)
        ctk.CTkLabel(badge, text=icon, font=("Segoe UI Emoji", 18)).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(self, text=title, font=("Segoe UI", 13, "bold"),
                     anchor="w").pack(fill="x", padx=16, pady=(10, 0))

        ctk.CTkLabel(self, text=description, font=("Segoe UI", 11),
                     text_color=("gray50", "gray60"),
                     wraplength=200, justify="left",
                     anchor="w").pack(fill="x", padx=16, pady=(4, 12))

        ctk.CTkButton(self, text="▶  Run", height=32,
                      font=("Segoe UI", 12, "bold"),
                      fg_color="transparent",
                      border_width=1,
                      border_color=("gray70", "gray40"),
                      text_color=("gray20", "gray80"),
                      hover_color=("gray90", "gray25"),
                      command=command).pack(fill="x", padx=16, pady=(0, 16))


class SidebarItem(ctk.CTkFrame):
    """Clickable sidebar nav row."""

    def __init__(self, master, icon: str, label: str, active=False, **kwargs):
        bg = ("gray85", "gray20") if active else ("transparent", "transparent")
        super().__init__(master, corner_radius=8, fg_color=bg, **kwargs)
        self.configure(cursor="hand2")

        ctk.CTkLabel(self, text=f"{icon}  {label}",
                     font=("Segoe UI", 13, "bold" if active else "normal"),
                     text_color=("gray15", "gray90") if active else ("gray40", "gray55"),
                     anchor="w").pack(fill="x", padx=12, pady=8)


# ═════════════════════════════════════════════════════════════════════════════
#  Main application
# ═════════════════════════════════════════════════════════════════════════════

class PhotoSorterApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title(APP_NAME)
        self.geometry("1080x680")
        self.minsize(900, 600)

        self.clip_classifier = None
        self.person_detector = None
        self.face_app = None

        self._build_ui()
        configure_model_paths()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main()

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=210, corner_radius=0,
                               fg_color=(SIDEBAR_BG_LIGHT, SIDEBAR_BG_DARK))
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(8, weight=1)

        # App brand
        brand = ctk.CTkFrame(sidebar, fg_color="transparent")
        brand.grid(row=0, column=0, padx=18, pady=(22, 16), sticky="ew")

        icon_badge = ctk.CTkFrame(brand, width=38, height=38, corner_radius=10,
                                  fg_color=("#1a1a2e", "#1a1a2e"))
        icon_badge.pack(anchor="w")
        icon_badge.pack_propagate(False)
        ctk.CTkLabel(icon_badge, text="🖼", font=("Segoe UI Emoji", 18)).place(relx=0.5, rely=0.5, anchor="center")

        ctk.CTkLabel(brand, text=APP_NAME, font=("Segoe UI", 15, "bold"),
                     anchor="w").pack(fill="x", pady=(8, 0))
        ctk.CTkLabel(brand, text="Organize your media",
                     font=("Segoe UI", 10),
                     text_color=("gray50", "gray55"),
                     anchor="w").pack(fill="x")

        # Separator
        ctk.CTkFrame(sidebar, height=1, fg_color=("gray80", "gray25")).grid(
            row=1, column=0, sticky="ew", padx=0, pady=0)

        # Nav items
        nav_items = [
            ("🗂", "Overview",      True),
            ("📅", "By date",       False),
            ("🏷", "By category",   False),
            ("👤", "By face",       False),
            ("📋", "History",       False),
            ("⚙️", "Settings",      False),
        ]
        for idx, (icon, label, active) in enumerate(nav_items):
            item = SidebarItem(sidebar, icon, label, active=active)
            item.grid(row=idx + 2, column=0, sticky="ew", padx=10, pady=2)

        # Status dot at bottom
        status = ctk.CTkFrame(sidebar, fg_color="transparent")
        status.grid(row=9, column=0, padx=18, pady=16, sticky="sw")
        dot = ctk.CTkFrame(status, width=8, height=8, corner_radius=4,
                           fg_color="#1D9E75")
        dot.pack(side="left", padx=(0, 6))
        dot.pack_propagate(False)
        ctk.CTkLabel(status, text="Models ready",
                     font=("Segoe UI", 11),
                     text_color=("gray50", "gray55")).pack(side="left")

    def _build_main(self):
        main = ctk.CTkFrame(self, corner_radius=0, fg_color=("gray95", "gray13"))
        main.grid(row=0, column=1, sticky="nsew")
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)

        self._build_topbar(main)
        self._build_content(main)

    def _build_topbar(self, parent):
        bar = ctk.CTkFrame(parent, corner_radius=0, height=60,
                           fg_color=(CARD_BG_LIGHT, CARD_BG_DARK),
                           border_width=0)
        bar.grid(row=0, column=0, sticky="ew")
        bar.grid_columnconfigure(0, weight=1)
        bar.grid_propagate(False)

        inner = ctk.CTkFrame(bar, fg_color="transparent")
        inner.pack(fill="x", padx=22, pady=12)
        inner.columnconfigure(0, weight=1)

        # Folder field
        field = ctk.CTkFrame(inner, corner_radius=8,
                             fg_color=("gray88", "gray20"),
                             border_width=1,
                             border_color=("gray75", "gray30"))
        field.grid(row=0, column=0, sticky="ew", ipady=2)

        ctk.CTkLabel(field, text="📁", font=("Segoe UI Emoji", 13),
                     padx=10).pack(side="left")

        self.folder_var = ctk.StringVar(value="No folder selected")
        ctk.CTkLabel(field, textvariable=self.folder_var,
                     font=("Consolas", 12),
                     text_color=("gray40", "gray60"),
                     anchor="w").pack(side="left", fill="x", expand=True)

        ctk.CTkButton(inner, text="📂  Browse", width=110, height=36,
                      font=("Segoe UI", 12),
                      fg_color="transparent",
                      border_width=1,
                      border_color=("gray70", "gray40"),
                      text_color=("gray15", "gray85"),
                      hover_color=("gray85", "gray25"),
                      command=self.choose_folder).grid(row=0, column=1, padx=(10, 0))

        # Separator
        ctk.CTkFrame(parent, height=1, fg_color=("gray80", "gray25")).grid(
            row=1, column=0, sticky="ew")

    def _build_content(self, parent):
        scroll = ctk.CTkScrollableFrame(parent, fg_color="transparent")
        scroll.grid(row=2, column=0, sticky="nsew", padx=22, pady=18)
        scroll.grid_columnconfigure(0, weight=1)

        # ── Actions ──
        self._section_label(scroll, "Actions").grid(row=0, column=0, sticky="w", pady=(0, 8))

        cards_row = ctk.CTkFrame(scroll, fg_color="transparent")
        cards_row.grid(row=1, column=0, sticky="ew")
        for i in range(3):
            cards_row.columnconfigure(i, weight=1)

        ActionCard(cards_row,
                   title="Sort by date",
                   description="Moves root-level media into YYYY-MM-DD folders. Files without dates go to no_date.",
                   icon="📅", accent="purple",
                   command=self.sort_by_date).grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        ActionCard(cards_row,
                   title="Sort by category",
                   description="Classifies images using CLIP into your labels, people, or not_classified.",
                   icon="🏷", accent="teal",
                   command=self.sort_by_category).grid(row=0, column=1, sticky="nsew", padx=4)

        ActionCard(cards_row,
                   title="Group similar faces",
                   description="Copies images by face identity into _face_groups/person_001 folders.",
                   icon="👤", accent="coral",
                   command=self.group_faces).grid(row=0, column=2, sticky="nsew", padx=(8, 0))

        # ── Category labels ──
        self._section_label(scroll, "Category labels").grid(
            row=2, column=0, sticky="w", pady=(20, 8))

        labels_card = ctk.CTkFrame(scroll, corner_radius=12,
                                   border_width=1,
                                   border_color=("gray80", "gray30"),
                                   fg_color=(CARD_BG_LIGHT, CARD_BG_DARK))
        labels_card.grid(row=3, column=0, sticky="ew")
        labels_card.columnconfigure(0, weight=1)

        ctk.CTkLabel(labels_card,
                     text="Used when sorting by category. People are detected automatically.",
                     font=("Segoe UI", 11),
                     text_color=("gray50", "gray55"),
                     anchor="w").grid(row=0, column=0, padx=16, pady=(14, 6), sticky="w")

        self.tags_entry = TagsEntry(labels_card,
                                    initial_tags=["food", "landscape", "document"])
        self.tags_entry.grid(row=1, column=0, padx=16, pady=(0, 14), sticky="ew")

        # ── Activity log ──
        self._section_label(scroll, "Activity log").grid(
            row=4, column=0, sticky="w", pady=(20, 8))

        log_card = ctk.CTkFrame(scroll, corner_radius=12,
                                border_width=1,
                                border_color=("gray80", "gray30"),
                                fg_color=(CARD_BG_LIGHT, CARD_BG_DARK))
        log_card.grid(row=5, column=0, sticky="ew")
        log_card.columnconfigure(0, weight=1)

        # Log header
        log_hdr = ctk.CTkFrame(log_card, fg_color="transparent")
        log_hdr.grid(row=0, column=0, sticky="ew", padx=14, pady=(10, 6))
        log_hdr.columnconfigure(0, weight=1)

        ctk.CTkLabel(log_hdr, text="⬛  Output",
                     font=("Segoe UI", 12, "bold"),
                     anchor="w").grid(row=0, column=0, sticky="w")

        ctk.CTkButton(log_hdr, text="Clear", width=52, height=24,
                      font=("Segoe UI", 11),
                      fg_color="transparent",
                      border_width=1,
                      border_color=("gray70", "gray40"),
                      text_color=("gray40", "gray55"),
                      hover_color=("gray88", "gray25"),
                      command=self.clear_log).grid(row=0, column=1)

        ctk.CTkFrame(log_card, height=1, fg_color=("gray82", "gray28")).grid(
            row=1, column=0, sticky="ew")

        self.log_box = ctk.CTkTextbox(log_card, height=240,
                                      font=("Consolas", 12),
                                      fg_color=(LOG_BG_LIGHT, LOG_BG_DARK),
                                      text_color=("gray30", "gray70"),
                                      corner_radius=0,
                                      border_width=0,
                                      wrap="word")
        self.log_box.grid(row=2, column=0, sticky="ew", padx=0, pady=0)
        self.log_box.configure(state="disabled")

    @staticmethod
    def _section_label(parent, text: str) -> ctk.CTkLabel:
        return ctk.CTkLabel(parent, text=text.upper(),
                            font=("Segoe UI", 10, "bold"),
                            text_color=("gray50", "gray50"),
                            anchor="w")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def choose_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.folder_var.set(folder)

    def selected_folder(self) -> Path:
        value = self.folder_var.get().strip().strip('"')
        folder = Path(value)
        if not folder.is_dir():
            raise ValueError(f"Folder does not exist: {folder}")
        return folder

    def log(self, text: str):
        self.after(0, self._append_log, text)

    def _append_log(self, text: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self.log_box.configure(state="normal")
        self.log_box.insert("end", f"{ts}  {text}\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def clear_log(self):
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")

    def run_task(self, fn):
        def worker():
            try:
                self.log("─" * 60)
                fn()
                self.log("✓ Done.")
            except Exception:
                error = traceback.format_exc()
                self.log(error)
                self.after(0, lambda: messagebox.showerror("Error", error))

        threading.Thread(target=worker, daemon=True).start()

    # ── Core actions (unchanged logic) ────────────────────────────────────────

    def sort_by_date(self):
        self.run_task(self._sort_by_date)

    def _sort_by_date(self):
        folder = self.selected_folder()
        files = list(iter_root_media(folder))
        self.log(f"Sort by date: {folder}")
        self.log(f"Found {len(files)} root-level media file(s).")
        moved = 0
        for path in files:
            dest = move_file(path, folder / date_folder(path))
            moved += 1
            self.log(f"{path.name}  →  {dest.parent.name}")
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
        self.run_task(self._sort_by_category)

    def _sort_by_category(self):
        folder = self.selected_folder()
        raw_labels = [safe_folder_name(x).lower() for x in self.tags_entry.get_tags() if x.strip()]
        raw_labels = [x for x in raw_labels if x != PERSON]
        labels = [(label, LABEL_PROMPTS.get(label, label)) for label in raw_labels]
        if not labels:
            raise ValueError("Add at least one category label, e.g. food, landscape, document")
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
            self.log(f"{path.name}  →  {dest.parent.name}  ({reason})")
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
        self.run_task(self._group_faces)

    def _group_faces(self):
        folder = self.selected_folder()
        output = folder / "_face_groups"
        self.load_face_model()
        files = [p for p in iter_images_recursive(folder) if output not in p.parents]
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
            embeddings = np.vstack([e for _p, e, _c in records])
            clustering = DBSCAN(eps=0.42, min_samples=1, metric="cosine").fit(embeddings)
            label_to_name = {}
            next_person = 1
            for (path, _e, _c), label in zip(records, clustering.labels_.tolist()):
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
            self.log(f"{path.name}  →  _face_groups/{group}")

        self.log(f"Face grouping complete. Copied {copied} file(s).")


# ═════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    app = PhotoSorterApp()
    app.mainloop()


if __name__ == "__main__":
    main()
