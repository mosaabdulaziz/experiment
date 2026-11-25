#!/usr/bin/env python3
"""Interactive YOLO annotation viewer/editor for plate crops.

Features:
  - Navigate frames with ←/→ arrows
  - Toggle bounding box overlay with Space
  - Delete current image + label with Delete/Backspace
  - Remove nearest bounding box with 'r' (click near one first)
  - Add bounding box by click-dragging with left mouse
  - Multi-class support in YOLO format: labels are 'class_id xc yc w h',
    class names are read/written to classes.txt (one name per line, index = id)
    in the labels dir. It is auto-created with a default 'plate' class if missing.
    An explicit classes file can be given with --classes (a .txt or a YOLO/dataset
    .yaml with a 'names' section, e.g. mtmc/datasets/vehicle_detection/data.yaml);
    otherwise a nearby data.yaml is auto-discovered next to the labels dir.
  - Change a box's class by clicking its number badge (top-left of the box)
    to cycle classes, or press 0-9 while pointing at a box: the box whose
    number badge is nearest the mouse gets that class (box under the cursor
    and the active box also count). With no box targeted, 0-9 sets the class
    used for newly drawn boxes
  - Add classes with 'n' (or the ＋ Class button), remove with 'x' (－ Class)
  - Reassign all boxes of one class to another across the folder with 'a'
    (or the ⇄ Reassign button), e.g. turn every class 2 box into class 0
  - Propagate the current frame's classes forward with Shift+C (or the
    🏷 Propagate → button): boxes in following frames that overlap the
    current boxes (IoU >= threshold) get their class overridden
  - Copy a bounding box with 'c': the box whose label (top-left number) is
    nearest the mouse is memorized (the same box is never memorized twice),
    and every frame you navigate to gets the copied classes stamped onto
    overlapping boxes. The memorized boxes are drawn as offset dashed ghosts.
    Press 'm' pointing near a memorized box's label to drop that memorized box
    (nearest label wins); with 'p' auto-paste toggles on/off
  - Save all changes with 's' or on exit
  - Jump to random frame with 'r' key when not hovering near a box
  - Show crop thumbnails for multi-plate frames

Usage:
    .venv/bin/python -m mtmc.annotate_viewer --frames <frames_dir> --labels <labels_dir>
    .venv/bin/python -m mtmc.annotate_viewer --frames <frames_dir> --labels <labels_dir> --crops <crops_dir>
    .venv/bin/python -m mtmc.annotate_viewer --frames <frames_dir> --labels <labels_dir> --classes mtmc/datasets/vehicle_detection/data.yaml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import tkinter as tk
from tkinter import messagebox, simpledialog

import json

from PIL import Image, ImageDraw, ImageFont, ImageTk


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Colors
COLOR_BOX = "#00ff00"
COLOR_BOX_SELECTED = "#ff4444"
COLOR_BOX_NEW = "#ffff00"
COLOR_MEM = "#ffffff"
COLOR_TEXT_BG = "#00aa00"
COLOR_TEXT_FG = "white"
COLOR_BG = "#202020"
COLOR_STATUS = "#aaaaaa"
COLOR_HUD = "#ffffff"

# Minimum IoU for a box in a following frame to inherit the current frame's
# class during class propagation
PROPAGATE_IOU_THRESHOLD = 0.3

# Per-class outline palette (cycled by class id)
CLASS_PALETTE = [
    "#00ff00", "#ff4444", "#00aaff", "#ffaa00", "#ff44ff",
    "#aaff00", "#00ffff", "#ff6699", "#cccc00", "#9966ff",
    "#66ffcc", "#ff9944",
]


def load_reviewed(labels_dir: Path) -> set[str]:
    """Load reviewed image names from .reviewed.json in labels_dir or parent dir."""
    for target_dir in (labels_dir, labels_dir.parent):
        rev_file = target_dir / ".reviewed.json"
        if rev_file.exists():
            try:
                data = json.loads(rev_file.read_text())
                return set(data.get("reviewed_images", []))
            except Exception:
                pass
    return set()


def save_reviewed(labels_dir: Path, reviewed_set: set[str]) -> None:
    """Save reviewed image names to .reviewed.json in labels_dir."""
    rev_file = labels_dir / ".reviewed.json"
    try:
        rev_file.write_text(json.dumps({"reviewed_images": sorted(list(reviewed_set))}, indent=2))
    except Exception as e:
        print(f"[WARN] Failed to write reviewed state to {rev_file}: {e}")


DEFAULT_CLASSES = ["plate"]


def _names_from_yaml_file(yaml_path: Path) -> list[str]:
    """Extract class names from a YOLO/dataset .yaml (e.g. data.yaml).

    Supports both the Ultralytics forms:
      names:
        0: truck
        1: trailer
      names: [truck, trailer]
    and a bare top-level list of names.
    """
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[WARN] Failed to parse classes YAML {yaml_path}: {e}")
        return []
    if not isinstance(data, dict):
        if isinstance(data, list):
            return [str(n).strip() for n in data if str(n).strip()]
        return []
    raw = data.get("names") or data.get("class_names")
    if isinstance(raw, list):
        return [str(n).strip() for n in raw if str(n).strip()]
    if isinstance(raw, dict):
        try:
            items = sorted(((int(k), v) for k, v in raw.items()),
                           key=lambda kv: kv[0])
        except (ValueError, TypeError):
            return [str(v).strip() for v in raw.values() if str(v).strip()]
        return [str(v).strip() for _, v in items if str(v).strip()]
    return []


def _names_from_text_file(path: Path) -> list[str]:
    """Class names from a plain text file (one name per line, index = id)."""
    try:
        names = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    except Exception as e:
        print(f"[WARN] Failed to read classes file {path}: {e}")
        return []
    return names


def load_classes(labels_dir: Path, classes_file: str | None = None) -> list[str]:
    """Load class names for the viewer.

    Priority: an explicitly provided classes file (--classes, .txt or .yaml) →
    labels_dir/classes.txt → a nearby dataset data.yaml → a freshly created
    default classes.txt (created if missing).
    """
    names: list[str] = []
    if classes_file:
        cpath = Path(classes_file)
        if cpath.suffix.lower() in {".yaml", ".yml"}:
            names = _names_from_yaml_file(cpath)
        else:
            names = _names_from_text_file(cpath)

    cls_file = labels_dir / "classes.txt"
    if not names and cls_file.exists():
        names = _names_from_text_file(cls_file)

    if not names:
        dataset_yaml = _find_dataset_yaml(labels_dir)
        if dataset_yaml is not None:
            names = _names_from_yaml_file(dataset_yaml)

    if not names:
        save_classes(labels_dir, list(DEFAULT_CLASSES))
        return list(DEFAULT_CLASSES)

    # Persist a classes.txt next to the labels so the names survive on disk
    # unless the user pointed at an explicit external classes file.
    if not classes_file and not cls_file.exists():
        save_classes(labels_dir, names)
    return names


def _find_dataset_yaml(labels_dir: Path) -> Path | None:
    """Search for a data.yaml next to the labels dir (labels/labels-parent)."""
    for d in (labels_dir, labels_dir.parent):
        if not d.exists():
            continue
        for cand in (d / "data.yaml", d / "dataset.yaml"):
            if cand.exists():
                return cand
    return None


def save_classes(labels_dir: Path, classes: list[str]) -> None:
    """Write class names to classes.txt in the labels dir."""
    try:
        labels_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    cls_file = labels_dir / "classes.txt"
    try:
        cls_file.write_text("\n".join(classes) + "\n")
    except Exception as e:
        print(f"[WARN] Failed to write classes to {cls_file}: {e}")


def discover_folders(
    frames_arg: str | None = None,
    labels_arg: str | None = None,
    crops_arg: str | None = None,
    root_arg: str | None = None,
) -> list[tuple[str, Path, Path, Path | None]]:
    """Discover folder targets (name, frames_dir, labels_dir, crops_dir)."""
    folders: list[tuple[str, Path, Path, Path | None]] = []

    target_root = Path(root_arg) if root_arg else (Path(frames_arg) if frames_arg else None)
    if target_root is None:
        raise ValueError("Must provide either --frames or --root")

    if target_root.is_dir():
        subdirs = sorted([d for d in target_root.iterdir() if d.is_dir()])
        for sd in subdirs:
            f_dir = sd / "frames" if (sd / "frames").is_dir() else (sd if list(sd.glob("*.jpg")) else None)
            if f_dir and f_dir.is_dir():
                l_dir = sd / "labels" if (sd / "labels").is_dir() else f_dir
                c_dir = sd / "crops" if (sd / "crops").is_dir() else None
                folders.append((sd.name, f_dir, l_dir, c_dir))

    if not folders and frames_arg:
        f_dir = Path(frames_arg)
        l_dir = Path(labels_arg) if labels_arg else (f_dir.parent / "labels" if (f_dir.parent / "labels").is_dir() else f_dir)
        c_dir = Path(crops_arg) if crops_arg else (f_dir.parent / "crops" if (f_dir.parent / "crops").is_dir() else None)
        folder_name = f_dir.parent.name if f_dir.name == "frames" else f_dir.name
        folders.append((folder_name, f_dir, l_dir, c_dir))

    return folders


class ClassPickerDialog:
    """Simple modal list picker used for choosing a class to remove."""

    def __init__(self, parent, classes: list[str], title: str = "Pick a class",
                 prompt: str = "Select the class to remove (its boxes will be deleted\nand higher class ids remapped down):"):
        self.result: int | None = None
        self.top = tk.Toplevel(parent)
        self.top.title(title)
        self.top.transient(parent)
        self.top.grab_set()
        self.top.configure(bg="#333333")
        tk.Label(self.top, text=prompt,
                 bg="#333", fg="white", justify=tk.LEFT).pack(padx=12, pady=(10, 4))
        for idx, name in enumerate(classes):
            btn = tk.Button(self.top, text=f"{idx}: {name}",
                            command=lambda i=idx: self._pick(i),
                            bg="#444", fg="white", relief=tk.FLAT, padx=10, pady=2)
            btn.pack(fill=tk.X, padx=12, pady=1)
        tk.Button(self.top, text="Cancel", command=self.top.destroy,
                  bg="#8b0000", fg="white", relief=tk.FLAT, padx=10).pack(pady=(6, 10))
        self.top.protocol("WM_DELETE_WINDOW", self.top.destroy)

    def _pick(self, idx: int):
        self.result = idx
        self.top.destroy()


class AnnotationEditor:
    def __init__(self, root, folders: list[tuple[str, Path, Path, Path | None]], unreviewed_only: bool = False,
                 classes_file: str | None = None):
        self.root = root
        self.folders = folders
        self.classes_file = classes_file
        if not self.folders:
            raise RuntimeError("No valid folders found to review.")

        self.folder_idx = 0
        self.unreviewed_only = unreviewed_only

        self.index = 0
        self.show_boxes = True
        self.photo = None
        self.scale = 1.0
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.img_w = 0
        self.img_h = 0
        self.base_pil_img = None

        self.undo_stack: list[tuple[int, list[tuple[int, float, float, float, float]]]] = []
        self.draw_start = None
        self.new_box = None
        self.is_drawing = False
        self.active_box_idx = -1
        self.resize_handle = None
        self.pan_start = None
        # Memorized boxes for copy/paste across frames
        self.memorized_boxes: list[tuple[int, float, float, float, float]] = []
        self.show_memorized = True
        # Canvas-space rects of the per-box class badges (for hit-testing)
        self._badge_rects: list[tuple[float, float, float, float, int]] = []
        self.auto_paste = False  # when True, memorized boxes auto-paste on navigation
        # When True, navigating stamps the last-copied frame's classes onto
        # overlapping existing boxes in each new frame (walking propagation)
        self.copy_classes_mode = False
        # Class snapshot taken when 'c' was pressed (for walking class-stamp)
        self._copied_classes: list[tuple[int, float, float, float, float]] = []

        # YOLO multi-class state (classes.txt lives next to the labels)
        self.classes: list[str] = []
        self.current_class = 0


        # Setup window
        self.root.title("YOLO Annotation Editor")
        self.root.geometry("1450x900")
        self.root.configure(bg=COLOR_BG)

        # Top toolbar
        toolbar = tk.Frame(root, bg="#333333", padx=5, pady=3)
        toolbar.pack(fill=tk.X)

        self.btn_prev_fold = tk.Button(toolbar, text="📁◀ Prev Fold", command=self.previous_folder,
                                       bg="#444", fg="white", relief=tk.FLAT, padx=6)
        self.btn_prev_fold.pack(side=tk.LEFT, padx=2)

        self.folder_var = tk.StringVar()
        self.folder_menu = tk.OptionMenu(toolbar, self.folder_var, "")
        self.folder_menu.config(bg="#444", fg="white", activebackground="#555", activeforeground="white", relief=tk.FLAT, highlightthickness=0)
        self.folder_menu["menu"].config(bg="#333", fg="white")
        self.folder_menu.pack(side=tk.LEFT, padx=2)

        self.btn_next_fold = tk.Button(toolbar, text="Fold ▶📁", command=self.next_folder,
                                       bg="#444", fg="white", relief=tk.FLAT, padx=6)
        self.btn_next_fold.pack(side=tk.LEFT, padx=2)

        self.btn_prev = tk.Button(toolbar, text="◀ Prev", command=self.previous,
                                   bg="#444", fg="white", relief=tk.FLAT, padx=8)
        self.btn_prev.pack(side=tk.LEFT, padx=(10, 2))

        self.btn_next = tk.Button(toolbar, text="Next ▶", command=lambda: self.next(auto_mark=True),
                                   bg="#444", fg="white", relief=tk.FLAT, padx=8)
        self.btn_next.pack(side=tk.LEFT, padx=2)

        self.btn_review = tk.Button(toolbar, text="✓ Reviewed", command=self.toggle_reviewed,
                                    bg="#006400", fg="white", relief=tk.FLAT, padx=8)
        self.btn_review.pack(side=tk.LEFT, padx=5)

        self.btn_filter = tk.Button(toolbar, text="Show: All", command=self.toggle_unreviewed_only,
                                    bg="#444", fg="white", relief=tk.FLAT, padx=8)
        self.btn_filter.pack(side=tk.LEFT, padx=2)

        self.btn_delete = tk.Button(toolbar, text="🗑 Delete Image", command=self.delete_image,
                                     bg="#8b0000", fg="white", relief=tk.FLAT, padx=8)
        self.btn_delete.pack(side=tk.LEFT, padx=5)

        self.btn_undo = tk.Button(toolbar, text="↩ Undo", command=self.undo,
                                   bg="#444", fg="white", relief=tk.FLAT, padx=8)
        self.btn_undo.pack(side=tk.LEFT, padx=2)

        self.btn_save = tk.Button(toolbar, text="💾 Save", command=self.save,
                                   bg="#006400", fg="white", relief=tk.FLAT, padx=8)
        self.btn_save.pack(side=tk.LEFT, padx=5)

        self.btn_toggle = tk.Button(toolbar, text="👁 Toggle Boxes", command=self.toggle_boxes,
                                     bg="#444", fg="white", relief=tk.FLAT, padx=8)
        self.btn_toggle.pack(side=tk.LEFT, padx=2)

        tk.Label(toolbar, text="Class:", bg="#333", fg="white").pack(side=tk.LEFT, padx=(10, 2))
        self.class_var = tk.StringVar(value="")
        self.class_menu = tk.OptionMenu(toolbar, self.class_var, "")
        self.class_menu.config(bg="#444", fg="white", activebackground="#555", activeforeground="white", relief=tk.FLAT, highlightthickness=0)
        self.class_menu["menu"].config(bg="#333", fg="white")
        self.class_menu.pack(side=tk.LEFT, padx=2)

        self.btn_add_class = tk.Button(toolbar, text="＋ Class", command=self.add_class_prompt,
                                       bg="#444", fg="white", relief=tk.FLAT, padx=6)
        self.btn_add_class.pack(side=tk.LEFT, padx=2)

        self.btn_del_class = tk.Button(toolbar, text="－ Class", command=self.remove_class_prompt,
                                       bg="#8b5a00", fg="white", relief=tk.FLAT, padx=6)
        self.btn_del_class.pack(side=tk.LEFT, padx=2)

        self.btn_reassign = tk.Button(toolbar, text="⇄ Reassign", command=self.reassign_class_prompt,
                                       bg="#444", fg="white", relief=tk.FLAT, padx=6)
        self.btn_reassign.pack(side=tk.LEFT, padx=2)

        self.btn_propagate = tk.Button(toolbar, text="🏷 Propagate →", command=self.propagate_classes_forward,
                                        bg="#00646b", fg="white", relief=tk.FLAT, padx=6)
        self.btn_propagate.pack(side=tk.LEFT, padx=5)

        tk.Label(toolbar, text="Jump:", bg="#333", fg="white").pack(side=tk.LEFT, padx=(10, 2))
        self.jump_var = tk.StringVar(value="1")
        self.jump_entry = tk.Entry(toolbar, textvariable=self.jump_var, width=5, bg="#555",
                                    fg="white", insertbackground="white")
        self.jump_entry.pack(side=tk.LEFT, padx=2)
        self.jump_entry.bind("<Return>", lambda e: self.jump_to_index())

        # Canvas
        self.canvas = tk.Canvas(root, bg=COLOR_BG, highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.box_count_var = tk.StringVar(value="0")
        tk.Label(self.canvas, textvariable=self.box_count_var, bg=COLOR_BG, fg="#00ff00", 
                 font=("Ubuntu", 32, "bold")).place(x=20, y=20)

        # Per-frame class counts + current draw class (below the big count)
        self.class_count_var = tk.StringVar(value="")
        self.class_count_lbl = tk.Label(self.canvas, textvariable=self.class_count_var, bg=COLOR_BG,
                                        fg="#00ff00", font=("Ubuntu", 12, "bold"), anchor=tk.W, justify=tk.LEFT)
        self.class_count_lbl.place(x=20, y=70)

        # Bottom strip listing all classes
        self.classes_var = tk.StringVar(value="")
        self.classes_lbl = tk.Label(root, textvariable=self.classes_var, bg="#333", fg="#00ff00",
                                    font=("Ubuntu", 10), anchor=tk.W, padx=10, pady=2)
        self.classes_lbl.pack(fill=tk.X)

        # Mouse & Key bindings
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_drag)
        self.canvas.bind("<MouseWheel>", self.on_zoom)
        self.canvas.bind("<Button-4>", self.on_zoom)
        self.canvas.bind("<Button-5>", self.on_zoom)

        self.root.bind("<Left>", lambda e: self.previous())
        self.root.bind("<Right>", lambda e: self.next(auto_mark=True))
        self.root.bind("<space>", lambda e: self.toggle_boxes())
        self.root.bind("<Delete>", lambda e: self.delete_image())
        self.root.bind("<BackSpace>", lambda e: self.delete_image())
        self.root.bind("<s>", lambda e: self.save())
        self.root.bind("<z>", lambda e: self.undo())
        self.root.bind("<r>", lambda e: self.remove_nearest_box())
        self.root.bind("<c>", lambda e: self.copy_and_enable_autopaste())
        self.root.bind("<p>", lambda e: self.toggle_auto_paste())
        self.root.bind("<m>", lambda e: self.remove_memorized_near_cursor())
        self.root.bind("<v>", lambda e: self.toggle_reviewed())
        self.root.bind("<V>", lambda e: self.toggle_reviewed())
        self.root.bind("<u>", lambda e: self.toggle_unreviewed_only())
        self.root.bind("<U>", lambda e: self.toggle_unreviewed_only())
        self.root.bind("<n>", lambda e: None if self._typing_in_field(e) else self.add_class_prompt())
        self.root.bind("<x>", lambda e: None if self._typing_in_field(e) else self.remove_class_prompt())
        self.root.bind("<a>", lambda e: None if self._typing_in_field(e) else self.reassign_class_prompt())
        self.root.bind("<C>", lambda e: None if self._typing_in_field(e) else self.propagate_classes_forward())
        for d in "0123456789":
            self.root.bind(d, lambda e, dg=int(d): None if self._typing_in_field(e) else self._on_digit(dg))
            self.root.bind(f"<KP_{d}>", lambda e, dg=int(d): None if self._typing_in_field(e) else self._on_digit(dg))

        # Status & HUD
        self.status_var = tk.StringVar(value="")
        self.status = tk.Label(root, textvariable=self.status_var, bg="#333",
                               fg=COLOR_STATUS, font=("Ubuntu", 11), anchor=tk.W, padx=10, pady=4)
        self.status.pack(fill=tk.X)

        self.hud_var = tk.StringVar(value="")
        self.hud = tk.Label(root, textvariable=self.hud_var, bg="#333",
                            fg=COLOR_HUD, font=("Ubuntu", 10), anchor=tk.W, padx=10, pady=2)
        self.hud.pack(fill=tk.X)

        self.load_folder(0)
        self.canvas.bind("<Configure>", lambda e: self.display())

    def _cycle_box_class(self, idx: int):
        """Cycle the class of box idx (triggered by clicking its number badge)."""
        if not self.images or self.index >= len(self.images) or len(self.classes) < 2:
            if len(self.classes) < 2:
                self.status_var.set("⚠️ Only one class defined — press 'n' to add more classes first.")
            return
        labels = self.read_labels(self.images[self.index])
        if not (0 <= idx < len(labels)):
            return
        self.push_undo()
        old_cid, xc, yc, w, h = labels[idx]
        new_cid = (old_cid + 1) % len(self.classes)
        labels[idx] = (new_cid, xc, yc, w, h)
        self.write_labels(self.images[self.index], labels)
        self.active_box_idx = idx
        self.status_var.set(f"🏷 Box #{idx}: class {old_cid} → {new_cid} ({self.classes[new_cid]})")
        self.display(redraw_image=False)

    def load_folder(self, folder_idx: int):
        if not (0 <= folder_idx < len(self.folders)):
            return
        self.folder_idx = folder_idx
        folder_name, self.frames_dir, self.labels_dir, self.crops_dir = self.folders[folder_idx]

        self.all_images = sorted(
            p for p in self.frames_dir.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ) if self.frames_dir.exists() else []

        self.reviewed_set = load_reviewed(self.labels_dir)
        self._load_classes()
        self._update_folder_menu()
        self._filter_images()
        self.index = 0
        self.new_box = None
        self.zoom_level = 1.0
        self.active_box_idx = -1
        self.base_pil_img = None
        self.copy_classes_mode = False  # classes copied in another folder don't carry over
        self.display()

    def _update_folder_menu(self):
        menu = self.folder_menu["menu"]
        menu.delete(0, "end")
        for idx, (fname, fdir, ldir, cdir) in enumerate(self.folders):
            r_set = load_reviewed(ldir)
            imgs = [p for p in fdir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS] if fdir.exists() else []
            label_text = f"{fname} ({len(r_set)}/{len(imgs)} reviewed)"
            menu.add_command(label=label_text, command=lambda i=idx: self.load_folder(i))
            if idx == self.folder_idx:
                self.folder_var.set(label_text)

    def _filter_images(self):
        if self.unreviewed_only:
            self.images = [p for p in self.all_images if p.name not in self.reviewed_set]
            self.btn_filter.config(text="Show: Unreviewed Only", bg="#8b5a00")
        else:
            self.images = list(self.all_images)
            self.btn_filter.config(text="Show: All Images", bg="#444")

        if self.index >= len(self.images):
            self.index = max(0, len(self.images) - 1)

    def next_folder(self):
        if self.folder_idx < len(self.folders) - 1:
            self.load_folder(self.folder_idx + 1)

    def previous_folder(self):
        if self.folder_idx > 0:
            self.load_folder(self.folder_idx - 1)

    def toggle_unreviewed_only(self):
        self.unreviewed_only = not self.unreviewed_only
        self._filter_images()
        self.index = 0
        self.base_pil_img = None
        self.display()

    def toggle_reviewed(self):
        if not self.images or self.index >= len(self.images):
            return
        img_name = self.images[self.index].name
        if img_name in self.reviewed_set:
            self.reviewed_set.remove(img_name)
        else:
            self.reviewed_set.add(img_name)
        save_reviewed(self.labels_dir, self.reviewed_set)
        self._update_folder_menu()
        if self.unreviewed_only:
            self._filter_images()
        self.display()

    # ── Class management (YOLO classes.txt) ───────────────────────────

    def _typing_in_field(self, event) -> bool:
        """True if a keyboard event originated in an Entry widget."""
        w = self.root.focus_get()
        return isinstance(w, tk.Entry)

    def _load_classes(self):
        self.classes = load_classes(self.labels_dir, self.classes_file)
        self.current_class = min(self.current_class, len(self.classes) - 1)
        self._update_class_menu()

    def _class_color(self, cls_id: int) -> str:
        return CLASS_PALETTE[cls_id % len(CLASS_PALETTE)]

    def _update_class_menu(self):
        menu = self.class_menu["menu"]
        menu.delete(0, "end")
        for idx, name in enumerate(self.classes):
            menu.add_command(label=f"{idx}: {name}", command=lambda i=idx: self._select_class(i))
        if 0 <= self.current_class < len(self.classes):
            self.class_var.set(f"{self.current_class}: {self.classes[self.current_class]}")

    def _select_class(self, idx: int):
        if 0 <= idx < len(self.classes):
            self.current_class = idx
            self._update_class_menu()
            self.display(redraw_image=False)

    def add_class_prompt(self):
        name = simpledialog.askstring("Add class", "New class name:", parent=self.root)
        if name is None:
            return
        name = name.strip()
        if not name:
            return
        if name in self.classes:
            messagebox.showwarning("Add class", f"Class '{name}' already exists.", parent=self.root)
            return
        self.classes.append(name)
        save_classes(self.labels_dir, self.classes)
        self._update_class_menu()
        self.status_var.set(f"➕ Added class {len(self.classes) - 1}: '{name}'")
        self.display(redraw_image=False)

    def remove_class_prompt(self):
        if not self.classes:
            return
        if len(self.classes) == 1:
            messagebox.showwarning("Remove class", "Cannot remove the last remaining class.", parent=self.root)
            return
        choice = ClassPickerDialog(self.root, self.classes, title="Remove class")
        self.root.wait_window(choice.top)
        if choice.result is None:
            return
        removed_id = choice.result
        removed_name = self.classes.pop(removed_id)
        save_classes(self.labels_dir, self.classes)
        # Remap label files so class ids stay contiguous (0..n-1)
        remapped = self._remap_labels_after_removal(removed_id)
        if self.current_class >= len(self.classes):
            self.current_class = len(self.classes) - 1
        self._update_class_menu()
        self.status_var.set(f"➖ Removed class {removed_id}: '{removed_name}' ({remapped} label files remapped)")
        self.display()

    def _remap_labels_after_removal(self, removed_id: int) -> int:
        """Shift class ids > removed_id down by one across all label files."""
        changed = 0
        if not self.labels_dir.exists():
            return 0
        for lp in self.labels_dir.glob("*.txt"):
            if lp.name == "classes.txt":
                continue
            lines = lp.read_text().strip().splitlines()
            new_lines = []
            file_changed = False
            for line in lines:
                parts = line.split()
                if len(parts) == 5:
                    try:
                        cid = int(parts[0])
                    except ValueError:
                        new_lines.append(line)
                        continue
                    if cid == removed_id:
                        file_changed = True
                        continue  # drop boxes of the removed class
                    if cid > removed_id:
                        parts[0] = str(cid - 1)
                        file_changed = True
                    new_lines.append(" ".join(parts))
            if file_changed:
                lp.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
                changed += 1
        return changed

    def reassign_class_prompt(self):
        """Rewrite every box of one class into another class, across all label
        files in the folder (e.g. turn all class 2 boxes into class 0)."""
        if not self.classes or len(self.classes) < 2:
            self.status_var.set("⚠️ Need at least 2 classes to reassign — press 'n' to add one.")
            return
        src = ClassPickerDialog(self.root, self.classes,
                                title="Reassign class — from",
                                prompt="Reassign all boxes of which class?")
        self.root.wait_window(src.top)
        if src.result is None:
            return
        src_id = src.result
        targets = [c for i, c in enumerate(self.classes) if i != src_id]
        dst = ClassPickerDialog(self.root, targets,
                                title="Reassign class — to",
                                prompt=f"Change every '{src_id}: {self.classes[src_id]}' box into which class?")
        self.root.wait_window(dst.top)
        if dst.result is None:
            return
        dst_id = dst.result if dst.result < src_id else dst.result + 1
        if dst_id == src_id:
            return
        n_boxes, n_files = self._reassign_class_in_all(src_id, dst_id)
        self.display(redraw_image=False)
        if n_boxes == 0:
            self.status_var.set(f"⚠️ No boxes of class {src_id} ({self.classes[src_id]}) found — nothing changed.")
        else:
            self.status_var.set(
                f"⇄ Reassigned {n_boxes} box(es) of class {src_id} ({self.classes[src_id]}) "
                f"→ {dst_id} ({self.classes[dst_id]}) in {n_files} file(s)"
            )

    def _reassign_class_in_all(self, src_id: int, dst_id: int) -> tuple[int, int]:
        """Replace class src_id with dst_id in all label files. Returns (boxes, files)."""
        n_boxes = 0
        n_files = 0
        if not self.labels_dir.exists():
            return 0, 0
        for lp in self.labels_dir.glob("*.txt"):
            if lp.name == "classes.txt":
                continue
            labels = self._read_label_file(lp)
            changed = False
            for i, (cid, xc, yc, w, h) in enumerate(labels):
                if cid == src_id:
                    labels[i] = (dst_id, xc, yc, w, h)
                    n_boxes += 1
                    changed = True
            if changed:
                self._write_label_file(lp, labels)
                n_files += 1
        return n_boxes, n_files

    def _mouse_canvas_coords(self) -> tuple[float, float]:
        """Current mouse position in canvas coordinates."""
        return (
            self.root.winfo_pointerx() - self.canvas.winfo_rootx(),
            self.root.winfo_pointery() - self.canvas.winfo_rooty(),
        )

    def _hit_badge(self, cx: float, cy: float) -> int | None:
        """Return the box idx if (cx, cy) is inside one of the class badges."""
        for bx1, by1, bx2, by2, idx in self._badge_rects:
            if bx1 - 2 <= cx <= bx2 + 2 and by1 - 2 <= cy <= by2 + 2:
                return idx
        return None

    def _box_idx_near_badge(self, radius: float = 35.0,
                            cx: float | None = None, cy: float | None = None) -> int | None:
        """Box whose top-left class badge is nearest the mouse (within radius px)."""
        if not self._badge_rects:
            return None
        if cx is None or cy is None:
            mx, my = self._mouse_canvas_coords()
        else:
            mx, my = cx, cy
        best_idx = None
        best_d = radius
        for bx1, by1, bx2, by2, idx in self._badge_rects:
            bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
            d = ((bcx - mx) ** 2 + (bcy - my) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best_idx = idx
        return best_idx

    def _box_idx_under_cursor(self, cx: float | None = None, cy: float | None = None) -> int | None:
        """Topmost box containing the mouse position, or None."""
        if not self.images or self.index >= len(self.images):
            return None
        if cx is None or cy is None:
            mx, my = self._mouse_canvas_coords()
        else:
            mx, my = cx, cy
        ix, iy = self.canvas_to_image(mx, my)
        labels = self.read_labels(self.images[self.index])
        for idx in reversed(range(len(labels))):
            _, xc, yc, w, h = labels[idx]
            x1 = (xc - w / 2) * self.img_w
            y1 = (yc - h / 2) * self.img_h
            x2 = (xc + w / 2) * self.img_w
            y2 = (yc + h / 2) * self.img_h
            if x1 <= ix <= x2 and y1 <= iy <= y2:
                return idx
        return None

    def _pick_box_clicked_or_badge(self, cx: float | None = None, cy: float | None = None) -> int | None:
        """Box to act on from a click: the box under the cursor (clicked one)
        wins; otherwise the box whose top-left label is nearest the cursor."""
        idx = self._box_idx_under_cursor(cx, cy)
        if idx is not None:
            return idx
        return self._box_idx_near_badge(cx=cx, cy=cy)

    def _pick_box_badge_first(self) -> int | None:
        """Box to copy with 'c': box whose top-left label is nearest the cursor
        wins, then the box under the cursor."""
        idx = self._box_idx_near_badge()
        if idx is None:
            idx = self._box_idx_under_cursor()
        return idx

    def _mem_label_anchor(self, idx: int) -> tuple[float, float]:
        """Canvas anchor (SW) of a memorized box's label — shared by the drawer
        and by the 'm' hit-test so they always agree."""
        _, xc, yc, w, h = self.memorized_boxes[idx]
        cx1, cy1 = self.image_to_canvas((xc - w / 2) * self.img_w, (yc - h / 2) * self.img_h)
        return cx1 + 2, cy1 - 8

    def _on_digit(self, digit: int):
        """Digit pressed: assign that class to the box being pointed at.

        Priority: box whose number badge (top-left) is nearest the mouse →
        box under the cursor → active box. With no box targeted, the digit
        selects the class used for newly drawn boxes.
        """
        if digit >= len(self.classes):
            self.status_var.set(
                f"⚠️ No class with id {digit} — press 'n' to add one "
                f"({len(self.classes)} defined: 0..{len(self.classes) - 1})"
            )
            return
        idx = self._pick_box_badge_first()
        if idx is None and self.active_box_idx >= 0:
            idx = self.active_box_idx
        if idx is not None:
            self.set_box_class(idx, digit)
        else:
            self._select_class(digit)
            self.status_var.set(f"✎ Draw class set to {digit}: {self.classes[digit]}")

    def set_box_class(self, idx: int, cls_id: int):
        """Assign class cls_id to box idx of the current frame."""
        if not self.images or self.index >= len(self.images) or not (0 <= cls_id < len(self.classes)):
            return
        labels = self.read_labels(self.images[self.index])
        if not (0 <= idx < len(labels)):
            return
        self.push_undo()
        old_cid, xc, yc, w, h = labels[idx]
        labels[idx] = (cls_id, xc, yc, w, h)
        self.write_labels(self.images[self.index], labels)
        self.active_box_idx = idx
        self.status_var.set(f"🏷 Box #{idx}: class {old_cid} → {cls_id} ({self.classes[cls_id]})")
        self.display(redraw_image=False)

    # ── Class propagation forward ──────────────────────────────────────

    def _propagate_count(self, ask: bool = True) -> int:
        """How many frames forward to propagate. -1 if cancelled."""
        if not ask:
            return len(self.images) - self.index - 1
        # Default: all remaining frames in the current folder
        default_n = len(self.images) - self.index - 1
        answer = simpledialog.askstring(
            "Propagate classes",
            f"Propagate classes of '{self.images[self.index].name}' forward to how many frames?\n"
            f"(matching boxes by overlap, {default_n} frames remain in this folder)",
            initialvalue=str(default_n),
            parent=self.root,
        )
        if answer is None:
            return -1
        try:
            n = max(0, int(answer))
        except ValueError:
            return -1
        return min(n, default_n)

    def propagate_classes_forward(self):
        """Copy class assignments of the current frame onto overlapping boxes in
        the next N frames (matched by IoU). Box geometry is left untouched —
        only the class id is overridden."""
        if not self.images or self.index >= len(self.images):
            return
        src_labels = self.read_labels(self.images[self.index])
        if not src_labels:
            self.status_var.set("⚠️ No boxes in current frame — nothing to propagate.")
            return
        n_frames = self._propagate_count()
        if n_frames <= 0:
            if n_frames == 0:
                self.status_var.set("⚠️ No frames after the current one to propagate to.")
            return
        n_changed = 0
        n_frames_touched = 0
        for off in range(1, n_frames + 1):
            tgt_img = self.images[self.index + off]
            tgt_labels = self.read_labels(tgt_img)
            if not tgt_labels:
                continue
            new_labels = list(tgt_labels)
            frame_changed = False
            for j, (cid, xc, yc, w, h) in enumerate(new_labels):
                best_iou = 0.0
                best_cid = None
                for (scid, sxc, syc, sw, sh) in src_labels:
                    iou = self._iou_xywh(xc, yc, w, h, sxc, syc, sw, sh)
                    if iou > best_iou:
                        best_iou = iou
                        best_cid = scid
                if best_cid is not None and best_iou >= PROPAGATE_IOU_THRESHOLD and best_cid != cid:
                    new_labels[j] = (best_cid, xc, yc, w, h)
                    frame_changed = True
                    n_changed += 1
            if frame_changed:
                self.write_labels(tgt_img, new_labels)
                n_frames_touched += 1
        self.display(redraw_image=False)
        self.status_var.set(
            f"🏷 Propagated classes {n_frames} frame(s) forward: "
            f"{n_changed} box(es) reassigned in {n_frames_touched} frame(s) "
            f"(IoU ≥ {PROPAGATE_IOU_THRESHOLD})"
        )

    @staticmethod
    def _iou_xywh(xc1, yc1, w1, h1, xc2, yc2, w2, h2) -> float:
        """IoU of two boxes given as normalized (xc, yc, w, h)."""
        ax1, ay1 = xc1 - w1 / 2, yc1 - h1 / 2
        ax2, ay2 = xc1 + w1 / 2, yc1 + h1 / 2
        bx1, by1 = xc2 - w2 / 2, yc2 - h2 / 2
        bx2, by2 = xc2 + w2 / 2, yc2 + h2 / 2
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0.0

    # ── Label I/O ──────────────────────────────────────────────────────

    def _label_path(self, image_path: Path) -> Path:
        return self.labels_dir / (image_path.stem + ".txt")

    def read_labels(self, image_path: Path) -> list[tuple[int, float, float, float, float]]:
        """Read YOLO labels. Returns list of (class_id, xc, yc, w, h) normalized."""
        return self._read_label_file(self._label_path(image_path))

    def _read_label_file(self, lp: Path, clamp_ids: bool = True) -> list[tuple[int, float, float, float, float]]:
        """Read a YOLO label file by path."""
        if not lp.exists():
            return []
        labels = []
        for line in lp.read_text().strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    cls_id = int(float(parts[0]))
                    xc, yc, w, h = map(float, parts[1:5])
                    # Clamp out-of-range class ids to the last known class
                    if clamp_ids and not (0 <= cls_id < len(self.classes)):
                        cls_id = max(0, len(self.classes) - 1)
                    labels.append((cls_id, xc, yc, w, h))
                except ValueError:
                    pass
        return labels

    def write_labels(self, image_path: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
        """Write YOLO labels ('class_id xc yc w h')."""
        self._write_label_file(self._label_path(image_path), labels)

    def _write_label_file(self, lp: Path, labels: list[tuple[int, float, float, float, float]]) -> None:
        """Write a YOLO label file by path."""
        if labels:
            lines = [f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for cid, xc, yc, w, h in labels]
            lp.write_text("\n".join(lines) + "\n")
        elif lp.exists():
            lp.unlink()

    def push_undo(self) -> None:
        """Save current state for undo."""
        if not self.images or self.index >= len(self.images):
            return
        img = self.images[self.index]
        labels = self.read_labels(img)
        self.undo_stack.append((self.index, labels))
        if len(self.undo_stack) > 100:
            self.undo_stack.pop(0)

    # ── Coordinate transforms ──────────────────────────────────────────

    def canvas_to_image(self, cx: float, cy: float) -> tuple[float, float]:
        """Convert canvas coords to original image coords."""
        ix = (cx - self.offset_x) / self.scale
        iy = (cy - self.offset_y) / self.scale
        return ix, iy

    def image_to_canvas(self, ix: float, iy: float) -> tuple[float, float]:
        """Convert original image coords to canvas coords."""
        cx = ix * self.scale + self.offset_x
        cy = iy * self.scale + self.offset_y
        return cx, cy

    # ── Display ────────────────────────────────────────────────────────

    def display(self, redraw_image=True):
        if not self.images:
            self.canvas.delete("all")
            fname_label = self.folders[self.folder_idx][0] if self.folders else ""
            cw = max(self.canvas.winfo_width(), 600)
            ch = max(self.canvas.winfo_height(), 400)
            self.canvas.create_text(
                cw // 2,
                ch // 2,
                text=f"All images in folder '{fname_label}' have been reviewed!\nPress 'Fold ▶' or 'u' to toggle filter.",
                fill="#00ff00",
                font=("Ubuntu", 18, "bold"),
                justify=tk.CENTER,
            )
            self.status_var.set(f"Folder: {fname_label}  |  0 unreviewed images")
            self.hud_var.set("v=toggle reviewed | u=unreviewed filter | Folders navigation")
            return

        image_path = self.images[self.index]
        MAX_RENDER = 4096  # cap rendered image to avoid tkinter OOM

        if redraw_image or self.base_pil_img is None:
            if self.base_pil_img is None or redraw_image:
                self.base_pil_img = Image.open(image_path).convert("RGB")
                self.img_w, self.img_h = self.base_pil_img.size

            canvas_w = max(self.canvas.winfo_width(), 100)
            canvas_h = max(self.canvas.winfo_height(), 100)
            
            base_scale = min(canvas_w / self.img_w, canvas_h / self.img_h)
            self.scale = base_scale * self.zoom_level
            
            render_w = int(self.img_w * self.scale)
            render_h = int(self.img_h * self.scale)

            # If render exceeds cap, crop to visible viewport before resize
            if render_w > MAX_RENDER or render_h > MAX_RENDER:
                # Compute which part of the source image is visible on canvas
                vis_src_x1 = max(0, -self.offset_x / self.scale)
                vis_src_y1 = max(0, -self.offset_y / self.scale)
                vis_src_x2 = min(self.img_w, (canvas_w - self.offset_x) / self.scale)
                vis_src_y2 = min(self.img_h, (canvas_h - self.offset_y) / self.scale)

                crop_box = (int(vis_src_x1), int(vis_src_y1),
                            min(self.img_w, int(vis_src_x2) + 1),
                            min(self.img_h, int(vis_src_y2) + 1))
                cropped = self.base_pil_img.crop(crop_box)
                crop_w = crop_box[2] - crop_box[0]
                crop_h = crop_box[3] - crop_box[1]
                target_w = min(MAX_RENDER, int(crop_w * self.scale))
                target_h = min(MAX_RENDER, int(crop_h * self.scale))
                resized_img = cropped.resize((target_w, target_h), Image.Resampling.BILINEAR)
                # Offset so cropped region aligns with canvas
                self.photo = ImageTk.PhotoImage(resized_img)
                self._crop_offset_x = vis_src_x1 * self.scale + self.offset_x
                self._crop_offset_y = vis_src_y1 * self.scale + self.offset_y
            else:
                new_size = (render_w, render_h)
                resized_img = self.base_pil_img.resize(new_size, Image.Resampling.BILINEAR)
                if new_size[0] < canvas_w:
                    self.offset_x = (canvas_w - new_size[0]) / 2
                if new_size[1] < canvas_h:
                    self.offset_y = (canvas_h - new_size[1]) / 2
                self.photo = ImageTk.PhotoImage(resized_img)
                self._crop_offset_x = self.offset_x
                self._crop_offset_y = self.offset_y

        self.canvas.delete("all")
        img_off_x = getattr(self, '_crop_offset_x', self.offset_x)
        img_off_y = getattr(self, '_crop_offset_y', self.offset_y)
        self.canvas.create_image(img_off_x, img_off_y, image=self.photo, anchor=tk.NW)

        labels = self.read_labels(image_path)
        self.box_count_var.set(f"{len(labels)}")
        self._badge_rects = []

        if self.show_boxes:
            for idx, (cls_id, xc, yc, w, h) in enumerate(labels):
                x1 = (xc - w / 2) * self.img_w
                y1 = (yc - h / 2) * self.img_h
                x2 = (xc + w / 2) * self.img_w
                y2 = (yc + h / 2) * self.img_h
                
                cx1, cy1 = self.image_to_canvas(x1, y1)
                cx2, cy2 = self.image_to_canvas(x2, y2)
                
                color = self._class_color(cls_id)
                width = 2
                if idx == self.active_box_idx:
                    color = COLOR_BOX_SELECTED
                    width = 3
                
                self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=color, width=width, tags=f"box_{idx}")
                
                # Clickable class badge (top-left corner): click to cycle the class
                cls_name = self.classes[cls_id] if 0 <= cls_id < len(self.classes) else str(cls_id)
                badge_text = f"{cls_id}"
                bx1, by1 = cx1 - 1, cy1 - 1
                bx2 = bx1 + max(14, 8 * len(badge_text) + 6)
                by2 = by1 + 16
                self.canvas.create_rectangle(bx1, by1, bx2, by2, fill=color, outline="black", tags=f"class_badge_{idx}")
                self.canvas.create_text((bx1 + bx2) / 2, (by1 + by2) / 2, text=badge_text,
                                        fill="black", font=("Ubuntu", 9, "bold"), tags=f"class_badge_{idx}")
                self.canvas.create_text(bx2 + 4, (by1 + by2) / 2, text=cls_name, fill=color,
                                        anchor=tk.W, font=("Ubuntu", 9), tags=f"cls_label_{idx}")
                self.canvas.tag_bind(f"class_badge_{idx}", "<Button-1>",
                                     lambda e, i=idx: self._cycle_box_class(i))
                self.canvas.tag_bind(f"cls_label_{idx}", "<Button-1>",
                                     lambda e, i=idx: self._cycle_box_class(i))
                self._badge_rects.append((bx1, by1, bx2, by2, idx))
                
                if idx == self.active_box_idx:
                    self._draw_handles(cx1, cy1, cx2, cy2)

        if self.show_memorized and self.memorized_boxes:
            for idx, (cls_id, xc, yc, w, h) in enumerate(self.memorized_boxes):
                x1 = (xc - w / 2) * self.img_w
                y1 = (yc - h / 2) * self.img_h
                x2 = (xc + w / 2) * self.img_w
                y2 = (yc + h / 2) * self.img_h
                cx1, cy1 = self.image_to_canvas(x1, y1)
                cx2, cy2 = self.image_to_canvas(x2, y2)
                # Ghost marker: fixed-color dashed outline offset outward so it
                # stays clearly distinguishable from the real box underneath
                off = 4
                self.canvas.create_rectangle(cx1 - off, cy1 - off, cx2 + off, cy2 + off,
                                             outline=COLOR_MEM, width=2, dash=(7, 5), tags=f"mem_{idx}")
                mcolor = self._class_color(cls_id)
                lx, ly = self._mem_label_anchor(idx)
                self.canvas.create_text(lx, ly, text=f"mem#{idx} c{cls_id}",
                                        fill=mcolor, anchor=tk.SW, font=("Ubuntu", 8, "bold"))

        if self.new_box:
            x1, y1, x2, y2 = self.new_box
            cx1, cy1 = self.image_to_canvas(x1, y1)
            cx2, cy2 = self.image_to_canvas(x2, y2)
            self.canvas.create_rectangle(cx1, cy1, cx2, cy2, outline=COLOR_BOX_NEW, width=2)

        # HUD: include class info
        cur_cls = self.classes[self.current_class] if 0 <= self.current_class < len(self.classes) else "?"
        cur_color = self._class_color(self.current_class)
        class_counts: dict[int, int] = {}
        for cid, *_ in labels:
            class_counts[cid] = class_counts.get(cid, 0) + 1
        counts_str = " ".join(f"{cid}:{n}" for cid, n in sorted(class_counts.items())) or "-"
        class_list = " | ".join(f"{i}:{n}" for i, n in enumerate(self.classes))
        self.class_count_var.set(f"{counts_str}   ✎ draw class: {self.current_class} ({cur_cls})")
        self.class_count_lbl.config(fg=cur_color)

        is_rev = image_path.name in self.reviewed_set
        if is_rev:
            self.btn_review.config(text="✓ Reviewed", bg="#006400")
            rev_badge = " [✓ Reviewed]"
        else:
            self.btn_review.config(text="Mark Reviewed", bg="#444")
            rev_badge = " [Unreviewed]"

        fname_label = self.folders[self.folder_idx][0] if self.folders else ""
        n_labels = len(labels)
        label_status = f"{n_labels} boxes" if n_labels > 0 else "no labels"
        self.status_var.set(
            f"Folder: {fname_label}  |  [{self.index + 1}/{len(self.images)}]  {image_path.name}{rev_badge}  "
            f"  {self.img_w}×{self.img_h}  Zoom: {self.zoom_level:.1f}x  {label_status}"
        )
        autopaste_badge = " 🟢 AUTO-PASTE" if self.auto_paste else ""
        mem_info = f" | 📋 memorized: {len(self.memorized_boxes)}{autopaste_badge}" if self.memorized_boxes else ""
        self.hud_var.set(
            f"←/→ navigate | Click-drag add | r=remove | Del=delete | z=undo{mem_info} | "
            "c=copy (label-nearest) | p=toggle auto-paste | m=drop memorized (label-nearest)"
        )
        self.classes_var.set(f"Classes: {class_list}   |   0-9: assign class to pointed box (nearest top-left number wins; none → draw class)   "
                             f"click box number = cycle class   n=add class   x=remove class")

    def _draw_handles(self, x1, y1, x2, y2):
        r = 4
        coords = [
            (x1, y1, "nw"), (x2, y1, "ne"), (x1, y2, "sw"), (x2, y2, "se"),
            ((x1+x2)/2, y1, "n"), ((x1+x2)/2, y2, "s"),
            (x1, (y1+y2)/2, "w"), (x2, (y1+y2)/2, "e")
        ]
        for hx, hy, cursor in coords:
            self.canvas.create_oval(hx-r, hy-r, hx+r, hy+r, fill=COLOR_BOX_NEW, outline="black")

    def on_zoom(self, event):
        if event.num == 4 or event.delta > 0:
            factor = 1.2
        else:
            factor = 0.8
        
        mx, my = event.x, event.y
        ix, iy = self.canvas_to_image(mx, my)
        
        self.zoom_level *= factor
        self.zoom_level = max(0.1, min(self.zoom_level, 20.0))
        
        canvas_w = max(self.canvas.winfo_width(), 100)
        canvas_h = max(self.canvas.winfo_height(), 100)
        base_scale = min(canvas_w / self.img_w, canvas_h / self.img_h)
        self.scale = base_scale * self.zoom_level
        
        self.offset_x = mx - ix * self.scale
        self.offset_y = my - iy * self.scale
        
        self.display()

    def on_pan_start(self, event):
        self.pan_start = (event.x, event.y)

    def on_pan_drag(self, event):
        if self.pan_start:
            dx = event.x - self.pan_start[0]
            dy = event.y - self.pan_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self.pan_start = (event.x, event.y)
            self.display()

    # ── Navigation ─────────────────────────────────────────────────────

    def _navigate_to(self, new_idx: int):
        """Move to a new frame index, auto-paste if enabled, then display."""
        self.index = new_idx
        self.new_box = None
        self.zoom_level = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.active_box_idx = -1
        self.base_pil_img = None
        if self.auto_paste and self.memorized_boxes:
            self._auto_paste_to_current()
        if self.copy_classes_mode:
            self._stamp_copied_classes()
        self.display()

    def _copy_classes_snapshot(self) -> list[tuple[int, float, float, float, float]]:
        """Snapshot of the current frame's boxes (class ids + geometry)."""
        if not self.images or self.index >= len(self.images):
            return []
        return list(self.read_labels(self.images[self.index]))

    def _stamp_copied_classes(self):
        """Stamp the copied frame's classes onto overlapping existing boxes of
        the current frame (geometry untouched, classes only)."""
        if not self._copied_classes:
            return
        if not self.images or self.index >= len(self.images):
            return
        img = self.images[self.index]
        labels = self.read_labels(img)
        if not labels:
            return
        changed = 0
        for j, (cid, xc, yc, w, h) in enumerate(labels):
            best_iou, best_cid = 0.0, None
            for (scid, sxc, syc, sw, sh) in self._copied_classes:
                iou = self._iou_xywh(xc, yc, w, h, sxc, syc, sw, sh)
                if iou > best_iou:
                    best_iou, best_cid = iou, scid
            if best_cid is not None and best_iou >= PROPAGATE_IOU_THRESHOLD and best_cid != cid:
                labels[j] = (best_cid, xc, yc, w, h)
                changed += 1
        if changed:
            self.write_labels(img, labels)
            self.status_var.set(f"🏷 Stamped {changed} box class(es) from copied frame — {img.name}")

    def _auto_paste_to_current(self):
        """Write memorized boxes into the current frame (skipping duplicates)."""
        if not self.images or self.index >= len(self.images):
            return
        img = self.images[self.index]
        labels = self.read_labels(img)
        changed = False
        for box in self.memorized_boxes:
            cls_id, xc, yc, w, h = box
            is_dup = any(
                (abs(ex_xc - xc) < 0.005 and abs(ex_yc - yc) < 0.005)
                or self._iou_xywh(xc, yc, w, h, ex_xc, ex_yc, ex_w, ex_h) >= PROPAGATE_IOU_THRESHOLD
                for _, ex_xc, ex_yc, ex_w, ex_h in labels
            )
            if not is_dup:
                labels.append(box)
                changed = True
        if changed:
            self.write_labels(img, labels)

    def next(self, auto_mark: bool = False):
        if self.images and 0 <= self.index < len(self.images):
            if auto_mark:
                img_name = self.images[self.index].name
                if img_name not in self.reviewed_set:
                    self.reviewed_set.add(img_name)
                    save_reviewed(self.labels_dir, self.reviewed_set)
                    self._update_folder_menu()

        if self.index < len(self.images) - 1:
            self._navigate_to(self.index + 1)
        elif self.folder_idx < len(self.folders) - 1:
            self.next_folder()

    def previous(self):
        if self.index > 0:
            self._navigate_to(self.index - 1)

    def jump_to_index(self):
        try:
            idx = int(self.jump_var.get()) - 1
            if 0 <= idx < len(self.images):
                self._navigate_to(idx)
        except ValueError:
            pass

    def toggle_boxes(self):
        self.show_boxes = not self.show_boxes
        self.display()

    # ── Delete ─────────────────────────────────────────────────────────

    def delete_image(self):
        if not self.images or self.index >= len(self.images):
            return
        img = self.images[self.index]
        self.push_undo()

        lp = self._label_path(img)
        if lp.exists():
            lp.unlink()

        img.unlink()

        if self.crops_dir and self.crops_dir.exists():
            for crop in self.crops_dir.glob(f"{img.stem}_plate*.jpg"):
                crop.unlink()

        if img.name in self.reviewed_set:
            self.reviewed_set.remove(img.name)
            save_reviewed(self.labels_dir, self.reviewed_set)

        self.images.pop(self.index)
        if img in self.all_images:
            self.all_images.remove(img)

        if not self.images:
            messagebox.showinfo("Done", "No more images in this filter/folder!")
            self._filter_images()
            self.display()
            return

        if self.index >= len(self.images):
            self.index = len(self.images) - 1

        self.new_box = None
        self.zoom_level = 1.0
        self.active_box_idx = -1
        self.display()

    # ── Mouse Handlers ───────────────────────────────────────────────

    def on_mouse_down(self, event):
        if not self.images or self.index >= len(self.images):
            return
        ix, iy = self.canvas_to_image(event.x, event.y)

        # Class badge click: let its tag binding cycle the class instead of
        # selecting the box / starting a corner-resize drag
        if self._hit_badge(event.x, event.y) is not None:
            return

        if self.active_box_idx != -1:
            handle = self._get_resize_handle(event.x, event.y)
            if handle:
                self.resize_handle = handle
                return

        sel = self._pick_box_clicked_or_badge(event.x, event.y)
        if sel is not None:
            self.active_box_idx = sel
            self.display(redraw_image=False)
            return

        self.active_box_idx = -1

        if 0 <= ix <= self.img_w and 0 <= iy <= self.img_h:
            if not self.is_drawing:
                self.draw_start = (ix, iy)
                self.is_drawing = True
                self.new_box = (ix, iy, ix, iy)
            else:
                self._finish_drawing(ix, iy)
            self.display(redraw_image=False)

    def on_mouse_drag(self, event):
        if not self.images or self.index >= len(self.images):
            return
        ix, iy = self.canvas_to_image(event.x, event.y)
        ix = max(0, min(self.img_w, ix))
        iy = max(0, min(self.img_h, iy))

        if self.resize_handle and self.active_box_idx != -1:
            self._do_resize(ix, iy)
            self.display(redraw_image=False)
            return

        if self.is_drawing and self.draw_start:
            sx, sy = self.draw_start
            self.new_box = (min(sx, ix), min(sy, iy), max(sx, ix), max(sy, iy))
            self.display(redraw_image=False)

    def on_mouse_up(self, event):
        if self.resize_handle:
            self.resize_handle = None
            return

        if self.is_drawing and self.draw_start:
            ix, iy = self.canvas_to_image(event.x, event.y)
            sx, sy = self.draw_start
            if abs(ix - sx) > 5 or abs(iy - sy) > 5:
                self._finish_drawing(ix, iy)

    def _finish_drawing(self, ix, iy):
        sx, sy = self.draw_start
        self.draw_start = None
        self.is_drawing = False

        x1, y1 = min(sx, ix), min(sy, iy)
        x2, y2 = max(sx, ix), max(sy, iy)
        
        if (x2 - x1) < 5 or (y2 - y1) < 5:
            self.new_box = None
            self.display()
            return

        xc = ((x1 + x2) / 2) / self.img_w
        yc = ((y1 + y2) / 2) / self.img_h
        w = (x2 - x1) / self.img_w
        h = (y2 - y1) / self.img_h

        self.push_undo()
        labels = self.read_labels(self.images[self.index])
        labels.append((self.current_class, xc, yc, w, h))
        self.write_labels(self.images[self.index], labels)
        self.new_box = None
        self.display()

    def _get_resize_handle(self, cx, cy):
        if not self.images or self.index >= len(self.images):
            return None
        idx = self.active_box_idx
        labels = self.read_labels(self.images[self.index])
        if idx < 0 or idx >= len(labels):
            return None
        _, xc, yc, w, h = labels[idx]
        x1, y1 = self.image_to_canvas((xc - w/2)*self.img_w, (yc - h/2)*self.img_h)
        x2, y2 = self.image_to_canvas((xc + w/2)*self.img_w, (yc + h/2)*self.img_h)
        
        r = 10
        if abs(cx-x1)<r and abs(cy-y1)<r: return "nw"
        if abs(cx-x2)<r and abs(cy-y1)<r: return "ne"
        if abs(cx-x1)<r and abs(cy-y2)<r: return "sw"
        if abs(cx-x2)<r and abs(cy-y2)<r: return "se"
        if abs(cy-y1)<r and x1<cx<x2: return "n"
        if abs(cy-y2)<r and x1<cx<x2: return "s"
        if abs(cx-x1)<r and y1<cy<y2: return "w"
        if abs(cx-x2)<r and y1<cy<y2: return "e"
        return None

    def _do_resize(self, ix, iy):
        if not self.images or self.index >= len(self.images):
            return
        labels = self.read_labels(self.images[self.index])
        if self.active_box_idx < 0 or self.active_box_idx >= len(labels):
            return
        _, xc, yc, w, h = labels[self.active_box_idx]
        x1 = (xc - w/2) * self.img_w
        y1 = (yc - h/2) * self.img_h
        x2 = (xc + w/2) * self.img_w
        y2 = (yc + h/2) * self.img_h

        if "n" in self.resize_handle: y1 = iy
        if "s" in self.resize_handle: y2 = iy
        if "w" in self.resize_handle: x1 = ix
        if "e" in self.resize_handle: x2 = ix

        nx1, nx2 = min(x1, x2), max(x1, x2)
        ny1, ny2 = min(y1, y2), max(y1, y2)
        
        cls_id = labels[self.active_box_idx][0]
        labels[self.active_box_idx] = (
            cls_id,
            ((nx1 + nx2) / 2) / self.img_w,
            ((ny1 + ny2) / 2) / self.img_h,
            (nx2 - nx1) / self.img_w,
            (ny2 - ny1) / self.img_h
        )
        self.write_labels(self.images[self.index], labels)

    def on_mouse_move(self, event):
        if self.is_drawing:
            ix, iy = self.canvas_to_image(event.x, event.y)
            sx, sy = self.draw_start
            self.new_box = (min(sx, ix), min(sy, iy), max(sx, ix), max(sy, iy))
            self.display(redraw_image=False)
            return

        handle = self._get_resize_handle(event.x, event.y) if self.active_box_idx != -1 else None
        if handle:
            cursors = {"n":"sb_v_double_arrow", "s":"sb_v_double_arrow", 
                       "e":"sb_h_double_arrow", "w":"sb_h_double_arrow",
                       "nw":"top_left_corner", "se":"bottom_right_corner",
                       "ne":"top_right_corner", "sw":"bottom_left_corner"}
            self.canvas.config(cursor=cursors.get(handle, ""))
        else:
            self.canvas.config(cursor="")

    # ── Copy / Paste memorized boxes ──────────────────────────────────

    @staticmethod
    def _same_box(a, b, tol: float = 0.01) -> bool:
        """True if two boxes describe the same (class, geometry)."""
        return (a[0] == b[0]
                and abs(a[1] - b[1]) <= tol
                and abs(a[2] - b[2]) <= tol
                and abs(a[3] - b[3]) <= tol
                and abs(a[4] - b[4]) <= tol)

    def copy_and_enable_autopaste(self):
        """Copy the box whose label (top-left number) is nearest the cursor to
        memory and enable auto-paste. The same box is never memorized twice."""
        if not self.images or self.index >= len(self.images):
            return
        labels = self.read_labels(self.images[self.index])
        if not labels:
            return

        # Priority: label (top-left number) nearest the cursor → box under the
        # cursor → box currently selected
        idx = self._pick_box_badge_first()
        if idx is None and self.active_box_idx >= 0:
            idx = self.active_box_idx
        if idx is None or idx >= len(labels):
            self.status_var.set("⚠️ Point at a box (or its top-left label) and press 'c' to copy it.")
            return

        box = labels[idx]
        already = any(self._same_box(box, m) for m in self.memorized_boxes)
        if not already:
            self.memorized_boxes.append(box)
        m_idx = next(i for i, m in enumerate(self.memorized_boxes) if self._same_box(m, box))
        self.auto_paste = True
        # Remember this frame's classes; navigating now stamps them onto
        # overlapping boxes in each frame you pass through
        self._copied_classes = self._copy_classes_snapshot()
        self.copy_classes_mode = True
        head = "📋 Already memorized" if already else f"📋 Copied box #{m_idx}"
        self.status_var.set(
            f"{head} + AUTO-PASTE ON ({len(self.memorized_boxes)} memorized) — "
            f"boxes auto-paste and classes stamp as you navigate ('m' to drop)"
        )
        self.display(redraw_image=False)

    def toggle_auto_paste(self):
        """Toggle auto-paste mode. Also paste memorized boxes into current frame."""
        if not self.memorized_boxes:
            self.status_var.set("⚠️ No memorized boxes. Use 'c' to copy a box first.")
            return

        self.auto_paste = not self.auto_paste

        if self.auto_paste:
            self.copy_classes_mode = True
            # Also paste into current frame right away
            self.push_undo()
            self._auto_paste_to_current()
            self._stamp_copied_classes()
            self.status_var.set(
                f"🟢 AUTO-PASTE ON — {len(self.memorized_boxes)} memorized box(es) "
                f"paste + copied classes stamp on every frame"
            )
        else:
            self.copy_classes_mode = False
            self.status_var.set(
                f"🔴 AUTO-PASTE OFF — memorized boxes kept in memory ({len(self.memorized_boxes)} memorized)"
            )
        self.display()

    def remove_memorized_near_cursor(self):
        """Drop the memorized box whose label is nearest the cursor (keeps
        already-pasted frames intact)."""
        if not self.memorized_boxes:
            if self.copy_classes_mode:
                self.copy_classes_mode = False
                self.status_var.set("🔴 Class-stamp mode OFF (no memorized boxes left).")
                self.display(redraw_image=False)
            else:
                self.status_var.set("⚠️ No memorized boxes to remove.")
            return

        mx, my = self._mouse_canvas_coords()
        best_idx = None
        best_d = 100.0
        for i in range(len(self.memorized_boxes)):
            lx, ly = self._mem_label_anchor(i)
            d = ((lx - mx) ** 2 + (ly - my) ** 2) ** 0.5
            if d < best_d:
                best_d = d
                best_idx = i
        if best_idx is None:
            self.status_var.set("ℹ️ Point near a copied box's label (mem#) and press 'm' to drop it.")
            return

        removed = self.memorized_boxes.pop(best_idx)
        self.copy_classes_mode = False
        if not self.memorized_boxes:
            self.auto_paste = False
        self.status_var.set(
            f"🗑 Removed memorized box #{best_idx} (center: {removed[1]:.3f}, {removed[2]:.3f}, class {removed[0]}) — "
            f"already-pasted frames kept | {len(self.memorized_boxes)} memorized remaining"
        )
        self.display(redraw_image=False)

    # ── Remove nearest box ────────────────────────────────────────────

    def remove_nearest_box(self):
        if not self.images or self.index >= len(self.images):
            return
        labels = self.read_labels(self.images[self.index])
        if not labels:
            return

        self.push_undo()

        if len(labels) == 1:
            self.write_labels(self.images[self.index], [])
            self.active_box_idx = -1
            self.display(redraw_image=False)
            return

        # Same pick as a mouse click: the box under the cursor wins, otherwise
        # the box whose top-left label is nearest the cursor
        sel = self._pick_box_clicked_or_badge()
        if sel is not None:
            labels.pop(sel)
            self.write_labels(self.images[self.index], labels)
            self.active_box_idx = -1
            self.display(redraw_image=False)
            return

        # Fallback: nearest box to the cursor by rectangle distance
        mx = self.root.winfo_pointerx() - self.canvas.winfo_rootx()
        my = self.root.winfo_pointery() - self.canvas.winfo_rooty()
        ix, iy = self.canvas_to_image(mx, my)

        nearest_idx = -1
        nearest_dist = float("inf")
        for idx, (_, xc, yc, w, h) in enumerate(labels):
            bx = (xc - w / 2) * self.img_w
            by = (yc - h / 2) * self.img_h
            bw = w * self.img_w
            bh = h * self.img_h
            dx = max(bx - ix, 0, ix - (bx + bw))
            dy = max(by - iy, 0, iy - (by + bh))
            d = (dx**2 + dy**2) ** 0.5
            if d < nearest_dist:
                nearest_dist = d
                nearest_idx = idx

        if nearest_idx >= 0:
            labels.pop(nearest_idx)
            self.write_labels(self.images[self.index], labels)
            self.active_box_idx = -1
            self.display(redraw_image=False)

    # ── Undo ───────────────────────────────────────────────────────────

    def undo(self):
        if not self.undo_stack or not self.images:
            return
        idx, old_labels = self.undo_stack.pop()
        if 0 <= idx < len(self.images):
            self.index = idx
            self.write_labels(self.images[idx], old_labels)
            self.new_box = None
            self.display()

    # ── Save ───────────────────────────────────────────────────────────

    def save(self):
        self.status_var.set("✅ All changes are saved automatically!")
        self.display()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--frames", default=None, help="Directory with frame images or root output directory")
    p.add_argument("--labels", default=None, help="Directory with YOLO label .txt files")
    p.add_argument("--crops", default=None, help="Directory with crop images (optional)")
    p.add_argument("--root", default=None, help="Root directory containing folder targets")
    p.add_argument("--classes", default=None,
                   help="Explicit class list file (.txt one name per line, or .yaml/.yml with "
                        "a 'names' section like YOLO data.yaml). If omitted, classes.txt next to "
                        "the labels is used, then a nearby data.yaml, and finally a default "
                        "classes.txt is created if none is found.")
    p.add_argument("--unreviewed-only", "--only-unreviewed", dest="unreviewed_only", action="store_true",
                   help="Show only unreviewed images")
    args = p.parse_args()

    folders = discover_folders(args.frames, args.labels, args.crops, args.root)
    if not folders:
        print("[ERROR] No valid folders found.")
        sys.exit(1)

    root = tk.Tk()
    editor = AnnotationEditor(root, folders, unreviewed_only=args.unreviewed_only, classes_file=args.classes)
    root.mainloop()


if __name__ == "__main__":
    main()

