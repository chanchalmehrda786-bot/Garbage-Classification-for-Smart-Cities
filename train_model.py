"""
Train SmartWaste classifier — Garbage Classification for Smart Cities.

Dataset (O = Organic/Biodegradable, R = Recyclable/Non-Biodegradable):
  ../../Dataset/DATASET/TRAIN/O|R
  ../../Waste Classification Dataset/DATASET/TRAIN/O|R

Usage:
  python train_model.py --quick
  python train_model.py --data_dir "..\..\Dataset\DATASET\TRAIN"
"""

import argparse
import json
import os
import random
import shutil
import sys
import tempfile

import tensorflow as tf
from tensorflow import keras

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
CONFIG_FILE = "model_config.json"
MODEL_FILE = "waste_classification_model.h5"

WASTE_CLASS_DISPLAY = {"O": "Organic Waste", "R": "Recyclable Waste"}
WASTE_BIO_CLASSES = {"O"}

TRASHNET_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
TRASHNET_BIO = {"paper", "cardboard"}
TRASHNET_DISPLAY = {
    "cardboard": "Cardboard",
    "glass": "Glass",
    "metal": "Metal",
    "paper": "Paper",
    "plastic": "Plastic",
    "trash": "Mixed Trash",
}


def script_dir():
    return os.path.dirname(os.path.abspath(__file__))


def _has_class_folders(path, min_classes=2):
    if not path or not os.path.isdir(path):
        return False
    dirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return len(dirs) >= min_classes


def find_or_dataset():
    """Returns (train_dir, val_dir_or_none, dataset_type)."""
    search_roots = [
        script_dir(),
        os.path.join(script_dir(), ".."),
        os.path.join(script_dir(), "..", ".."),
        os.path.join(script_dir(), "..", "..", ".."),
    ]
    layouts = [
        ("Waste Classification Dataset", "DATASET"),
        ("Dataset", "DATASET"),
        ("Dataset", None),
        (None, "DATASET"),
    ]
    for base in search_roots:
        base = os.path.abspath(base)
        for folder_name, inner in layouts:
            train_parts = [p for p in (folder_name, inner, "TRAIN") if p]
            test_parts = [p for p in (folder_name, inner, "TEST") if p]
            train_wr = os.path.join(base, *train_parts)
            test_wr = os.path.join(base, *test_parts)
            if _has_class_folders(train_wr, min_classes=2):
                if _has_class_folders(test_wr, min_classes=2):
                    return train_wr, test_wr, "waste_classification"
                return train_wr, None, "waste_classification"

    root = script_dir()
    for name in ("dataset-resized", os.path.join("data", "dataset-resized")):
        path = os.path.join(root, name)
        if _has_class_folders(path, min_classes=4):
            return path, None, "trashnet"
    return None, None, None


def build_sampled_train_dir(source_train, max_per_class, seed=42):
    rng = random.Random(seed)
    tmp = tempfile.mkdtemp(prefix="smartwaste_train_")
    for cls in sorted(os.listdir(source_train)):
        cls_path = os.path.join(source_train, cls)
        if not os.path.isdir(cls_path):
            continue
        files = [
            f
            for f in os.listdir(cls_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]
        rng.shuffle(files)
        out_cls = os.path.join(tmp, cls)
        os.makedirs(out_cls, exist_ok=True)
        for f in files[:max_per_class]:
            shutil.copy2(os.path.join(cls_path, f), os.path.join(out_cls, f))
    return tmp


def resolve_dirs(cli_train=None):
    if cli_train:
        train = os.path.abspath(os.path.expanduser(cli_train))
        if not _has_class_folders(train, 2):
            print(f"ERROR: No class subfolders in {train}")
            sys.exit(1)
        return train, None, "custom"

    train, test, dtype = find_or_dataset()
    if train:
        print(f"Dataset type: {dtype}")
        print(f"Training images: {train}")
        if test:
            print(f"Validation images: {test}")
        return train, test, dtype

    print(
        "ERROR: Dataset not found.\n\n"
        "Place dataset under project root, e.g.:\n"
        "  Automatic Waste Detection Using Deep Learning/Dataset/DATASET/TRAIN/O/\n"
        "  Automatic Waste Detection Using Deep Learning/Dataset/DATASET/TRAIN/R/\n\n"
        "Then run:  python train_model.py --quick\n"
    )
    sys.exit(1)


def build_model(num_classes: int, fine_tune_at: int = 100) -> keras.Model:
    base = keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.35)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model._base_model = base
    model._fine_tune_at = fine_tune_at
    return model


def save_config(class_names, dataset_type, path=CONFIG_FILE):
    names = list(class_names)
    if dataset_type == "waste_classification":
        bio = [c for c in names if c in WASTE_BIO_CLASSES]
        display = {c: WASTE_CLASS_DISPLAY.get(c, c) for c in names}
    else:
        bio = [c for c in names if c in TRASHNET_BIO]
        display = {c: TRASHNET_DISPLAY.get(c, c.title()) for c in names}

    cfg = {
        "dataset_type": dataset_type,
        "class_names": names,
        "biodegradable_classes": bio,
        "display_names": display,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"Saved {path}: classes={names}, biodegradable={bio}")


def main():
    parser = argparse.ArgumentParser(description="Train SmartWaste waste classifier")
    parser.add_argument("--data_dir", default=None, help="Training folder (class subfolders inside)")
    parser.add_argument("--test_dir", default=None, help="Optional test/validation folder")
    parser.add_argument("--output", default=MODEL_FILE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--quick", action="store_true", help="Faster training on a subset")
    parser.add_argument("--max_per_class", type=int, default=None)
    args = parser.parse_args()

    if args.quick:
        args.epochs = min(args.epochs, 10)
        if args.max_per_class is None:
            args.max_per_class = 500

    if args.data_dir:
        train_dir = os.path.abspath(os.path.expanduser(args.data_dir))
        if not _has_class_folders(train_dir, 2):
            print(f"ERROR: No class subfolders in {train_dir}")
            sys.exit(1)
        test_dir = os.path.abspath(args.test_dir) if args.test_dir else None
        folder_names = {d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))}
        dtype = "waste_classification" if {"O", "R"}.issubset(folder_names) else "custom"
    else:
        train_dir, test_dir, dtype = resolve_dirs()

    sampled_tmp = None
    if args.max_per_class:
        print(f"Sampling up to {args.max_per_class} images per class...")
        sampled_tmp = build_sampled_train_dir(train_dir, args.max_per_class)
        train_dir = sampled_tmp

    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=IMG_SIZE,
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=True,
        seed=42,
    )
    class_names = list(train_ds.class_names)
    print("Classes (index order):", class_names)

    if test_dir:
        val_ds = keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=IMG_SIZE,
            batch_size=args.batch_size,
            label_mode="int",
            shuffle=False,
        )
    else:
        train_ds, val_ds = keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="both",
            seed=42,
            image_size=IMG_SIZE,
            batch_size=args.batch_size,
            label_mode="int",
        )
        class_names = list(train_ds.class_names)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    model = build_model(len(class_names))
    callbacks = [
        keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_accuracy"),
        keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
        keras.callbacks.ModelCheckpoint(args.output, save_best_only=True, monitor="val_accuracy"),
    ]

    phase1 = 4 if args.quick else min(12, args.epochs)
    print(f"\n--- Phase 1: train classifier head ({phase1} epochs) ---")
    model.fit(train_ds, validation_data=val_ds, epochs=phase1, callbacks=callbacks)

    print("\n--- Phase 2: fine-tune MobileNetV2 ---")
    base = model._base_model
    fine_tune_at = model._fine_tune_at
    base.trainable = True
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    remaining = 4 if args.quick else max(args.epochs - phase1, 8)
    model.fit(train_ds, validation_data=val_ds, epochs=remaining, callbacks=callbacks)

    model.save(args.output)
    save_config(class_names, dtype)
    if sampled_tmp:
        shutil.rmtree(sampled_tmp, ignore_errors=True)
    print(f"\nDone. Model: {args.output}")
    print("Restart Streamlit:  streamlit run app.py")


if __name__ == "__main__":
    main()
