"""
AgroMate – Plant Disease Detection Model Trainer
=================================================
Uses Transfer Learning with MobileNetV2 on the PlantVillage dataset.

Dataset structure expected:
    data/PlantVillageDataset/train_val_test/
        train/   <class_folders>
        val/     <class_folders>
        test/    <class_folders>

Usage:
    python train_disease_model.py

Output:
    models/disease_model.keras
    models/disease_class_names.pkl
"""

import os
import sys
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data", "PlantVillageDataset", "train_val_test")
TRAIN_DIR   = os.path.join(DATA_DIR, "train")
VAL_DIR     = os.path.join(DATA_DIR, "val")
TEST_DIR    = os.path.join(DATA_DIR, "test")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
MODEL_PATH  = os.path.join(MODELS_DIR, "disease_model.keras")
NAMES_PATH  = os.path.join(MODELS_DIR, "disease_class_names.pkl")

# ── Config ─────────────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 20          # EarlyStopping will stop earlier if needed
FINE_TUNE   = True        # Unfreeze top layers of MobileNetV2 after initial training

# ── Validate paths ─────────────────────────────────────────────────────────────
for path, name in [(TRAIN_DIR, "train"), (VAL_DIR, "val"), (TEST_DIR, "test")]:
    if not os.path.exists(path):
        print(f"[ERROR] {name} folder not found at: {path}")
        print("Make sure your dataset is at: data/PlantVillageDataset/train_val_test/")
        sys.exit(1)

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Data Generators ────────────────────────────────────────────────────────────
print("Setting up data generators...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Save class names
class_names = list(train_gen.class_indices.keys())
NUM_CLASSES = len(class_names)
print(f"\n  Classes ({NUM_CLASSES}): {class_names}\n")

# ── Build Model (MobileNetV2 Transfer Learning) ────────────────────────────────
print("Building MobileNetV2 model...")

base_model = MobileNetV2(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # Freeze base initially

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.4)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Total params     : {model.count_params():,}")
print(f"  Trainable params : {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}\n")

# ── Phase 1: Train only top layers ─────────────────────────────────────────────
print("=" * 55)
print("  Phase 1: Training top layers (base frozen)")
print("=" * 55)

callbacks_phase1 = [
    EarlyStopping(monitor='val_accuracy', patience=5,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=3, min_lr=1e-6, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                    save_best_only=True, verbose=1)
]

history1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    callbacks=callbacks_phase1,
    verbose=1
)

# ── Phase 2: Fine-tune top layers of MobileNetV2 ──────────────────────────────
if FINE_TUNE:
    print("\n" + "=" * 55)
    print("  Phase 2: Fine-tuning top MobileNetV2 layers")
    print("=" * 55)

    # Unfreeze the top 30 layers of MobileNetV2
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-4),   # lower LR for fine-tuning
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks_phase2 = [
        EarlyStopping(monitor='val_accuracy', patience=5,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                        save_best_only=True, verbose=1)
    ]

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks_phase2,
        verbose=1
    )

# ── Evaluate on test set ───────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  Evaluating on Test Set")
print("=" * 55)

test_loss, test_acc = model.evaluate(test_gen, verbose=0)
print(f"\n  Test Accuracy : {test_acc * 100:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}\n")

# Per-class accuracy
print("Per-class Results:")
preds      = model.predict(test_gen, verbose=0)
pred_labels = np.argmax(preds, axis=1)
true_labels = test_gen.classes
from sklearn.metrics import classification_report
print(classification_report(true_labels, pred_labels, target_names=class_names))

# ── Save ───────────────────────────────────────────────────────────────────────
model.save(MODEL_PATH)
joblib.dump(class_names, NAMES_PATH)

print(f"✓ Model saved       : {MODEL_PATH}")
print(f"✓ Class names saved : {NAMES_PATH}")
print(f"\nClasses the model can detect:")
for i, name in enumerate(class_names):
    print(f"  {i+1:>2}. {name}")
print("\nAll done! Run 'python app.py' to start the server.")
