import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import json
import sqlite3
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras import layers, models

DB_PATH = os.getenv('LANDMARKS_DB_PATH')

def train_model(args):
    print("Training MLP model...")
    print ("GPU Available: ", tf.config.list_physical_devices('GPU'))

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    DATASET_VERSION = args.dataset_version
    INPUT_DIM = 42
    CLASS_NAMES = []
    TRAIN_SAMPLES = 0
    VAL_SAMPLES = 0
    TEST_SAMPLES = 0

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        rows = cur.execute("""
            SELECT gesture, landmarks FROM gestures_processed
            WHERE dataset_version = ?
        """, (DATASET_VERSION,)).fetchall() # pass a tuple of 1 element because of how sqlite binding works

        conn.close()

        X = []
        y = []
        gesture_to_idx = {}

        for gesture, landmarks_json in rows:
            landmarks = np.array(json.loads(landmarks_json)).flatten() # (21, 2) -> (42,)
            X.append(landmarks)

            # map gesture to integer label
            if gesture not in gesture_to_idx:
                gesture_to_idx[gesture] = len(gesture_to_idx)

            y.append(gesture_to_idx[gesture])

        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        print("Fetched the data from the database")

        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=10)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=10)

        TRAIN_SAMPLES = len(X_train)
        VAL_SAMPLES = len(X_val)
        TEST_SAMPLES = len(X_test)
        print(f"Train samples: {TRAIN_SAMPLES}")
        print(f"Validation samples: {VAL_SAMPLES}")
        print(f"Test samples: {TEST_SAMPLES}")

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1024).batch(BATCH_SIZE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(BATCH_SIZE)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(BATCH_SIZE)

        num_classes = len(gesture_to_idx)
        CLASS_NAMES = list(gesture_to_idx.keys())

        print(f"Number of classes: {num_classes}")
        print(f"Class names: {CLASS_NAMES}")

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    model = models.Sequential([
        layers.Input(shape=(INPUT_DIM,)),

        # Hidden layers
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(64, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(num_classes, activation='softmax'),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Stop trainign when val_loss does not improve for 10 consecutive epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # model training
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate model
    val_loss, val_accuracy = model.evaluate(val_dataset, verbose=2)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=2)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    y_pred_probs = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)

    y_true = np.concatenate([y for x, y in test_dataset], axis=0)

    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')

    conf_matrix = confusion_matrix(y_true, y_pred)

    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Save model with versioning
    version = args.version_name

    output_path = Path(args.model_output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model_path = output_path / f"gesture_model_{version}.keras"

    model.save(model_path)
    print(f"Model saved: {model_path}")

    if not args.no_set_active:
        active_path = output_path / "active_model.json"

        active_data = {
            "model_file": f"gesture_model_{version}.keras",
            "class_names": CLASS_NAMES
        }

        try:
            with open(active_path, 'w') as f:
                json.dump(active_data, f, indent=2)
            print(f"Set as active model: gesture_model_{version}.keras")
        except Exception as e:
            print(f"Failed to write active model file: {e}")


    # Build metrics dict for API/callback (stored in DB, not file)
    # Training metrics from history
    train_acc = float(history.history.get('accuracy', [0.0])[-1])
    train_loss_val = float(history.history.get('loss', [0.0])[-1])

    metrics = {
        'model_file': f"gesture_model_{version}.keras",
        'version': version,
        'config': {
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
        },
        'dataset': {
            'class_labels': CLASS_NAMES,
            'train_count': TRAIN_SAMPLES,
            'validation_count': VAL_SAMPLES,
            'test_count': TEST_SAMPLES,
        },
        'train': {
            'accuracy': float(train_acc),
            'loss': float(train_loss_val),
        },
        'validation': {
            'accuracy': float(val_accuracy),
            'loss': float(val_loss),
        },
        'test': {
            'accuracy': float(test_accuracy),
            'loss': float(test_loss),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': conf_matrix.tolist()
        }
    }

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a MLP model for gesture recognition.")
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--version', type=str, help='Model version identifier')
    parser.add_argument('--no-set-active', action='store_true', help='Do not set the trained model as active')
    parser.add_argument('--model-output-path', default='/models', help='Directory to save the trained model and metadata')

    args = parser.parse_args()
    train_model(args)
