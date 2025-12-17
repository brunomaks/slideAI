import argparse
from pathlib import Path
from datetime import datetime


def train_model(args):
    import tensorflow as tf
    from tensorflow.keras import layers, models
    print("Training CNN model...")
    print ("GPU Available: ", tf.config.list_physical_devices('GPU'))

    IMG_SIZE = (args.img_size, args.img_size)
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    try:
        train_data = tf.keras.utils.image_dataset_from_directory(
            '/images/train',
            shuffle=True,
            color_mode="rgb",
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )

        val_data = tf.keras.utils.image_dataset_from_directory(
            '/images/train',
            shuffle=True,
            color_mode='rgb',
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )

        test_data = tf.keras.utils.image_dataset_from_directory(
            '/images/test',
            shuffle=False,
            color_mode='rgb',
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )

        # Inspect a batch of data
        for images, labels in train_data.take(1):
            print(f"Image batch shape: {images.shape}")
            print(f"Labels batch shape: {labels.shape}")
            print(f"Label values: {labels.numpy()}")

        num_classes = len(train_data.class_names)
        class_names = train_data.class_names

        print(f"Number of classes: {num_classes}")
        print(f"Class names: {class_names}")

        # optimize dataset performance by prefetching
        train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
        val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # model training
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        verbose=1
    )

    # Evaluate model
    val_loss, val_accuracy = model.evaluate(val_data, verbose=2)
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    test_loss, test_accuracy = model.evaluate(test_data, verbose=2)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Save model with versioning
    version = args.version or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Path(args.model_output_path) / f"gesture_model_{version}.keras"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_path)

    print(f"Model saved: {model_path}")

    if args.set_active:
        active_path = Path(args.model_output_path) / "active_model.txt"
        active_path.write_text(f"gesture_model_{version}.keras")
        print(f"Set as active model: gesture_model_{version}.keras")

    # Export metrics alongside the model for downstream registration
    try:
        # Derive dataset counts from filesystem (best-effort)
        train_total = len(list(Path('/images/train').glob('*/*')))
        val_count = int(train_total * 0.2)
        train_count = train_total - val_count
        test_count = len(list(Path('/images/test').glob('*/*')))

        # Training metrics from history
        train_acc = float(history.history.get('accuracy', [0.0])[-1])
        train_loss = float(history.history.get('loss', [0.0])[-1])

        metrics = {
            'model_file': f"gesture_model_{version}.keras",
            'version': version,
            'config': {
                'epochs': EPOCHS,
                'batch_size': BATCH_SIZE,
                'image_size': args.img_size,
            },
            'dataset': {
                'train_count': train_count,
                'validation_count': val_count,
                'test_count': test_count,
            },
            'train': {
                'accuracy': float(train_acc),
                'loss': float(train_loss),
            },
            'validation': {
                'accuracy': float(val_accuracy),
                'loss': float(val_loss),
            },
            'test': {
                'accuracy': float(test_accuracy),
                'loss': float(test_loss),
            }
        }

        import json
        metrics_path = Path(args.model_output_path) / f"gesture_model_{version}.metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f)
        print(f"Metrics saved: {metrics_path}")
    except Exception as e:
        print(f"Failed to write metrics file: {e}")

    return test_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN model for gesture recognition.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=128, help='Image size (width and height)')
    parser.add_argument('--version', type=str, help='Model version identifier')
    parser.add_argument('--set-active', action='store_true', help='Set the trained model as active')
    parser.add_argument('--model-output-path', default='/models', help='Directory to save the trained model and metadata')

    args = parser.parse_args()
    train_model(args)
