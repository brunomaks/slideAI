import tensorflow as tf
from tensorflow.keras import layers, models
import argparse
import joblib
from pathlib import Path
from datetime import datetime


def train_model(args):
    print("🚀 Training CNN model...")
    print ("GPU Available: ", tf.config.list_physical_devices('GPU'))

    IMG_SIZE = (args.img_size, args.img_size)
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    # Load dataset
    train_data = tf.keras.utils.image_dataset_from_directory(
        'images/train',
        shuffle=True,
        color_mode='grayscale',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    val_data = tf.keras.utils.image_dataset_from_directory(
        'images/train',
        shuffle=True,
        color_mode='grayscale',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    num_classes = len(train_data.class_names)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {train_data.class_names}")

    # Performance optimization
    train_data = train_data.prefetch(buffer_size=tf.data.AUTOTUNE)
    val_data = val_data.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Build model
    model = models.Sequential([
        layers.Rescaling(1./255, input_shape=IMG_SIZE + (1,)),

        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # Train model
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=EPOCHS,
        verbose=1
    )

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(val_data, verbose=2)
    print(f"✅ Test Accuracy: {test_accuracy:.4f}")

    # Save model with versioning
    version = args.version or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Path(args.model_output_path) / f"gesture_model_{version}.h5"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    model.save(model_path)

    metadata_path = Path(args.model_output_path) / f"model_metadata_{version}.pkl"

    joblib.dump({
        'model': model,
        'version': version,
        'accuracy': test_accuracy,
        'loss': test_loss,
        'num_classes': num_classes,
        'class_names': train_data.class_names,
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'training_history': history.history
    }, metadata_path)

    print(f"✅ Model saved: {model_path}")
    print(f"✅ Metadata saved: {metadata_path}")

    if args.set_active:
        active_path = Path(args.model_output_path) / "active_model.txt"
        active_path.write_text(f"gesture_model_{version}.h5")
        print(f"✅ Set as active model: gesture_model_{version}.h5")

    return test_accuracy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CNN model for gesture recognition.")
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--img-size', type=int, default=64, help='Image size (width and height)')
    parser.add_argument('--version', type=str, help='Model version identifier')
    parser.add_argument('--set-active', action='store_true', help='Set the trained model as active')
    parser.add_argument('--model-output-path', default='/models', help='Directory to save the trained model and metadata')

    args = parser.parse_args()
    train_model(args)
