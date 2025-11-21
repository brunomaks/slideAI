import argparse
import joblib
from pathlib import Path
from datetime import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model(args):
    print("🚀 Training model...")

    # Dummy data
    X = np.random.rand(1000, 20)
    y = np.random.randint(0, 2, 1000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print(f"✅ Accuracy: {accuracy:.4f}")

    # Save model
    version = args.version or datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = Path(args.model_output_path) / f"model_{version}.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        'model': model,
        'version': version,
        'accuracy': accuracy
    }, model_path)

    print(f"✅ Model saved: {model_path}")

    if args.set_active:
        active_path = Path(args.model_output_path) / "active_model.txt"
        active_path.write_text(f"model_{version}.pkl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--version', type=str)
    parser.add_argument('--set-active', action='store_true')
    parser.add_argument('--model-output-path', default='/models')

    args = parser.parse_args()
    train_model(args)
