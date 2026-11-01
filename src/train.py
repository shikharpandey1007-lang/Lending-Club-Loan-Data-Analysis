"""Train baseline ML models and a Keras neural network.

Run from the repository root:
    python -m src.train --data Data/loan_data.csv
"""
from pathlib import Path
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

TARGET = "not.fully.paid"
RANDOM_STATE = 42


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain target column {TARGET!r}.")
    df = df.drop_duplicates().copy()
    df = df.dropna(subset=[TARGET])
    df[TARGET] = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1.")
    return df


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    categorical = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    numerical = [c for c in X.columns if c not in categorical]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipe, numerical),
        ("categorical", categorical_pipe, categorical),
    ])


def choose_threshold(y_true, probabilities) -> float:
    """Choose the validation threshold with the best F1; never tune on test data."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if len(thresholds) == 0:
        return 0.5
    f1 = (2 * precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.nanargmax(f1))])


def score_model(name, y_true, probabilities, threshold=0.5):
    predictions = (probabilities >= threshold).astype(int)
    return {
        "model": name,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision_default_class": float(precision_score(y_true, predictions, zero_division=0)),
        "recall_default_class": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_default_class": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
        "classification_report": classification_report(
            y_true, predictions, output_dict=True, zero_division=0
        ),
    }


def build_network(input_dim: int) -> tf.keras.Model:
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.30),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.20),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.AUC(name="pr_auc", curve="PR"),
        ],
    )
    return model


def save_curves(y_true, probabilities, output_dir: Path):
    fpr, tpr, _ = roc_curve(y_true, probabilities)
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc_score(y_true, probabilities):.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Test ROC curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "roc_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"PR-AUC = {average_precision_score(y_true, probabilities):.3f}")
    ax.set(xlabel="Recall", ylabel="Precision", title="Test precision-recall curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "precision_recall_curve.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Data/loan_data.csv")
    parser.add_argument("--output", default="artifacts")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20,
        random_state=RANDOM_STATE, stratify=y_train_full
    )

    preprocessor = make_preprocessor(X_train)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)
    # Keras Dense layers require dense arrays. This dataset is small enough for conversion.
    X_train_t = np.asarray(X_train_t.toarray() if hasattr(X_train_t, "toarray") else X_train_t, dtype=np.float32)
    X_val_t = np.asarray(X_val_t.toarray() if hasattr(X_val_t, "toarray") else X_val_t, dtype=np.float32)
    X_test_t = np.asarray(X_test_t.toarray() if hasattr(X_test_t, "toarray") else X_test_t, dtype=np.float32)

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights = {int(k): float(v) for k, v in zip(classes, weights)}

    results = []
    # Classical baselines use the same training-only preprocessing.
    logistic = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)
    logistic.fit(X_train_t, y_train)
    val_prob = logistic.predict_proba(X_val_t)[:, 1]
    logistic_threshold = choose_threshold(y_val, val_prob)
    results.append(score_model("Logistic Regression", y_test, logistic.predict_proba(X_test_t)[:, 1], logistic_threshold))
    joblib.dump(logistic, output_dir / "logistic_regression.joblib")

    forest = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    forest.fit(X_train_t, y_train)
    forest_threshold = choose_threshold(y_val, forest.predict_proba(X_val_t)[:, 1])
    results.append(score_model("Random Forest", y_test, forest.predict_proba(X_test_t)[:, 1], forest_threshold))
    joblib.dump(forest, output_dir / "random_forest.joblib")

    network = build_network(X_train_t.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_pr_auc", mode="max", patience=6, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]
    history = network.fit(
        X_train_t, y_train.to_numpy(), validation_data=(X_val_t, y_val.to_numpy()),
        epochs=args.epochs, batch_size=args.batch_size, class_weight=class_weights,
        callbacks=callbacks, verbose=1,
    )
    val_prob = network.predict(X_val_t, verbose=0).ravel()
    nn_threshold = choose_threshold(y_val, val_prob)
    test_prob = network.predict(X_test_t, verbose=0).ravel()
    results.append(score_model("Keras Neural Network", y_test, test_prob, nn_threshold))
    network.save(output_dir / "loan_default_model.keras")
    joblib.dump(preprocessor, output_dir / "preprocessing_pipeline.joblib")

    history_data = {k: [float(v) for v in values] for k, values in history.history.items()}
    (output_dir / "training_history.json").write_text(json.dumps(history_data, indent=2), encoding="utf-8")
    (output_dir / "test_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_curves(y_test, test_prob, output_dir)

    comparison = pd.DataFrame([{k: v for k, v in row.items() if k not in {"confusion_matrix", "classification_report"}} for row in results])
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)
    print("\nTest-set comparison (default class = 1):")
    print(comparison.to_string(index=False))
    print(f"\nSaved models, preprocessing pipeline, metrics and curves to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
