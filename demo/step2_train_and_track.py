"""STEP 2 - Experiment tracking + model registry.

Train several model families. Each run is logged as ONE document; the
'params' sub-document has a different shape per model -> no schema migration
needed. Trained model binaries are stored in MongoDB via GridFS.
"""
import io
import time
from datetime import datetime, timezone

import gridfs
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from rich.table import Table

from common import console, get_db, show_doc, step


def load_split(db, split):
    cursor = db.samples.find({"split": split}, {"_id": 0, "features": 1, "label": 1})
    rows = list(cursor)
    feature_names = list(rows[0]["features"].keys())
    X = [[r["features"][f] for f in feature_names] for r in rows]
    y = [r["label"] for r in rows]
    return X, y, feature_names


EXPERIMENTS = [
    ("LogisticRegression", LogisticRegression, {"C": 0.1, "max_iter": 1000}),
    ("LogisticRegression", LogisticRegression, {"C": 10.0, "max_iter": 1000}),
    ("RandomForest", RandomForestClassifier, {"n_estimators": 10, "max_depth": 2, "random_state": 0}),
    ("RandomForest", RandomForestClassifier, {"n_estimators": 200, "max_depth": None, "random_state": 0}),
    ("KNN", KNeighborsClassifier, {"n_neighbors": 3, "weights": "distance"}),
    ("KNN", KNeighborsClassifier, {"n_neighbors": 15, "weights": "uniform"}),
]

db = get_db()
fs = gridfs.GridFS(db, collection="models")
step("STEP 2 - Train models, log runs to 'experiments', store models in GridFS")

X_train, y_train, feature_names = load_split(db, "train")
X_test, y_test, _ = load_split(db, "test")
console.print(f"Loaded train={len(y_train)} / test={len(y_test)} samples [dim]from MongoDB[/]")

db.experiments.drop()
db.models.files.drop()
db.models.chunks.drop()

table = Table("model_type", "params", "accuracy", "f1_macro")
for model_type, cls, params in EXPERIMENTS:
    start = time.perf_counter()
    model = make_pipeline(StandardScaler(), cls(**params)).fit(X_train, y_train)
    train_seconds = round(time.perf_counter() - start, 4)
    preds = model.predict(X_test)
    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "f1_macro": round(f1_score(y_test, preds, average="macro"), 4),
    }

    # serialize the trained model and store it in GridFS (files > 16 MB are fine)
    buffer = io.BytesIO()
    joblib.dump(model, buffer)
    model_file_id = fs.put(buffer.getvalue(), filename=f"{model_type}.joblib")

    db.experiments.insert_one({
        "model_type": model_type,
        "params": params,                      # different shape for every model type!
        "metrics": metrics,
        "train_seconds": train_seconds,
        "features": feature_names,
        "model_file_id": model_file_id,
        "stage": "candidate",
        "tags": ["wine", "baseline"] if model_type == "LogisticRegression" else ["wine"],
        "created_at": datetime.now(timezone.utc),
    })
    table.add_row(model_type, str(params), str(metrics["accuracy"]), str(metrics["f1_macro"]))

console.print(table)
show_doc(db.experiments.find_one({"model_type": "RandomForest"},
                                 {"_id": 0, "features": 0}),
         "Example run document")
show_doc(db.experiments.find_one({"model_type": "KNN"}, {"_id": 0, "model_type": 1, "params": 1}),
         "Same collection, different 'params' shape (no schema change!)")
