"""STEP 1 - Data storage: load the Wine dataset into MongoDB as JSON-like documents."""
import random
from datetime import datetime, timezone

from pymongo import ASCENDING
from sklearn.datasets import load_wine

from common import console, get_db, show_doc, step

db = get_db()
step("STEP 1 - Ingest raw data into the 'samples' collection")

wine = load_wine()
random.seed(42)

docs = []
for i, (row, label) in enumerate(zip(wine.data, wine.target)):
    docs.append({
        "sample_id": i,
        "features": {name: round(float(v), 3) for name, v in zip(wine.feature_names, row)},
        "label": int(label),
        "label_name": str(wine.target_names[label]),
        "split": "test" if random.random() < 0.25 else "train",
        "source": {"dataset": "sklearn.load_wine", "version": "v1"},
        "ingested_at": datetime.now(timezone.utc),
    })

db.samples.drop()
db.samples.insert_many(docs)
db.samples.create_index([("split", ASCENDING)])

console.print(f"Inserted [bold]{db.samples.count_documents({})}[/] documents "
              f"(train={db.samples.count_documents({'split': 'train'})}, "
              f"test={db.samples.count_documents({'split': 'test'})})")
show_doc(db.samples.find_one({}, {"_id": 0}), "One sample document (nested 'features' object)")
