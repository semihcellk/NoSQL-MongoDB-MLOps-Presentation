"""STEP 4 - Serve the production model and log every prediction for monitoring."""
import io
import random
from datetime import datetime, timezone

import gridfs
import joblib
from rich.table import Table

from common import console, get_db, show_doc, step

db = get_db()
fs = gridfs.GridFS(db, collection="models")
step("STEP 4 - Load production model from GridFS, predict, log predictions")

# 1) Model registry lookup -> download binary from GridFS
run = db.experiments.find_one({"stage": "production"})
model = joblib.load(io.BytesIO(fs.get(run["model_file_id"]).read()))
console.print(f"Loaded production model: [bold]{run['model_type']}[/] {run['params']}")

# 2) Simulate incoming requests (test samples + some noise)
random.seed(7)
incoming = random.sample(list(db.samples.find({"split": "test"})), 30)
db.predictions.drop()
# TTL index: MongoDB automatically deletes prediction logs after 30 days
db.predictions.create_index("created_at", expireAfterSeconds=30 * 24 * 3600)

logs = []
for s in incoming:
    x = [s["features"][f] * random.uniform(0.9, 1.1) for f in run["features"]]
    proba = model.predict_proba([x])[0]
    pred = int(proba.argmax())
    logs.append({
        "model_run_id": run["_id"],
        "input": dict(zip(run["features"], [round(v, 3) for v in x])),
        "prediction": pred,
        "confidence": round(float(proba.max()), 4),
        "ground_truth": s["label"],          # arrives later in real life
        "created_at": datetime.now(timezone.utc),
    })
db.predictions.insert_many(logs)
console.print(f"Logged [bold]{len(logs)}[/] predictions to 'predictions' collection")
show_doc(db.predictions.find_one({}, {"_id": 0, "input": 0}), "One prediction log")

# 3) Monitoring query: live accuracy + confidence per predicted class
pipeline = [
    {"$group": {
        "_id": "$prediction",
        "count": {"$sum": 1},
        "avg_confidence": {"$avg": "$confidence"},
        "correct": {"$sum": {"$cond": [{"$eq": ["$prediction", "$ground_truth"]}, 1, 0]}},
    }},
    {"$sort": {"_id": 1}},
]
table = Table("predicted_class", "count", "avg_confidence", "accuracy")
for row in db.predictions.aggregate(pipeline):
    table.add_row(str(row["_id"]), str(row["count"]),
                  f"{row['avg_confidence']:.3f}", f"{row['correct'] / row['count']:.2%}")
console.rule("[bold cyan]Monitoring dashboard (aggregation pipeline)")
console.print(table)

low_conf = db.predictions.count_documents({"confidence": {"$lt": 0.6}})
console.print(f"Low-confidence predictions (<0.6): [bold red]{low_conf}[/] -> candidates for review / retraining")
