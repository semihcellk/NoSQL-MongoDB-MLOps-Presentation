"""STEP 3 - Query experiments and promote the best model to production."""
from rich.table import Table

from common import console, get_db, step

db = get_db()
step("STEP 3 - Query runs, aggregate, promote best model")

# 1) Rich query on a nested field that only SOME documents have
console.print('[bold]find({"params.n_estimators": {"$gte": 100}})[/]')
for run in db.experiments.find({"params.n_estimators": {"$gte": 100}}, {"_id": 0, "model_type": 1, "params": 1}):
    console.print(f"  -> {run['model_type']}  {run['params']}")

# 2) Aggregation pipeline: leaderboard per model family
console.print("\n[bold]aggregate: $group by model_type -> avg / best f1[/]")
pipeline = [
    {"$group": {
        "_id": "$model_type",
        "runs": {"$sum": 1},
        "avg_f1": {"$avg": "$metrics.f1_macro"},
        "best_f1": {"$max": "$metrics.f1_macro"},
    }},
    {"$sort": {"best_f1": -1}},
]
table = Table("model_type", "runs", "avg_f1", "best_f1")
for row in db.experiments.aggregate(pipeline):
    table.add_row(row["_id"], str(row["runs"]), f"{row['avg_f1']:.4f}", f"{row['best_f1']:.4f}")
console.print(table)

# 3) Model registry: promote best run (highest f1, fastest on tie) to production
best = db.experiments.find_one(sort=[("metrics.f1_macro", -1), ("train_seconds", 1)])
db.experiments.update_many({"stage": "production"}, {"$set": {"stage": "archived"}})
db.experiments.update_one({"_id": best["_id"]}, {"$set": {"stage": "production"}})

console.print(f"\n[bold green]Promoted to production:[/] {best['model_type']} "
              f"{best['params']}  f1={best['metrics']['f1_macro']}")
