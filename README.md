# NoSQL for MLOps: MongoDB as an ML Metadata & Serving Store

**YZV 448E – Machine Learning System Development and Operations (MLOps)**
Tool presentation demo · Semih Çelik (150230104)

This repo shows how **MongoDB**, a document-oriented NoSQL database, can serve as the
storage backbone of a small ML system. Using the **Wine classification** dataset, the demo
covers four lifecycle stages:

| Step | ML lifecycle stage | What MongoDB does | Collection(s) |
|------|--------------------|-------------------|---------------|
| 1 | **Data storage** | Stores raw samples as JSON-like documents with nested `features` | `samples` |
| 2 | **Experiment tracking** | Logs one document per training run; `params` have a different shape per model type, so no schema migration is needed | `experiments` |
| 2 | **Model registry** | Stores trained model binaries with **GridFS** | `models.files`, `models.chunks` |
| 3 | **Model selection** | Queries nested fields, builds a leaderboard with the **aggregation pipeline**, promotes the best run to `production` | `experiments` |
| 4 | **Serving & monitoring** | Loads the production model from GridFS, logs every prediction (with a **TTL index**) and computes live accuracy and confidence | `predictions` |

```mermaid
flowchart LR
    A[sklearn Wine dataset] -->|step1 insert_many| S[(samples)]
    S -->|step2 find split=train/test| T[Train 6 models]
    T -->|run metadata| E[(experiments)]
    T -->|model.joblib| G[(GridFS: models)]
    E -->|step3 aggregate + update stage| E
    E -->|step4 stage=production| P[Serve model]
    G --> P
    P -->|log prediction| L[(predictions, TTL 30d)]
    L -->|aggregate| M[Monitoring table]
```

## Project structure

```
.
├── demo/
│   ├── common.py                   # MongoDB connection (MONGO_URI) + pretty printing
│   ├── step1_ingest.py             # dataset -> 'samples'
│   ├── step2_train_and_track.py    # train models, log runs, save models to GridFS
│   ├── step3_query_and_promote.py  # queries, aggregation leaderboard, promote best
│   ├── step4_serve_and_monitor.py  # load from GridFS, predict, log + monitor
│   └── run_all.py                  # runs all steps (Enter between steps)
├── requirements.txt
└── README.md
```

## Setup

### 1. Install and start MongoDB (Community Edition)

- **Windows:** `winget install MongoDB.Server` installs MongoDB and runs it as a Windows service on port `27017`.
- **macOS:** `brew tap mongodb/brew && brew install mongodb-community && brew services start mongodb-community`
- **Linux:** see the [official install guide](https://www.mongodb.com/docs/manual/administration/install-on-linux/).
- **Alternatives:** use a free [MongoDB Atlas](https://www.mongodb.com/atlas) cluster, or run
  `docker run -d -p 27017:27017 mongo`.

Optional: [MongoDB Compass](https://www.mongodb.com/products/tools/compass) (GUI) lets you browse the collections (`winget install MongoDB.Compass.Full`).

### 2. Install Python dependencies (Python 3.10+)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 3. (Optional) Point to another MongoDB

By default the demo connects to `mongodb://127.0.0.1:27017` and uses the database `mlops_demo`.

```bash
# PowerShell: $env:MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net"
export MONGO_URI="mongodb+srv://user:pass@cluster.mongodb.net"
export MONGO_DB="mlops_demo"
```

## Run

Run everything (press Enter between steps):

```bash
python demo/run_all.py            # add --no-pause to run without stopping
```

Or run the steps one by one:

```bash
python demo/step1_ingest.py
python demo/step2_train_and_track.py
python demo/step3_query_and_promote.py
python demo/step4_serve_and_monitor.py
```

Each step drops and recreates its own collections, so you can re-run the demo as many times as you like.
The full demo takes about 15 seconds.

## Things to try in Compass / mongosh

Connect to `mongodb://127.0.0.1:27017` and open the `mlops_demo` database:

```js
// runs with a hyperparameter that only RandomForest documents have
db.experiments.find({ "params.n_estimators": { $gte: 100 } })

// current production model
db.experiments.find({ stage: "production" }, { model_type: 1, params: 1, metrics: 1 })

// leaderboard per model family
db.experiments.aggregate([
  { $group: { _id: "$model_type", best_f1: { $max: "$metrics.f1_macro" } } },
  { $sort: { best_f1: -1 } }
])

// low-confidence predictions to review
db.predictions.find({ confidence: { $lt: 0.6 } })
```

## Key MongoDB concepts shown

- **Document model / flexible schema:** every run stores its own `params` object; no `ALTER TABLE` is needed.
- **Nested-field queries** with dot notation (`"metrics.f1_macro"`, `"params.n_estimators"`).
- **Indexes:** a secondary index on `samples.split` and a **TTL index** on `predictions.created_at`.
- **Aggregation pipeline** (`$group`, `$sort`, `$cond`) for leaderboards and monitoring.
- **GridFS** for storing binary artifacts (models) next to their metadata.
- **Atomic updates** (`update_one` with `$set`) to move a model from `candidate` to `production`.
