"""Shared helpers: MongoDB connection + pretty printing."""
import json
import os

from bson import json_util
from pymongo import MongoClient
from rich.console import Console
from rich.syntax import Syntax

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017")
DB_NAME = os.getenv("MONGO_DB", "mlops_demo")

console = Console()


def get_db():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)
    client.admin.command("ping")  # fail fast if MongoDB is not running
    return client[DB_NAME]


def show_doc(doc, title=None):
    """Print a MongoDB document as highlighted JSON."""
    text = json.dumps(json.loads(json_util.dumps(doc)), indent=2, ensure_ascii=False)
    if title:
        console.rule(f"[bold cyan]{title}")
    console.print(Syntax(text, "json", theme="monokai", word_wrap=True))


def step(title):
    console.rule(f"[bold green]{title}")
