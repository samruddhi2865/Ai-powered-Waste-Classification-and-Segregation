"""
database.py
-----------
Handles all SQLite operations for the Waste Classification System.
Run this file once to initialise the database: python database.py
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "waste_data.db")

# ─────────────────────────────────────────────
# Biodegradable / Non-biodegradable mapping
# ─────────────────────────────────────────────
BIODEGRADABLE_TYPES = {"organic_waste", "paper", "cardboard"}

NON_BIODEGRADABLE_TYPES = {
    "plastic", "metal", "glass", "can",
    "cable", "e_waste", "medical_waste",
}

WASTE_NATURE = {
    **{k: "Biodegradable"     for k in BIODEGRADABLE_TYPES},
    **{k: "Non-Biodegradable" for k in NON_BIODEGRADABLE_TYPES},
}


def get_waste_nature(class_name: str) -> str:
    return WASTE_NATURE.get(class_name, "Unknown")


# ─────────────────────────────────────────────
# Initialise DB
# ─────────────────────────────────────────────
def init_db():
    """Create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT    NOT NULL,
            source       TEXT    NOT NULL,          -- 'webcam' or 'upload'
            class_name   TEXT    NOT NULL,
            confidence   REAL    NOT NULL,
            bin_label    TEXT    NOT NULL,
            waste_nature TEXT    NOT NULL            -- 'Biodegradable' / 'Non-Biodegradable'
        )
    """)
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# Write
# ─────────────────────────────────────────────
def save_detections(detections: list[dict], source: str = "upload"):
    """
    detections : list of dicts from classify_image / run_inference_on_frame
    source     : 'webcam' or 'upload'
    """
    if not detections:
        return
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        (
            ts,
            source,
            d["class"],
            round(d["confidence"], 4),
            d["category_info"]["bin"],
            get_waste_nature(d["class"]),
        )
        for d in detections
    ]
    cur.executemany(
        "INSERT INTO detections (timestamp, source, class_name, confidence, bin_label, waste_nature) VALUES (?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# Read helpers
# ─────────────────────────────────────────────
def fetch_all_detections() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute("SELECT * FROM detections ORDER BY id DESC")
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def fetch_summary() -> dict:
    """Returns aggregate stats for the dashboard header cards."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM detections")
    total = cur.fetchone()[0]

    cur.execute("SELECT class_name, COUNT(*) as cnt FROM detections GROUP BY class_name ORDER BY cnt DESC LIMIT 1")
    row = cur.fetchone()
    most_common = row[0] if row else "—"

    cur.execute("SELECT AVG(confidence) FROM detections")
    avg_conf = cur.fetchone()[0] or 0.0

    cur.execute("SELECT COUNT(*) FROM detections WHERE waste_nature='Biodegradable'")
    bio_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM detections WHERE waste_nature='Non-Biodegradable'")
    non_bio_count = cur.fetchone()[0]

    conn.close()
    return {
        "total":         total,
        "most_common":   most_common,
        "avg_confidence": round(avg_conf, 3),
        "biodegradable":     bio_count,
        "non_biodegradable": non_bio_count,
    }


def fetch_distribution() -> list[dict]:
    """Count per class_name for the donut/pie chart."""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("""
        SELECT class_name, waste_nature, COUNT(*) as count
        FROM detections
        GROUP BY class_name
        ORDER BY count DESC
    """)
    rows = [{"class_name": r[0], "waste_nature": r[1], "count": r[2]} for r in cur.fetchall()]
    conn.close()
    return rows


def fetch_nature_split() -> dict:
    """{'Biodegradable': N, 'Non-Biodegradable': M}"""
    conn = sqlite3.connect(DB_PATH)
    cur  = conn.cursor()
    cur.execute("SELECT waste_nature, COUNT(*) FROM detections GROUP BY waste_nature")
    result = {r[0]: r[1] for r in cur.fetchall()}
    conn.close()
    return result


def fetch_recent(limit: int = 20) -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur  = conn.cursor()
    cur.execute(
        "SELECT * FROM detections ORDER BY id DESC LIMIT ?", (limit,)
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def clear_all_detections():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM detections")
    conn.commit()
    conn.close()


# ─────────────────────────────────────────────
# Entry point – initialise on import
# ─────────────────────────────────────────────
init_db()


if __name__ == "__main__":
    print(f"✅ Database initialised at: {DB_PATH}")
