# src/integrations/sku_db.py
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sku (
  sku_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  brand TEXT,
  category TEXT,
  barcode TEXT,
  status TEXT DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS sku_images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  sku_id TEXT NOT NULL,
  path TEXT NOT NULL,
  source TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (sku_id) REFERENCES sku(sku_id)
);
"""

class SkuDB:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        self._init()

    def _init(self) -> None:
        con = sqlite3.connect(self.db_path)
        try:
            con.executescript(SCHEMA_SQL)
            con.commit()
        finally:
            con.close()

    def get_sku(self, sku_id: str) -> Optional[Dict]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            row = con.execute("SELECT * FROM sku WHERE sku_id=?", (sku_id,)).fetchone()
            return dict(row) if row else None
        finally:
            con.close()

    def list_active(self) -> List[Dict]:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        try:
            rows = con.execute("SELECT * FROM sku WHERE status='active'").fetchall()
            return [dict(r) for r in rows]
        finally:
            con.close()
