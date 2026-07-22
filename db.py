"""
db.py — SQLite storage for enrolled employees and attendance records.

Embeddings are stored as raw float32 bytes (BLOB) and reloaded as numpy
arrays. This is fine at POC scale (tens of employees); if this ever grows
to hundreds+, swap this for a proper vector store (e.g. FAISS or a
Postgres + pgvector table).
"""
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime, date

DB_PATH = "attendance.db"


def get_conn():
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT,
            shift_start TEXT DEFAULT '09:00',
            embedding BLOB NOT NULL,
            enrolled_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            check_in TEXT,
            check_out TEXT,
            status TEXT,
            FOREIGN KEY(employee_id) REFERENCES employees(id)
        )
    """)
    conn.commit()
    conn.close()


def add_employee(name, department, embedding, shift_start="09:00"):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "INSERT INTO employees (name, department, shift_start, embedding, enrolled_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (name, department, shift_start,
         np.asarray(embedding, dtype=np.float32).tobytes(),
         datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_all_employees():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT id, name, department, shift_start, embedding FROM employees")
    rows = c.fetchall()
    conn.close()
    employees = []
    for r in rows:
        emb = np.frombuffer(r[4], dtype=np.float32)
        employees.append({
            "id": r[0], "name": r[1], "department": r[2],
            "shift_start": r[3], "embedding": emb
        })
    return employees


def _today_row(employee_id, today):
    conn = get_conn()
    c = conn.cursor()
    c.execute(
        "SELECT id, check_in, check_out FROM attendance WHERE employee_id=? AND date=?",
        (employee_id, today)
    )
    row = c.fetchone()
    conn.close()
    return row


def log_check_in(employee_id, shift_start="09:00", grace_minutes=10):
    """Logs a check-in for today if one doesn't already exist. Returns a status string."""
    today = date.today().isoformat()
    now = datetime.now()
    existing = _today_row(employee_id, today)
    if existing and existing[1]:
        return "already_checked_in"

    sh, sm = map(int, shift_start.split(":"))
    shift_dt = now.replace(hour=sh, minute=sm, second=0, microsecond=0)
    late_cutoff = shift_dt.timestamp() + grace_minutes * 60
    status = "on_time" if now.timestamp() <= late_cutoff else "late"

    conn = get_conn()
    c = conn.cursor()
    if existing:
        c.execute("UPDATE attendance SET check_in=?, status=? WHERE id=?",
                   (now.strftime("%H:%M:%S"), status, existing[0]))
    else:
        c.execute(
            "INSERT INTO attendance (employee_id, date, check_in, status) VALUES (?, ?, ?, ?)",
            (employee_id, today, now.strftime("%H:%M:%S"), status)
        )
    conn.commit()
    conn.close()
    return status


def log_check_out(employee_id):
    today = date.today().isoformat()
    now = datetime.now()
    existing = _today_row(employee_id, today)
    if not existing or not existing[1]:
        return "no_check_in_found"
    conn = get_conn()
    c = conn.cursor()
    c.execute("UPDATE attendance SET check_out=? WHERE id=?",
              (now.strftime("%H:%M:%S"), existing[0]))
    conn.commit()
    conn.close()
    return "checked_out"


def get_attendance_df():
    conn = get_conn()
    df = pd.read_sql_query("""
        SELECT a.date, e.name, e.department, a.check_in, a.check_out, a.status
        FROM attendance a JOIN employees e ON a.employee_id = e.id
        ORDER BY a.date DESC, a.check_in DESC
    """, conn)
    conn.close()
    return df


def get_today_summary():
    """Returns (total_employees, attended_today, absent_today)."""
    today = date.today().isoformat()
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM employees")
    total = c.fetchone()[0]
    c.execute(
        "SELECT COUNT(DISTINCT employee_id) FROM attendance WHERE date=? AND check_in IS NOT NULL",
        (today,)
    )
    attended = c.fetchone()[0]
    conn.close()
    return total, attended, max(total - attended, 0)