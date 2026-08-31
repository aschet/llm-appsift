"""SQLite-backed store for the task list."""
import os
import sqlite3

SEED = [
    ("Read the specification", "Everything the grader checks is in SPEC.md", "#3366cc"),
    ("Wire up the storage", "SQLite, keyed on position", "#cc7722"),
    ("Draw the page", "Colour swatch, checkbox, drag handle", "#22aa88"),
]


def db_path():
    return os.environ.get("TODO_DB") or "tasks.db"


def connect():
    conn = sqlite3.connect(db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("""CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT NOT NULL, description TEXT NOT NULL DEFAULT '',
        color TEXT NOT NULL DEFAULT '#888888', done INTEGER NOT NULL DEFAULT 0,
        position INTEGER NOT NULL DEFAULT 0)""")
    if not conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]:
        for i, (title, desc, color) in enumerate(SEED):
            conn.execute("INSERT INTO tasks (title, description, color, done, position)"
                         " VALUES (?,?,?,0,?)", (title, desc, color, i))
        conn.commit()
    return conn


def as_task(row):
    return dict(id=row["id"], title=row["title"], description=row["description"],
                color=row["color"], done=bool(row["done"]), position=row["position"])


def listing(done=None, query=None):
    conn = connect()
    sql, args = "SELECT * FROM tasks", []
    where = []
    if done is not None:
        where.append("done = ?")
        args.append(1 if done else 0)
    if query:
        where.append("(LOWER(title) LIKE ? OR LOWER(description) LIKE ?)")
        args += [f"%{query.lower()}%"] * 2
    if where:
        sql += " WHERE " + " AND ".join(where)
    rows = conn.execute(sql + " ORDER BY position ASC, id ASC", args).fetchall()
    conn.close()
    return [as_task(r) for r in rows]


def get(task_id):
    conn = connect()
    row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
    conn.close()
    return as_task(row) if row else None


def create(title, description="", color="#888888"):
    conn = connect()
    last = conn.execute("SELECT COALESCE(MAX(position), -1) FROM tasks").fetchone()[0]
    cur = conn.execute("INSERT INTO tasks (title, description, color, done, position)"
                       " VALUES (?,?,?,0,?)", (title, description, color, last + 1))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()
    return get(new_id)


def update(task_id, fields):
    allowed = {k: v for k, v in fields.items()
               if k in ("title", "description", "color", "done")}
    if not get(task_id):
        return None
    if allowed:
        if "done" in allowed:
            allowed["done"] = 1 if allowed["done"] else 0
        conn = connect()
        conn.execute("UPDATE tasks SET " + ", ".join(f"{k} = ?" for k in allowed)
                     + " WHERE id = ?", list(allowed.values()) + [task_id])
        conn.commit()
        conn.close()
    return get(task_id)


def delete(task_id):
    if not get(task_id):
        return False
    conn = connect()
    conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
    conn.commit()
    conn.close()
    return True


def reorder(order):
    conn = connect()
    for position, task_id in enumerate(order):
        conn.execute("UPDATE tasks SET position = ? WHERE id = ?", (position, task_id))
    conn.commit()
    conn.close()
    return listing()
