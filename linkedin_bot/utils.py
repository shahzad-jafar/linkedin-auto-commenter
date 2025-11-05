import sqlite3, os, datetime

DB_PATH = "data/logs.db"

def init_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS comments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        post_text TEXT,
        comment TEXT,
        timestamp TEXT
    )
    """)
    conn.commit()
    return conn

def log_comment(post_text, comment):
    conn = init_db()
    conn.execute("INSERT INTO comments (post_text, comment, timestamp) VALUES (?, ?, ?)",
                 (post_text, comment, datetime.datetime.now().isoformat()))
    conn.commit()
    conn.close()
