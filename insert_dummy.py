import sqlite3, os, datetime
p='data/logs.db'
if not os.path.exists(p):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    conn=sqlite3.connect(p)
    conn.execute('CREATE TABLE IF NOT EXISTS comments (id INTEGER PRIMARY KEY AUTOINCREMENT, post_text TEXT, comment TEXT, timestamp TEXT)')
    conn.commit()
else:
    conn=sqlite3.connect(p)

conn.execute("INSERT INTO comments (post_text, comment, timestamp) VALUES (?, ?, ?)",
             ('Test post about SEO and content strategy', 'This is a test comment from the debug tool', datetime.datetime.now().isoformat()))
conn.commit()
conn.close()
print('Inserted dummy row')