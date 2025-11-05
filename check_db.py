import sqlite3, os
p='data/logs.db'
print('DB path exists:', os.path.exists(p))
if not os.path.exists(p):
    raise SystemExit('DB file not found')
conn=sqlite3.connect(p)
cur=conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='comments'")
print('comments table:', cur.fetchone())
try:
    cur.execute('SELECT COUNT(*) FROM comments')
    print('rows in comments:', cur.fetchone()[0])
    cur.execute('SELECT post_text, comment, timestamp FROM comments ORDER BY id DESC LIMIT 5')
    for r in cur.fetchall():
        print('---')
        print('post_text snippet:', (r[0] or '')[:80].replace('\n',' '))
        print('comment:', r[1])
        print('timestamp:', r[2])
except Exception as e:
    print('select error:', e)
conn.close()