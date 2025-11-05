
# import asyncio
# import os
# import time
# import random
# import json
# import sqlite3
# from datetime import datetime, timedelta
# from pathlib import Path
# from linkedin_bot.utils import log_comment, init_db

# # Ensure database exists
# init_db()

# # Ensure proper async loop on Windows
# if os.name == "nt":
#     try:
#         asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
#     except AttributeError:
#         pass


# class LinkedInBot:
#     def __init__(self, agent, interval=10, force=False):
#         """
#         interval: minutes between feed scans
#         force: always comment regardless of relevance
#         """
#         self.agent = agent
#         self.force = force
#         self.interval = interval * 60  # convert minutes → seconds
#         self.min_delay = 45
#         self.max_delay = 120

#         # SQLite connection (thread-safe)
#         self.conn = sqlite3.connect("data/logs.db", check_same_thread=False)
#         self.cur = self.conn.cursor()
#         self.cur.execute(
#             """CREATE TABLE IF NOT EXISTS comments (
#                 id INTEGER PRIMARY KEY AUTOINCREMENT,
#                 post_text TEXT UNIQUE,
#                 comment TEXT,
#                 timestamp TEXT
#             )"""
#         )
#         self.conn.commit()

#     def _random_wait(self, action_name="next action"):
#         """Random human-like delay between actions."""
#         delay = random.uniform(self.min_delay, self.max_delay)
#         jitter = random.uniform(-5, 5)
#         delay = max(15, delay + jitter)
#         next_time = datetime.now() + timedelta(seconds=delay)
#         print(f"⏳ Waiting {int(delay)}s before {action_name} (next at {next_time.strftime('%H:%M:%S')})")
#         while delay > 0:
#             print(f"   ... {int(delay)}s remaining", end="\r", flush=True)
#             time.sleep(5)
#             delay -= 5
#         print("\n✅ Continuing...")

#     def _already_commented(self, post_text: str) -> bool:
#         """Check if the post was already commented on."""
#         self.cur.execute("SELECT 1 FROM comments WHERE post_text = ?", (post_text,))
#         return self.cur.fetchone() is not None

#     def _save_comment(self, post_text: str, comment: str):
#         """Save comment to SQLite safely, avoid UNIQUE constraint errors."""
#         timestamp = datetime.now().isoformat()
#         try:
#             self.cur.execute("SELECT 1 FROM comments WHERE post_text = ?", (post_text,))
#             if not self.cur.fetchone():
#                 self.cur.execute(
#                     "INSERT INTO comments (post_text, comment, timestamp) VALUES (?, ?, ?)",
#                     (post_text, comment, timestamp),
#                 )
#             else:
#                 self.cur.execute(
#                     "UPDATE comments SET comment = ?, timestamp = ? WHERE post_text = ?",
#                     (comment, timestamp, post_text),
#                 )
#             self.conn.commit()
#         except Exception as e:
#             print("⚠️ Failed to save comment:", e)

#     def run(self, email: str = None, password: str = None, interactive=False):
#         """Start automation: read feed, decide, and auto-comment."""
#         try:
#             from playwright.sync_api import sync_playwright, TimeoutError
#         except Exception as e:
#             raise ModuleNotFoundError(
#                 "Playwright not installed. Run: pip install playwright && playwright install"
#             ) from e

#         storage_path = Path("data/cookies/linkedin_state.json")
#         storage_path.parent.mkdir(parents=True, exist_ok=True)

#         use_storage = False
#         if storage_path.exists() and storage_path.stat().st_size > 0:
#             try:
#                 json.load(storage_path.open("r", encoding="utf-8"))
#                 use_storage = True
#             except Exception:
#                 print("⚠️ Invalid cookies, login required.")

#         with sync_playwright() as p:
#             browser = p.chromium.launch(headless=False, slow_mo=200)
#             context = browser.new_context(storage_state=str(storage_path)) if use_storage else browser.new_context()
#             page = context.new_page()

#             # Auto-login if needed
#             if not use_storage and email and password:
#                 try:
#                     print("🔐 Logging in...")
#                     page.goto("https://www.linkedin.com/login", timeout=120000, wait_until="networkidle")
#                     page.fill("input[name='session_key']", email)
#                     page.fill("input[name='session_password']", password)
#                     page.click("button[type='submit']")
#                     page.wait_for_selector("div.feed-shared-update-v2", timeout=120000)
#                     context.storage_state(path=str(storage_path))
#                     print(f"✅ Saved session to {storage_path}")
#                     use_storage = True
#                 except TimeoutError:
#                     print("⏱️ Login timed out, please log in manually.")
#                 except Exception as e:
#                     print("❌ Login failed:", e)

#             # Manual login fallback
#             if not use_storage:
#                 print("🧠 Manual login required in the opened browser...")
#                 retries = 3
#                 for attempt in range(retries):
#                     try:
#                         page.goto("https://www.linkedin.com/feed/", timeout=120000, wait_until="networkidle")
#                         page.wait_for_selector("div.feed-shared-update-v2", timeout=120000)
#                         context.storage_state(path=str(storage_path))
#                         print(f"✅ Saved new session to {storage_path}")
#                         break
#                     except TimeoutError:
#                         print(f"⚠️ Feed load failed (attempt {attempt+1}/{retries})")
#                         if attempt < retries - 1:
#                             time.sleep(5)
#                         else:
#                             print("❌ Could not load LinkedIn feed. Exiting.")
#                             return

#             print("🚀 Bot live on your LinkedIn feed.")
#             page.goto("https://www.linkedin.com/feed/", timeout=120000, wait_until="networkidle")
#             time.sleep(5)

#             # Continuous automation loop
#             while True:
#                 posts = page.query_selector_all("div.feed-shared-update-v2")

#                 for post in posts:
#                     try:
#                         post_text = post.inner_text().strip()[:700]
#                         if not post_text or self._already_commented(post_text):
#                             print("⏭️ Post already commented or empty, skipping.")
#                             continue

#                         print("\n📝 Post snippet:")
#                         print(post_text[:200].replace("\n", " ") + "...")

#                         # Generate comment
#                         comment = self.agent.ai.generate(post_text, "Professional") if self.force else self.agent.decide_and_generate(post_text, "Professional")

#                         if comment:
#                             try:
#                                 comment_button = post.query_selector("button[aria-label*='Comment']")
#                                 if comment_button:
#                                     comment_button.click()
#                                     time.sleep(1)
#                                     comment_box = page.query_selector("div[role='textbox']")

#                                     if comment_box:
#                                         comment_box.click()
#                                         comment_box.fill(comment)

#                                         if interactive:
#                                             input(f"💬 Comment ready: '{comment}'\nPress Enter to post...")
#                                         page.keyboard.press("Enter")

#                                         self._save_comment(post_text, comment)
#                                         log_comment(post_text, comment)
#                                         print(f"✅ Commented: {comment[:80]}...")
#                                     else:
#                                         print("⚠️ Comment box not found.")
#                                 else:
#                                     print("⚠️ Comment button not found.")
#                             except Exception as e:
#                                 print(f"❌ Error commenting: {e}")
#                         else:
#                             print("⏭️ Skipped post (irrelevant).")

#                         self._random_wait("next post")

#                     except Exception as e:
#                         print("⚠️ Error processing post:", e)

#                 print(f"\n🔁 Reloading feed after {self.interval / 60:.1f} minutes...")
#                 time.sleep(self.interval)
#                 page.reload()


import asyncio
import os
import time
import random
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from linkedin_bot.utils import log_comment, init_db

# Ensure database exists
init_db()

# Ensure proper async loop on Windows
if os.name == "nt":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    except AttributeError:
        pass


class LinkedInBot:
    def __init__(self, agent, interval=10, force=False):
        """
        interval: minutes between feed scans
        force: always comment regardless of relevance
        """
        self.agent = agent
        self.force = force
        self.interval = interval * 60  # convert minutes → seconds
        self.min_delay = 45
        self.max_delay = 120

        # SQLite connection (thread-safe)
        self.db_path = Path("data/logs.db")
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.cur = self.conn.cursor()
        self.cur.execute(
            """CREATE TABLE IF NOT EXISTS comments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_text TEXT UNIQUE,
                comment TEXT,
                timestamp TEXT
            )"""
        )
        self.conn.commit()

        # Cookie path
        self.cookie_path = Path("data/cookies/linkedin_state.json")
        self.cookie_path.parent.mkdir(parents=True, exist_ok=True)

    # ---------------------------
    # Cache clearing method
    # ---------------------------
    def clear_cache(self):
        """Delete SQLite logs and LinkedIn cookies."""
        try:
            if self.conn:
                self.conn.close()
            if self.db_path.exists():
                self.db_path.unlink()
                print("✅ Deleted logs.db")
            if self.cookie_path.exists():
                self.cookie_path.unlink()
                print("✅ Deleted LinkedIn cookies")
            # Recreate DB
            init_db()
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.cur = self.conn.cursor()
            self.cur.execute(
                """CREATE TABLE IF NOT EXISTS comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_text TEXT UNIQUE,
                    comment TEXT,
                    timestamp TEXT
                )"""
            )
            self.conn.commit()
        except Exception as e:
            print("⚠️ Failed to clear cache:", e)

    # ---------------------------
    def _random_wait(self, action_name="next action"):
        """Random human-like delay between actions."""
        delay = random.uniform(self.min_delay, self.max_delay)
        jitter = random.uniform(-5, 5)
        delay = max(15, delay + jitter)
        next_time = datetime.now() + timedelta(seconds=delay)
        print(f"⏳ Waiting {int(delay)}s before {action_name} (next at {next_time.strftime('%H:%M:%S')})")
        while delay > 0:
            print(f"   ... {int(delay)}s remaining", end="\r", flush=True)
            time.sleep(5)
            delay -= 5
        print("\n✅ Continuing...")

    def _already_commented(self, post_text: str) -> bool:
        """Check if the post was already commented on."""
        self.cur.execute("SELECT 1 FROM comments WHERE post_text = ?", (post_text,))
        return self.cur.fetchone() is not None

    def _save_comment(self, post_text: str, comment: str):
        """Save comment to SQLite safely, avoid UNIQUE constraint errors."""
        timestamp = datetime.now().isoformat()
        try:
            self.cur.execute("SELECT 1 FROM comments WHERE post_text = ?", (post_text,))
            if not self.cur.fetchone():
                self.cur.execute(
                    "INSERT INTO comments (post_text, comment, timestamp) VALUES (?, ?, ?)",
                    (post_text, comment, timestamp),
                )
            else:
                self.cur.execute(
                    "UPDATE comments SET comment = ?, timestamp = ? WHERE post_text = ?",
                    (comment, timestamp, post_text),
                )
            self.conn.commit()
        except Exception as e:
            print("⚠️ Failed to save comment:", e)

    # ---------------------------
    def run(self, email: str = None, password: str = None, interactive=False):
        """Start automation: read feed, decide, and auto-comment."""
        try:
            from playwright.sync_api import sync_playwright, TimeoutError
        except Exception as e:
            raise ModuleNotFoundError(
                "Playwright not installed. Run: pip install playwright && playwright install"
            ) from e

        use_storage = False
        if self.cookie_path.exists() and self.cookie_path.stat().st_size > 0:
            try:
                json.load(self.cookie_path.open("r", encoding="utf-8"))
                use_storage = True
            except Exception:
                print("⚠️ Invalid cookies, login required.")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False, slow_mo=200)
            context = browser.new_context(storage_state=str(self.cookie_path)) if use_storage else browser.new_context()
            page = context.new_page()

            # Auto-login if needed
            if not use_storage and email and password:
                try:
                    print("🔐 Logging in...")
                    page.goto("https://www.linkedin.com/login", timeout=120000, wait_until="networkidle")
                    page.fill("input[name='session_key']", email)
                    page.fill("input[name='session_password']", password)
                    page.click("button[type='submit']")
                    page.wait_for_selector("div.feed-shared-update-v2", timeout=120000)
                    context.storage_state(path=str(self.cookie_path))
                    print(f"✅ Saved session to {self.cookie_path}")
                    use_storage = True
                except TimeoutError:
                    print("⏱️ Login timed out, please log in manually.")
                except Exception as e:
                    print("❌ Login failed:", e)

            # Manual login fallback
            if not use_storage:
                print("🧠 Manual login required in the opened browser...")
                retries = 3
                for attempt in range(retries):
                    try:
                        page.goto("https://www.linkedin.com/feed/", timeout=120000, wait_until="networkidle")
                        page.wait_for_selector("div.feed-shared-update-v2", timeout=120000)
                        context.storage_state(path=str(self.cookie_path))
                        print(f"✅ Saved new session to {self.cookie_path}")
                        break
                    except TimeoutError:
                        print(f"⚠️ Feed load failed (attempt {attempt+1}/{retries})")
                        if attempt < retries - 1:
                            time.sleep(5)
                        else:
                            print("❌ Could not load LinkedIn feed. Exiting.")
                            return

            print("🚀 Bot live on your LinkedIn feed.")
            page.goto("https://www.linkedin.com/feed/", timeout=120000, wait_until="networkidle")
            time.sleep(5)

            # Continuous automation loop
            while True:
                posts = page.query_selector_all("div.feed-shared-update-v2")

                for post in posts:
                    try:
                        post_text = post.inner_text().strip()[:700]
                        if not post_text or self._already_commented(post_text):
                            print("⏭️ Post already commented or empty, skipping.")
                            continue

                        print("\n📝 Post snippet:")
                        print(post_text[:200].replace("\n", " ") + "...")

                        # Generate comment
                        comment = self.agent.ai.generate(post_text, "Professional") if self.force else self.agent.decide_and_generate(post_text, "Professional")

                        if comment:
                            try:
                                comment_button = post.query_selector("button[aria-label*='Comment']")
                                if comment_button:
                                    comment_button.click()
                                    time.sleep(1)
                                    comment_box = page.query_selector("div[role='textbox']")

                                    if comment_box:
                                        comment_box.click()
                                        comment_box.fill(comment)

                                        if interactive:
                                            input(f"💬 Comment ready: '{comment}'\nPress Enter to post...")
                                        page.keyboard.press("Enter")

                                        self._save_comment(post_text, comment)
                                        log_comment(post_text, comment)
                                        print(f"✅ Commented: {comment[:80]}...")
                                    else:
                                        print("⚠️ Comment box not found.")
                                else:
                                    print("⚠️ Comment button not found.")
                            except Exception as e:
                                print(f"❌ Error commenting: {e}")
                        else:
                            print("⏭️ Skipped post (irrelevant).")

                        self._random_wait("next post")

                    except Exception as e:
                        print("⚠️ Error processing post:", e)

                print(f"\n🔁 Reloading feed after {self.interval / 60:.1f} minutes...")
                time.sleep(self.interval)
                page.reload()
