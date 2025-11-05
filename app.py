

# import streamlit as st
# from linkedin_bot.automation import LinkedInBot  # ✅ matches new filename
# from linkedin_bot.ai_engine_clean import AICommentGenerator
# from linkedin_bot.decision_agent import CommentDecisionAgent
# from linkedin_bot.config import load_config, save_config
# from linkedin_bot.utils import init_db
# import sqlite3
# import os
# import json
# import sys
# import threading
# import time
# from pathlib import Path

# # ✅ Ensure database exists
# init_db()

# # ✅ Check saved LinkedIn login session
# storage_path = Path("data/cookies/linkedin_state.json")
# storage_valid = False
# if storage_path.exists():
#     try:
#         if storage_path.stat().st_size > 0:
#             json.load(storage_path.open("r", encoding="utf-8"))
#             storage_valid = True
#     except Exception:
#         storage_valid = False

# config = load_config()

# # ✅ Sidebar controls
# st.sidebar.header("⚙️ Settings")
# config["niche"] = st.sidebar.text_input("Target Niche / Keywords", config.get("niche", "SEO, Marketing"))
# config["tone"] = st.sidebar.selectbox("Comment Tone", ["Professional", "Friendly", "Expert"], index=0)
# config["model"] = st.sidebar.selectbox("AI Model", ["ollama", "huggingface"], index=0)
# config["hf_token"] = st.sidebar.text_input("HuggingFace API Token", config.get("hf_token", ""), type="password")
# config["interval"] = st.sidebar.slider("Comment Interval (minutes)", 1, 60, config.get("interval", 10))

# force_comments = st.sidebar.checkbox(
#     "Force comments (debug)",
#     value=False,
#     help="When enabled, the bot will comment on every post without checking relevance (use only for testing)."
# )

# run = st.sidebar.button("▶️ Start Automation")
# stop = st.sidebar.button("⛔ Stop Bot")  # ✅ new stop button

# save_config(config)

# # ✅ Optional HuggingFace auth
# if config.get("hf_token"):
#     os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["hf_token"]

# # ✅ Ask for credentials if cookies missing
# email = password = None
# if not storage_valid:
#     st.sidebar.markdown("### 🔑 LinkedIn Login Required")
#     email = st.sidebar.text_input("LinkedIn Email")
#     password = st.sidebar.text_input("LinkedIn Password", type="password")
#     st.sidebar.caption("Used once to log in. Never stored locally.")
# else:
#     st.sidebar.success("✅ Logged in via saved LinkedIn session")

# # Global flag to stop bot
# st.session_state.setdefault("stop_bot", False)

# # ✅ Start automation
# if run:
#     st.success("🚀 Starting AI LinkedIn Comment Bot...")

#     try:
#         ai = AICommentGenerator(model_type=config["model"])
#     except ModuleNotFoundError:
#         st.error("Missing LLM dependencies.")
#         st.code(f'"{sys.executable}" -m pip install langchain langchain-community huggingface-hub ollama')
#         st.stop()
#     except RuntimeError as e:
#         st.error(str(e))
#         st.stop()

#     agent = CommentDecisionAgent(ai, niche=config["niche"])
#     bot = LinkedInBot(agent, interval=config["interval"], force=force_comments)

#     log_box = st.empty()
#     status_placeholder = st.empty()

#     def run_bot():
#         try:
#             if not storage_valid and email and password:
#                 bot.run(email=email, password=password)
#             else:
#                 bot.run()
#         except Exception as e:
#             print("Automation thread error:", e)

#     # ✅ Background thread
#     t = threading.Thread(target=run_bot, daemon=True)
#     t.start()

#     with st.spinner("🤖 Automation running..."):
#         while t.is_alive():
#             if st.session_state["stop_bot"]:
#                 st.warning("🛑 Stopping bot...")
#                 break

#             # ✅ Show logs live (read last 10 lines of data/decisions.jsonl)
#             diag_path = Path("data/decisions.jsonl")
#             if diag_path.exists():
#                 try:
#                     lines = diag_path.read_text(encoding="utf-8").splitlines()[-10:]
#                     display = "\n".join(
#                         [json.loads(l).get("comment_preview", "") or json.loads(l).get("decision_raw", "") for l in lines]
#                     )
#                     log_box.text_area("📜 Live Logs", display, height=200)
#                 except Exception:
#                     pass

#             status_placeholder.info("Bot active — check Recent Comments below.")
#             time.sleep(5)

#     status_placeholder.success("✅ Automation completed or stopped.")

# # ✅ Stop automation manually
# if stop:
#     st.session_state["stop_bot"] = True
#     st.warning("🛑 Bot stopped manually.")

# # ✅ Show recent comments (auto-refresh)
# st.subheader("🗂️ Recent Comments")
# try:
#     conn = sqlite3.connect("data/logs.db")
#     cur = conn.cursor()
#     cur.execute("SELECT post_text, comment, timestamp FROM comments ORDER BY id DESC LIMIT 20")
#     rows = cur.fetchall()
#     conn.close()
# except Exception as e:
#     rows = []
#     st.error(f"Error reading comments: {e}")

# if not rows:
#     st.write("No comments yet.")
# else:
#     for post_text, comment_text, created_at in rows:
#         snippet = post_text.replace("\n", " ")[:120]
#         st.markdown(f"- **{created_at}** — {snippet}\n\n    {comment_text}")

# # ✅ Diagnostics section
# st.subheader("🧠 Diagnostics — Decisions & Generations")
# diag_path = Path("data/decisions.jsonl")
# if diag_path.exists():
#     try:
#         lines = diag_path.read_text(encoding="utf-8").splitlines()[-40:]
#         for e in reversed([json.loads(l) for l in lines]):
#             ts = e.get("timestamp", "")
#             etype = e.get("type", "")
#             snippet = (e.get("post_snippet") or "")[:120]
#             if etype == "decision":
#                 st.markdown(f"- **{ts}** — Decision — {snippet}\n    `{e.get('decision_raw')}`")
#             else:
#                 st.markdown(f"- **{ts}** — Generation — {snippet}\n    {e.get('comment_preview')}")
#     except Exception as e:
#         st.error(f"Failed to read diagnostics: {e}")
# else:
#     st.write("No diagnostics logged yet.")


import streamlit as st
from linkedin_bot.automation import LinkedInBot  # ✅ matches new filename
from linkedin_bot.ai_engine_clean import AICommentGenerator
from linkedin_bot.decision_agent import CommentDecisionAgent
from linkedin_bot.config import load_config, save_config
from linkedin_bot.utils import init_db
import sqlite3
import os
import json
import sys
import threading
import time
from pathlib import Path
import shutil

# ✅ Ensure database exists
init_db()

# ✅ Check saved LinkedIn login session
storage_path = Path("data/cookies/linkedin_state.json")
storage_valid = False
if storage_path.exists():
    try:
        if storage_path.stat().st_size > 0:
            json.load(storage_path.open("r", encoding="utf-8"))
            storage_valid = True
    except Exception:
        storage_valid = False

config = load_config()

# ✅ Sidebar controls
st.sidebar.header("⚙️ Settings")
config["niche"] = st.sidebar.text_input("Target Niche / Keywords", config.get("niche", "SEO, Marketing"))
config["tone"] = st.sidebar.selectbox("Comment Tone", ["Professional", "Friendly", "Expert"], index=0)
config["model"] = st.sidebar.selectbox("AI Model", ["ollama", "huggingface"], index=0)
config["hf_token"] = st.sidebar.text_input("HuggingFace API Token", config.get("hf_token", ""), type="password")
config["interval"] = st.sidebar.slider("Comment Interval (minutes)", 1, 60, config.get("interval", 10))

force_comments = st.sidebar.checkbox(
    "Force comments (debug)",
    value=False,
    help="When enabled, the bot will comment on every post without checking relevance (use only for testing)."
)

run = st.sidebar.button("▶️ Start Automation")
stop = st.sidebar.button("⛔ Stop Bot")  
clear_cache = st.sidebar.button("🗑️ Clear Cache / Reset Bot")  # ✅ new button

save_config(config)

# ✅ Optional HuggingFace auth
if config.get("hf_token"):
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["hf_token"]

# ✅ Ask for credentials if cookies missing
email = password = None
if not storage_valid:
    st.sidebar.markdown("### 🔑 LinkedIn Login Required")
    email = st.sidebar.text_input("LinkedIn Email")
    password = st.sidebar.text_input("LinkedIn Password", type="password")
    st.sidebar.caption("Used once to log in. Never stored locally.")
else:
    st.sidebar.success("✅ Logged in via saved LinkedIn session")

# Global flag to stop bot
st.session_state.setdefault("stop_bot", False)

# ✅ Clear cache / reset bot
if clear_cache:
    try:
        # Delete cookies
        if storage_path.exists():
            storage_path.unlink()
        # Clear database
        db_path = Path("data/logs.db")
        if db_path.exists():
            db_path.unlink()
        # Clear decisions log
        diag_path = Path("data/decisions.jsonl")
        if diag_path.exists():
            diag_path.unlink()
        # Re-init DB
        init_db()
        st.success("✅ Cache cleared! Cookies, comments, and diagnostics removed.")
        storage_valid = False
    except Exception as e:
        st.error(f"Failed to clear cache: {e}")

# ✅ Start automation
if run:
    st.success("🚀 Starting AI LinkedIn Comment Bot...")

    try:
        ai = AICommentGenerator(model_type=config["model"])
    except ModuleNotFoundError:
        st.error("Missing LLM dependencies.")
        st.code(f'"{sys.executable}" -m pip install langchain langchain-community huggingface-hub ollama')
        st.stop()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()

    agent = CommentDecisionAgent(ai, niche=config["niche"])
    bot = LinkedInBot(agent, interval=config["interval"], force=force_comments)

    log_box = st.empty()
    status_placeholder = st.empty()

    def run_bot():
        try:
            if not storage_valid and email and password:
                bot.run(email=email, password=password)
            else:
                bot.run()
        except Exception as e:
            print("Automation thread error:", e)

    # ✅ Background thread
    t = threading.Thread(target=run_bot, daemon=True)
    t.start()

    with st.spinner("🤖 Automation running..."):
        while t.is_alive():
            if st.session_state["stop_bot"]:
                st.warning("🛑 Stopping bot...")
                break

            # ✅ Show live logs (last 10 lines of data/decisions.jsonl)
            diag_path = Path("data/decisions.jsonl")
            if diag_path.exists():
                try:
                    lines = diag_path.read_text(encoding="utf-8").splitlines()[-10:]
                    display = "\n".join(
                        [json.loads(l).get("comment_preview", "") or json.loads(l).get("decision_raw", "") for l in lines]
                    )
                    log_box.text_area("📜 Live Logs", display, height=200)
                except Exception:
                    pass

            status_placeholder.info("Bot active — check Recent Comments below.")
            time.sleep(5)

    status_placeholder.success("✅ Automation completed or stopped.")

# ✅ Stop automation manually
if stop:
    st.session_state["stop_bot"] = True
    st.warning("🛑 Bot stopped manually.")

# ✅ Show recent comments (auto-refresh)
st.subheader("🗂️ Recent Comments")
try:
    conn = sqlite3.connect("data/logs.db")
    cur = conn.cursor()
    cur.execute("SELECT post_text, comment, timestamp FROM comments ORDER BY id DESC LIMIT 20")
    rows = cur.fetchall()
    conn.close()
except Exception as e:
    rows = []
    st.error(f"Error reading comments: {e}")

if not rows:
    st.write("No comments yet.")
else:
    for post_text, comment_text, created_at in rows:
        snippet = post_text.replace("\n", " ")[:120]
        st.markdown(f"- **{created_at}** — {snippet}\n\n    {comment_text}")

# ✅ Diagnostics section
st.subheader("🧠 Diagnostics — Decisions & Generations")
diag_path = Path("data/decisions.jsonl")
if diag_path.exists():
    try:
        lines = diag_path.read_text(encoding="utf-8").splitlines()[-40:]
        for e in reversed([json.loads(l) for l in lines]):
            ts = e.get("timestamp", "")
            etype = e.get("type", "")
            snippet = (e.get("post_snippet") or "")[:120]
            if etype == "decision":
                st.markdown(f"- **{ts}** — Decision — {snippet}\n    `{e.get('decision_raw')}`")
            else:
                st.markdown(f"- **{ts}** — Generation — {snippet}\n    {e.get('comment_preview')}")
    except Exception as e:
        st.error(f"Failed to read diagnostics: {e}")
else:
    st.write("No diagnostics logged yet.")
