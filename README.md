# 🤖 LinkedIn AI Auto Commenter
**An Agentic AI-powered automation tool that reads, analyzes, and comments intelligently on LinkedIn posts related to your niche.**

---

## 🧩 Overview

The **LinkedIn AI Auto Commenter** automatically engages with relevant LinkedIn posts using **Agentic AI** reasoning and **LangChain** models.  
It identifies niche-related posts, decides whether to comment, generates human-like, context-aware responses, and posts them automatically using **Playwright** automation.  
The app comes with a **Streamlit dashboard** for easy control.

---

## 🧠 Key Features

✅ Intelligent comment generation using **LangChain**  
✅ Agentic AI (goal-based decision-making for relevancy)  
✅ Automation using **Playwright**  
✅ Real-time control through **Streamlit UI**  
✅ Configurable tone, niche, and comment intervals  
✅ Works with **Ollama (local)** or **Hugging Face (cloud)** models  
✅ Activity logging in SQLite database  

---

## ⚙️ Architecture Overview

```
Streamlit (UI)
   ↓
AI Engine (LangChain + Agentic AI)
   ↓
Decision Agent (Goal-based reasoning)
   ↓
Playwright Automation (LinkedIn posts)
   ↓
SQLite Logs (comment history)
```

---

## 📁 Folder Structure

```
linkedin_ai_auto_commenter/
│
├── app.py                         # Streamlit main app
│
├── linkedin_bot/
│   ├── __init__.py
│   ├── automation.py              # LinkedIn automation
│   ├── ai_engine.py               # AI model logic (LangChain)
│   ├── decision_agent.py          # Agentic AI decision making
│   ├── config.py                  # Config management
│   └── utils.py                   # Helpers + DB logging
│
├── data/
│   ├── logs.db                    # SQLite log file
│   └── cookies/
│       └── linkedin_state.json    # Saved LinkedIn session cookies
│
├── requirements.txt
└── README.md
```

---

## 🧰 Installation

You can run this project easily using **CMD or VS Code terminal**.

### 1️⃣ Create & activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# or
source venv/bin/activate   # Mac/Linux
```

### 2️⃣ Install dependencies
```bash
pip install streamlit playwright langchain langchain-community huggingface_hub ollama
```

### 3️⃣ Install Playwright browser
```bash
python -m playwright install chromium
```

---

## ⚙️ Optional Setup for AI Models

### 🅰️ Option A — Ollama (Local)
Install [Ollama](https://ollama.com/download)  
Then pull your model:
```bash
ollama pull llama3
```

### 🅱️ Option B — Hugging Face (Cloud)
Create a Hugging Face access token → [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

Login:
```bash
huggingface-cli login
```

---

## 🔐 LinkedIn Login Setup

1. Run:
   ```bash
   python -m playwright codegen https://www.linkedin.com/login
   ```
2. Log in manually  
3. Click on **three dots → Save storage**  
4. Save the file as:  
   ```
   data/cookies/linkedin_state.json
   ```

---

## 🚀 Running the App

Start the Streamlit dashboard:
```bash
streamlit run app.py
```

Then open:
👉 [http://localhost:8501](http://localhost:8501)

---

## 🧩 How Agentic AI Works

| Agent Component | Purpose | Function |
|------------------|----------|-----------|
| **Decision Agent** | Determines *when* to comment | Uses reasoning and goal prompts |
| **AI Comment Generator** | Decides *what* to comment | Generates tone-matched, context-aware text |
| **Automation Bot** | Handles *how* to comment | Uses Playwright to post safely on LinkedIn |

The system uses **goal-based reasoning**, so it learns to focus only on posts that truly match your niche.

---

## 🧾 Logs & Storage

- All comments are saved in:  
  `data/logs.db`

- Each entry includes:  
  - Post snippet  
  - Generated comment  
  - Timestamp  

---

## 🛡️ Safety & Compliance

✅ No LinkedIn password is stored  
✅ Session stored as encrypted cookie file  
✅ Comments spaced by interval to avoid spam detection  
✅ Compliant with LinkedIn Fair Use guidelines  

---

## 🧭 Future Enhancements

- Multi-account support  
- Sentiment-adaptive comment tone  
- Engagement analytics  
- Cloud-based 24/7 scheduler  
- Integration with more LLM providers  

---

## 👨‍💻 Author
**Project by:** Shahzad Ahmad
**Contact Me** shahzadjafar@live.com 
**AI Engineering/ Data Scientist:**   
