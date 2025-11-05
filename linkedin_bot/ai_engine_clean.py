# """
# AICommentGenerator - uses local Ollama or OpenAI to generate short, natural LinkedIn comments.
# Prefers your local model 'shahzadjaafar/model:latest' → falls back to 'llama3.2:latest' → then to smaller safe model.
# """

# from __future__ import annotations
# import os, random

# DEFAULT_OLLAMA_URL = "http://localhost:11434"


# class AICommentGenerator:
#     def __init__(self, model_type: str | None = None, prefer_model: str | None = None):
#         prefer = prefer_model or model_type or "ollama"
#         self.backend_name = "local-fallback"
#         self.llm = None
#         self.chain = None

#         # ✅ prefer your own model first
#         os.environ.setdefault("OLLAMA_MODEL", "shahzadjaafar/model:latest")

#         self._setup_local_fallback()

#         if prefer == "ollama":
#             self._try_init_ollama()
#             if self.llm is None and os.environ.get("OPENAI_API_KEY"):
#                 self._try_init_openai()
#         else:
#             if os.environ.get("OPENAI_API_KEY"):
#                 self._try_init_openai()
#             if self.llm is None:
#                 self._try_init_ollama()

#         print(f"[ai_engine_clean] ✅ ready with backend={self.backend_name}")

#     # ---------------------------------------------------------------------
#     def _try_init_ollama(self):
#         """Try initializing Ollama (LangChain-compatible)"""
#         try:
#             import requests
#             from langchain_community.llms import Ollama
#             from langchain.prompts import PromptTemplate
#             from langchain.schema.runnable import RunnableSequence

#             try:
#                 requests.get(DEFAULT_OLLAMA_URL, timeout=1)
#             except Exception:
#                 print("[ai_engine_clean] ⚠️ Ollama not reachable. Start it via: ollama serve")
#                 return

#             # Try preferred + fallbacks
#             models = [
#                 os.environ.get("OLLAMA_MODEL"),
#                 "llama3.2:latest",
#                 "llama3.2",
#                 "mistral:7b",
#                 "llama3",
#                 "llama2",
#             ]
#             models = [m for m in models if m]

#             for model_name in models:
#                 try:
#                     self.llm = Ollama(model=model_name)
#                     from langchain.prompts import PromptTemplate

#                     prompt = PromptTemplate(
#                         input_variables=["post", "tone"],
#                         template=(
#                             "You are a professional LinkedIn expert. Read the following post "
#                             "and write a {tone} comment in **under 2 short sentences**. "
#                             "Be concise, insightful, and avoid generic praise.\n\nPost:\n{post}\n\nComment:"
#                         ),
#                     )

#                     self.chain = RunnableSequence(prompt, self.llm)
#                     self.backend_name = f"ollama({model_name})"

#                     # Quick test
#                     try:
#                         _ = self.chain.invoke({"post": "Quick test", "tone": "Professional"})
#                         print(f"[ai_engine_clean] ✅ Using model {model_name}")
#                         return
#                     except Exception as e:
#                         print(f"[ai_engine_clean] ⚠️ Test failed for {model_name}: {e}")
#                         self.llm = None
#                         continue
#                 except Exception as e:
#                     print(f"[ai_engine_clean] ⚠️ Could not load model {model_name}: {e}")
#                     continue

#             print("[ai_engine_clean] ❌ No Ollama model initialized.")
#         except Exception as e:
#             print("[ai_engine_clean] ⚠️ Ollama init failed:", e)

#     # ---------------------------------------------------------------------
#     def _try_init_openai(self):
#         """Fallback to OpenAI if key is available"""
#         try:
#             from langchain.chat_models import ChatOpenAI
#             from langchain.prompts import PromptTemplate
#             from langchain.schema.runnable import RunnableSequence

#             self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
#             template = PromptTemplate(
#                 input_variables=["post", "tone"],
#                 template=(
#                     "You are a professional LinkedIn expert. Read the following post "
#                     "and write a {tone} comment in **under 2 short sentences**. "
#                     "Be concise and valuable.\n\nPost:\n{post}\n\nComment:"
#                 ),
#             )

#             self.chain = RunnableSequence(template, self.llm)
#             self.backend_name = "openai"
#             print("[ai_engine_clean] ✅ OpenAI Chat ready.")
#         except Exception as e:
#             print("[ai_engine_clean] ⚠️ OpenAI init failed:", e)

#     # ---------------------------------------------------------------------
#     def _setup_local_fallback(self):
#         """Local deterministic generator if no LLM works"""
#         self.backend_name = "local-fallback"
#         self._fallback_templates = {
#             "Professional": [
#                 "Great point about {keyword} — it really shows how {takeaway}.",
#                 "Interesting insight on {keyword}. Especially true for {takeaway}.",
#             ],
#             "Friendly": [
#                 "Nice post! Loved your take on {keyword}.",
#                 "Totally agree — {keyword} matters a lot for {takeaway}.",
#             ],
#             "Expert": [
#                 "Excellent point about {keyword}. For scaling, consider {takeaway}.",
#                 "Spot on — {keyword} drives better {takeaway}.",
#             ],
#         }

#     def _extract_keyword(self, text: str) -> str:
#         words = [w.strip(".,!?()[]") for w in text.split() if len(w) > 3]
#         return words[0] if words else "this"

#     def _make_takeaway(self, keyword: str) -> str:
#         return random.choice(
#             [f"scaling {keyword}", f"measuring {keyword}", f"aligning teams on {keyword}"]
#         )

#     # ---------------------------------------------------------------------
#     def generate(self, post_text: str, tone: str = "Professional") -> str:
#         if self.chain and self.llm:
#             try:
#                 result = self.chain.invoke({"post": post_text, "tone": tone})
#                 text_out = str(getattr(result, "content", result)).strip()
#                 if text_out:
#                     # limit to ~2 lines
#                     lines = text_out.split(". ")
#                     return ". ".join(lines[:2]).strip()[:240]
#             except Exception as e:
#                 print("[ai_engine_clean] ⚠️ Generation failed, fallback:", e)

#         # Local fallback
#         keyword = self._extract_keyword(post_text)
#         takeaway = self._make_takeaway(keyword)
#         template = random.choice(self._fallback_templates.get(tone, self._fallback_templates["Professional"]))
#         return template.format(keyword=keyword, takeaway=takeaway)[:240]


# if __name__ == "__main__":
#     gen = AICommentGenerator()
#     print("backend:", gen.backend_name)
#     print("comment:", gen.generate("Post about AI trends in 2025", "Professional"))











# # # """Clean AI engine used temporarily while original ai_engine.py is repaired.

# # # """
# # # AICommentGenerator — clean, production-safe AI engine for LinkedIn automation.

# # # ✅ Prefers local Ollama (e.g. llama3.2:latest)
# # # ✅ Falls back to OpenAI Chat if available
# # # ✅ Finally falls back to short, human-like template comments (no API required)

# # # Enhancements:
# # # - Generates concise, natural 1–2 line comments (max 40 words)
# # # - Randomizes timing intervals to simulate natural behavior
# # # """

# # # from __future__ import annotations
# # # import os
# # # import random
# # # from typing import Optional

# # # DEFAULT_OLLAMA_URL = "http://localhost:11434"


# # # class AICommentGenerator:
# # #     def __init__(self, model_type: Optional[str] = None, prefer_model: Optional[str] = None) -> None:
# # #         prefer = prefer_model or model_type or "ollama"
# # #         self.backend_name = "local-fallback"
# # #         self.llm = None
# # #         self.chain = None

# # #         # ✅ Preferred local Ollama model
# # #         os.environ["OLLAMA_MODEL"] = "llama3.2:latest"

# # #         self._setup_local_fallback()

# # #         # Try initializing AI backends in priority order
# # #         if prefer == "ollama":
# # #             self._try_init_ollama()
# # #             if self.llm is None and os.environ.get("OPENAI_API_KEY"):
# # #                 self._try_init_openai()
# # #         else:
# # #             if os.environ.get("OPENAI_API_KEY"):
# # #                 self._try_init_openai()
# # #             if self.llm is None:
# # #                 self._try_init_ollama()

# # #         print(f"[ai_engine_clean] ✅ ready with backend={self.backend_name}")

# # #     # ---------------------------------------------------------------------
# # #     # ✅ Ollama setup
# # #     # ---------------------------------------------------------------------
# # #     def _try_init_ollama(self) -> None:
# # #         try:
# # #             import requests
# # #             requests.get(DEFAULT_OLLAMA_URL, timeout=1)
# # #         except Exception:
# # #             print("[ai_engine_clean] ⚠️ Ollama daemon not reachable (run: ollama serve)")
# # #             return

# # #         try:
# # #             from langchain_community.llms import Ollama as LC_Ollama
# # #             from langchain.prompts import PromptTemplate

# # #             env_model = os.environ.get("OLLAMA_MODEL")
# # #             models_to_try = [m for m in [env_model, "llama3.2:latest", "llama3.2", "llama3"] if m]

# # #             last_exc = None
# # #             for model_name in models_to_try:
# # #                 try:
# # #                     self.llm = LC_Ollama(model=model_name)
# # #                     self.template = PromptTemplate(
# # #                         input_variables=["post", "tone"],
# # #                         template=(
# # #                             "You are a seasoned LinkedIn professional.\n"
# # #                             "Write a concise (1–2 lines, max 40 words) {tone} comment "
# # #                             "that adds genuine insight, not generic praise.\n"
# # #                             "Avoid hashtags, bullet points, or fluff.\n\n"
# # #                             "Post:\n{post}\n\nComment:"
# # #                         ),
# # #                     )
# # #                     try:
# # #                         from langchain.schema.runnable import RunnableSequence
# # #                         self.chain = self.template | self.llm
# # #                     except Exception:
# # #                         from langchain.chains import LLMChain
# # #                         self.chain = LLMChain(prompt=self.template, llm=self.llm)

# # #                     self.backend_name = f"ollama({model_name})"
# # #                     print(f"[ai_engine_clean] ✅ Ollama initialized: {model_name}")
# # #                     return
# # #                 except Exception as e:
# # #                     last_exc = e
# # #                     self.llm = None
# # #                     self.chain = None
# # #                     continue

# # #             print("[ai_engine_clean] ❌ Ollama init failed. Tried:", models_to_try, "Error:", last_exc)
# # #         except Exception as e:
# # #             print("[ai_engine_clean] ❌ Ollama init failed:", e)

# # #     # ---------------------------------------------------------------------
# # #     # ✅ OpenAI Chat fallback
# # #     # ---------------------------------------------------------------------
# # #     def _try_init_openai(self) -> None:
# # #         try:
# # #             from langchain.chat_models import ChatOpenAI
# # #             from langchain.prompts import PromptTemplate
# # #             from langchain.chains import LLMChain

# # #             self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
# # #             self.template = PromptTemplate(
# # #                 input_variables=["post", "tone"],
# # #                 template=(
# # #                     "You are a LinkedIn expert.\n"
# # #                     "Write a short, natural {tone} comment (1–2 lines, under 40 words) "
# # #                     "that adds real value to the post.\n\n"
# # #                     "Post:\n{post}\n\nComment:"
# # #                 ),
# # #             )
# # #             self.chain = LLMChain(prompt=self.template, llm=self.llm)
# # #             self.backend_name = "openai"
# # #             print("[ai_engine_clean] ✅ OpenAI Chat initialized")
# # #         except Exception as e:
# # #             print("[ai_engine_clean] ❌ OpenAI init failed:", e)

# # #     # ---------------------------------------------------------------------
# # #     # ✅ Local fallback (if no LLM available)
# # #     # ---------------------------------------------------------------------
# # #     def _setup_local_fallback(self) -> None:
# # #         self.backend_name = "local-fallback"
# # #         self._fallback_templates = {
# # #             "Professional": [
# # #                 "Insightful take on {keyword}. It's a key factor when {takeaway}.",
# # #                 "Completely agree — {keyword} plays a vital role in {takeaway}.",
# # #             ],
# # #             "Friendly": [
# # #                 "Nice post! Loved the point about {keyword}. So true for {takeaway}.",
# # #                 "Totally relatable! {keyword} really makes a difference in {takeaway}.",
# # #             ],
# # #             "Expert": [
# # #                 "Excellent point — {keyword} directly impacts {takeaway}, often underestimated.",
# # #                 "Solid insight. I'd also connect {keyword} to {takeaway} for better outcomes.",
# # #             ],
# # #         }

# # #     # ---------------------------------------------------------------------
# # #     # ✅ Helper methods
# # #     # ---------------------------------------------------------------------
# # #     def _extract_keyword(self, text: str) -> str:
# # #         if not text:
# # #             return "this"
# # #         first = text.split(".\n")[0].split(".")[0]
# # #         words = [w.strip(".,!?()[]") for w in first.split() if len(w) > 3]
# # #         if not words:
# # #             words = first.split()
# # #         return words[0] if words else "this"

# # #     def _make_takeaway(self, keyword: str) -> str:
# # #         choices = [
# # #             f"scaling {keyword}",
# # #             f"measuring {keyword}",
# # #             f"aligning teams around {keyword}",
# # #             f"balancing creativity and consistency in {keyword}",
# # #         ]
# # #         return random.choice(choices)

# # #     # ---------------------------------------------------------------------
# # #     # ✅ Comment generator (final)
# # #     # ---------------------------------------------------------------------
# # #     def generate(self, post_text: str, tone: str = "Professional") -> str:
# # #         # If LLM is available, use it first
# # #         if getattr(self, "chain", None) and getattr(self, "llm", None):
# # #             try:
# # #                 try:
# # #                     out = self.chain.invoke({"post": post_text, "tone": tone})
# # #                 except Exception:
# # #                     out = self.chain.run({"post": post_text, "tone": tone})

# # #                 text_out = str(getattr(out, "output", out)).strip()
# # #                 if text_out:
# # #                     # Limit to ~40 words max (≈2 lines)
# # #                     words = text_out.split()
# # #                     if len(words) > 40:
# # #                         text_out = " ".join(words[:40]) + "..."
# # #                     return text_out
# # #             except Exception as e:
# # #                 print("[ai_engine_clean] ⚠️ LLM generation failed — fallback:", e)

# # #         # Local fallback comment
# # #         keyword = self._extract_keyword(post_text)
# # #         takeaway = self._make_takeaway(keyword)
# # #         template = random.choice(self._fallback_templates.get(tone, self._fallback_templates["Professional"]))
# # #         comment = template.format(keyword=keyword, takeaway=takeaway)
# # #         if len(comment) > 240:
# # #             comment = comment[:237].rsplit(" ", 1)[0] + "..."
# # #         return comment

# # #     # ---------------------------------------------------------------------
# # #     # 🎲 Random interval generator (human-like delay)
# # #     # ---------------------------------------------------------------------
# # #     @staticmethod
# # #     def random_interval(base_minutes: float) -> float:
# # #         """
# # #         Returns a randomized wait time (in seconds) around the configured interval.
# # #         Example: base=10 → random between 8–12 minutes.
# # #         """
# # #         jitter = random.uniform(-0.2, 0.3)  # ±20–30%
# # #         wait_minutes = max(1, base_minutes * (1 + jitter))
# # #         return wait_minutes * 60  # convert to seconds


# # # # -------------------------------------------------------------------------
# # # # 🔧 Quick test
# # # # -------------------------------------------------------------------------
# # # if __name__ == "__main__":
# # #     gen = AICommentGenerator()
# # #     print("Backend:", gen.backend_name)
# # #     post = "We recently improved our onboarding process for SaaS users."
# # #     print("Generated Comment:", gen.generate(post, "Professional"))
# # #     print("Next interval (sec):", gen.random_interval(10))





# # # Prefer local Ollama; fall back to OpenAI Chat if OPENAI_API_KEY present; else deterministic templates.
# # # This module preserves the external constructor signature used by the app (model_type).
# # # """
# # from __future__ import annotations

# # import os
# # import random
# # from typing import Optional

# # DEFAULT_OLLAMA_URL = "http://localhost:11434"


# # class AICommentGenerator:
# #     def __init__(self, model_type: Optional[str] = None, prefer_model: Optional[str] = None) -> None:
# #         # Keep compatibility: app passes model_type; older calls may pass prefer_model
# #         prefer = prefer_model or model_type or "ollama"
# #         self.backend_name = "local-fallback"
# #         self.llm = None
# #         self.chain = None

# #         # ✅ Force your installed Ollama model
# #         os.environ["OLLAMA_MODEL"] = "llama3.2:latest"

# #         self._setup_local_fallback()

# #         if prefer == "ollama":
# #             self._try_init_ollama()
# #             if self.llm is None and os.environ.get("OPENAI_API_KEY"):
# #                 self._try_init_openai()
# #         else:
# #             if os.environ.get("OPENAI_API_KEY"):
# #                 self._try_init_openai()
# #             if self.llm is None:
# #                 self._try_init_ollama()

# #         print(f"[ai_engine_clean] ready with backend={self.backend_name}")

# #     def _try_init_ollama(self) -> None:
# #         try:
# #             import requests
# #             requests.get(DEFAULT_OLLAMA_URL, timeout=1)
# #         except Exception:
# #             print("[ai_engine_clean] Ollama daemon not reachable (make sure it's running with: ollama serve)")
# #             return

# #         try:
# #             try:
# #                 from langchain_community.llms import Ollama as LC_Ollama  # type: ignore
# #             except Exception:
# #                 from langchain.llms import Ollama as LC_Ollama  # type: ignore

# #             env_model = os.environ.get("OLLAMA_MODEL")
# #             models_to_try = [env_model] if env_model else []
# #             models_to_try += [
# #                 "llama3.2:latest",
# #                 "shahzadjaafar/model:latest",
# #                 "llama3.2",
# #                 "llama3",
# #                 "llama2",
# #                 "llama",
# #             ]

# #             # Remove duplicates and empty values
# #             seen = set()
# #             models_to_try = [m for m in models_to_try if m and not (m in seen or seen.add(m))]

# #             from langchain.prompts import PromptTemplate  # type: ignore

# #             last_exc = None
# #             for model_name in models_to_try:
# #                 try:
# #                     self.llm = LC_Ollama(model=model_name)
# #                     self.template = PromptTemplate(
# #                         input_variables=["post", "tone"],
# #                         template=(
# #                             "You are a professional LinkedIn expert. Read the following post "
# #                             "and write a {tone} comment in under 2 sentences that adds genuine "
# #                             "insight and value. Avoid generic praise.\n\nPost:\n{post}\n\nComment:"
# #                         ),
# #                     )

# #                     # Build chain (prefer new LangChain RunnableSequence)
# #                     try:
# #                         from langchain.schema.runnable import RunnableSequence  # type: ignore
# #                         self.chain = self.template | self.llm
# #                     except Exception:
# #                         from langchain.chains import LLMChain  # type: ignore
# #                         self.chain = LLMChain(prompt=self.template, llm=self.llm)

# #                     self.backend_name = f"ollama({model_name})"

# #                     # ✅ Dry run to test if model works
# #                     try:
# #                         test_out = self.chain.invoke({"post": "test", "tone": "Professional"})
# #                     except Exception:
# #                         # fallback to .run if .invoke not available
# #                         test_out = self.chain.run({"post": "test", "tone": "Professional"})

# #                     print(f"[ai_engine_clean] Ollama initialized via LangChain using model={model_name}")
# #                     return  # success!

# #                 except Exception as e:
# #                     last_exc = e
# #                     self.llm = None
# #                     self.chain = None
# #                     continue

# #             print("[ai_engine_clean] Ollama init failed - tried models:", models_to_try, "error:", last_exc)
# #         except Exception as e:
# #             print("[ai_engine_clean] Ollama init failed:", e)

# #     def _try_init_openai(self) -> None:
# #         try:
# #             from langchain.chat_models import ChatOpenAI  # type: ignore
# #             from langchain.prompts import PromptTemplate  # type: ignore
# #             from langchain.chains import LLMChain  # type: ignore

# #             self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
# #             self.template = PromptTemplate(
# #                 input_variables=["post", "tone"],
# #                 template=(
# #                     "You are a professional LinkedIn expert. Read the following post "
# #                     "and write a {tone} comment in under 2 sentences that adds genuine "
# #                     "insight and value. Avoid generic praise.\n\nPost:\n{post}\n\nComment:"
# #                 ),
# #             )
# #             self.chain = LLMChain(prompt=self.template, llm=self.llm)
# #             self.backend_name = "openai"
# #             print("[ai_engine_clean] OpenAI Chat initialized via LangChain")
# #         except Exception as e:
# #             print("[ai_engine_clean] OpenAI init failed:", e)

# #     def _setup_local_fallback(self) -> None:
# #         self.backend_name = "local-fallback"
# #         self._fallback_templates = {
# #             "Professional": [
# #                 "Sharp insight — particularly around {keyword}. Thanks for sharing.",
# #                 "Great point on {keyword}. I think this also highlights {takeaway}.",
# #             ],
# #             "Friendly": [
# #                 "Nice share! I especially liked the bit about {keyword} — very relatable.",
# #                 "Loved this — the {keyword} angle is spot on. Thanks for posting!",
# #             ],
# #             "Expert": [
# #                 "Excellent analysis on {keyword}. For teams, I'd consider {takeaway} to complement this.",
# #                 "Thought-provoking — {keyword} ties directly to {takeaway}, which often gets overlooked.",
# #             ],
# #         }

# #     def _extract_keyword(self, text: str) -> str:
# #         if not text:
# #             return "this"
# #         first = text.split(".\n")[0].split(".")[0]
# #         words = [w.strip(".,!?()[]") for w in first.split() if len(w) > 3]
# #         if not words:
# #             words = first.split()
# #         return words[0] if words else "this"

# #     def _make_takeaway(self, keyword: str) -> str:
# #         choices = [
# #             f"scaling {keyword}",
# #             f"measuring {keyword}",
# #             f"aligning teams around {keyword}",
# #             f"balancing speed and quality in {keyword}",
# #         ]
# #         return random.choice(choices)

# #     def generate(self, post_text: str, tone: str = "Professional") -> str:
# #         if getattr(self, "chain", None) is not None and getattr(self, "llm", None) is not None:
# #             try:
# #                 # Prefer invoke() where available
# #                 try:
# #                     out = self.chain.invoke({"post": post_text, "tone": tone})
# #                 except Exception:
# #                     out = self.chain.run({"post": post_text, "tone": tone})

# #                 text_out = str(getattr(out, "output", out)).strip()
# #                 if isinstance(text_out, str) and text_out:
# #                     return text_out
# #             except Exception as e:
# #                 print("[ai_engine_clean] LLM chain generation failed, falling back:", e)

# #         # Local fallback
# #         keyword = self._extract_keyword(post_text)
# #         takeaway = self._make_takeaway(keyword)
# #         template = random.choice(self._fallback_templates.get(tone, self._fallback_templates["Professional"]))
# #         comment = template.format(keyword=keyword, takeaway=takeaway)
# #         if len(comment) > 240:
# #             comment = comment[:237].rsplit(" ", 1)[0] + "..."
# #         return comment


# # if __name__ == "__main__":
# #     gen = AICommentGenerator()
# #     print("backend:", gen.backend_name)
# #     print("comment:", gen.generate("Quick post about improving onboarding flows for SMBs", tone="Professional"))



"""
AICommentGenerator - clean, production-safe comment generator for LinkedIn automation.
✅ Prefers your local Ollama model 'shahzadjaafar/model:latest'
✅ Falls back to 'llama3.2:latest' or OpenAI if available
✅ Always ensures concise, natural 1–2 line comments
"""

from __future__ import annotations
import os, random

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class AICommentGenerator:
    def __init__(self, model_type: str | None = None, prefer_model: str | None = None):
        prefer = prefer_model or model_type or "ollama"
        self.backend_name = "local-fallback"
        self.llm = None
        self.chain = None

        # ✅ Prefer your own local model
        os.environ.setdefault("OLLAMA_MODEL", "shahzadjaafar/model:latest")

        self._setup_local_fallback()

        if prefer == "ollama":
            self._try_init_ollama()
            if self.llm is None and os.environ.get("OPENAI_API_KEY"):
                self._try_init_openai()
        else:
            if os.environ.get("OPENAI_API_KEY"):
                self._try_init_openai()
            if self.llm is None:
                self._try_init_ollama()

        print(f"[ai_engine_clean] ✅ Ready with backend={self.backend_name}")

    # ---------------------------------------------------------------------
    def _try_init_ollama(self):
        """Try initializing Ollama via LangChain"""
        try:
            import requests
            requests.get(DEFAULT_OLLAMA_URL, timeout=1)
        except Exception:
            print("[ai_engine_clean] ⚠️ Ollama not reachable. Start it via: ollama serve")
            return

        try:
            from langchain_community.llms import Ollama
            from langchain.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            models = [
                os.environ.get("OLLAMA_MODEL"),
                "llama3.2:latest",
                "llama3.2",
                "mistral:7b",
                "llama3",
                "llama2",
            ]
            models = [m for m in models if m]

            for model_name in models:
                try:
                    self.llm = Ollama(model=model_name)
                    prompt = PromptTemplate.from_template(
                        """You are a professional LinkedIn expert.
Write a {tone} comment (1–2 lines, under 40 words) that sounds natural and insightful.
Avoid generic praise, emojis, or hashtags.

Post:
{post}

Comment:"""
                    )

                    # ✅ modern Runnable chain
                    from langchain_core.runnables import RunnableSequence
                    self.chain = RunnableSequence(prompt, self.llm, StrOutputParser())

                    # Quick test
                    _ = self.chain.invoke({"post": "Quick test about AI", "tone": "Professional"})
                    self.backend_name = f"ollama({model_name})"
                    print(f"[ai_engine_clean] ✅ Using Ollama model: {model_name}")
                    return
                except Exception as e:
                    print(f"[ai_engine_clean] ⚠️ Could not load model {model_name}: {e}")
                    self.llm = None
                    self.chain = None
                    continue

            print("[ai_engine_clean] ❌ No Ollama model could be initialized.")
        except Exception as e:
            print("[ai_engine_clean] ⚠️ Ollama init failed:", e)

    # ---------------------------------------------------------------------
    def _try_init_openai(self):
        """Fallback to OpenAI if key is available"""
        try:
            from langchain.chat_models import ChatOpenAI
            from langchain.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.runnables import RunnableSequence

            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)
            prompt = PromptTemplate.from_template(
                """You are a professional LinkedIn expert.
Write a {tone} comment (1–2 lines, under 40 words) that adds real insight to the post.
Avoid generic praise or emojis.

Post:
{post}

Comment:"""
            )
            self.chain = RunnableSequence(prompt, self.llm, StrOutputParser())
            self.backend_name = "openai"
            print("[ai_engine_clean] ✅ OpenAI Chat initialized.")
        except Exception as e:
            print("[ai_engine_clean] ⚠️ OpenAI init failed:", e)

    # ---------------------------------------------------------------------
    def _setup_local_fallback(self):
        """If no LLM available, fall back to templated comments"""
        self.backend_name = "local-fallback"
        self._fallback_templates = {
            "Professional": [
                "Great point about {keyword} — it highlights how {takeaway}.",
                "Insightful perspective on {keyword}. Especially relevant for {takeaway}.",
            ],
            "Friendly": [
                "Nice post! Loved your view on {keyword}.",
                "Totally agree — {keyword} really matters for {takeaway}.",
            ],
            "Expert": [
                "Excellent insight about {keyword}. For scaling, I’d focus on {takeaway}.",
                "Spot on — {keyword} directly impacts {takeaway}.",
            ],
        }

    def _extract_keyword(self, text: str) -> str:
        words = [w.strip(".,!?()[]") for w in text.split() if len(w) > 3]
        return words[0] if words else "this"

    def _make_takeaway(self, keyword: str) -> str:
        return random.choice(
            [f"scaling {keyword}", f"measuring {keyword}", f"aligning teams on {keyword}"]
        )

    # ---------------------------------------------------------------------
    def generate(self, post_text: str, tone: str = "Professional") -> str:
        """Generate a short (1–2 line) comment."""
        if self.chain and self.llm:
            try:
                result = self.chain.invoke({"post": post_text, "tone": tone})
                text_out = str(getattr(result, "content", result)).strip()
                if text_out:
                    # Limit to ~2 lines / 40 words
                    words = text_out.split()
                    if len(words) > 40:
                        text_out = " ".join(words[:40]) + "..."
                    return text_out.strip()[:240]
            except Exception as e:
                print("[ai_engine_clean] ⚠️ Generation failed, using fallback:", e)

        # Fallback template
        keyword = self._extract_keyword(post_text)
        takeaway = self._make_takeaway(keyword)
        template = random.choice(self._fallback_templates.get(tone, self._fallback_templates["Professional"]))
        return template.format(keyword=keyword, takeaway=takeaway)[:240]


if __name__ == "__main__":
    gen = AICommentGenerator()
    print("backend:", gen.backend_name)
    print("comment:", gen.generate("AI is transforming how teams collaborate and automate marketing workflows.", "Professional"))
