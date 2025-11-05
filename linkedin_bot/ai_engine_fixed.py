"""Clean AI comment generator (fixed copy).

This module provides a safe, deterministic fallback so the app runs even when
LangChain/HuggingFace/Ollama aren't available. Prefer using this file while the
original ai_engine.py is being repaired.
"""

import os
import random
from typing import Optional


class AICommentGenerator:
    def __init__(self, model_type: str = "ollama"):
        self.backend_name = "local-fallback"
        have_langchain = False

        # Lazy import langchain components if present
        try:
            from langchain_community.llms import Ollama, HuggingFaceHub  # type: ignore
            from langchain.prompts import PromptTemplate  # type: ignore
            from langchain.chains import LLMChain  # type: ignore

            self._Ollama = Ollama
            self._HuggingFaceHub = HuggingFaceHub
            self._PromptTemplate = PromptTemplate
            self._LLMChain = LLMChain
            have_langchain = True
        except Exception:
            have_langchain = False
            print("[ai_engine_fixed] langchain components not available; using local fallback.")

        # Prefer Ollama if available
        if have_langchain and model_type != "huggingface":
            try:
                import requests  # type: ignore

                try:
                    requests.get("http://localhost:11434", timeout=1)
                    try:
                        self.llm = self._Ollama(model="llama3")
                        self.backend_name = "ollama"
                        print("[ai_engine_fixed] using Ollama backend")
                    except Exception as e:
                        print("[ai_engine_fixed] Ollama client init failed:", e)
                except Exception:
                    print("[ai_engine_fixed] Ollama daemon not reachable at http://localhost:11434")
            except Exception:
                print("[ai_engine_fixed] requests not available; skipping Ollama check")

        # Try HuggingFaceHub if token present
        if have_langchain and self.backend_name == "local-fallback":
            if os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
                try:
                    self.llm = self._HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct", model_kwargs={"temperature": 0.7})
                    self.backend_name = "huggingface"
                    print("[ai_engine_fixed] using HuggingFaceHub backend")
                except Exception as e:
                    print("[ai_engine_fixed] HuggingFaceHub init failed:", e)

        # Wire chain if we have a real LLM client
        if hasattr(self, "llm") and have_langchain:
            try:
                self.template = self._PromptTemplate(
                    input_variables=["post", "tone"],
                    template=(
                        "You are a professional LinkedIn expert. Read the following post and write a {tone} "
                        "comment in under 2 sentences that adds genuine insight and value. Avoid generic praise.\n\n"
                        "Post:\n{post}\n\nComment:"
                    ),
                )
                self.chain = self._LLMChain(prompt=self.template, llm=self.llm)
            except Exception as e:
                print("[ai_engine_fixed] failed to wire LLMChain, falling back:", e)
                self._setup_local_fallback()
        else:
            self._setup_local_fallback()

    def _setup_local_fallback(self) -> None:
        self.backend_name = "local-fallback"
        self._fallback_templates = {
            "Professional": [
                "Sharp insight — particularly around {keyword}. Thanks for sharing.",
                "Great point on {keyword}. I think this also highlights {takeaway}.",
            ],
            "Friendly": [
                "Nice share! I especially liked the bit about {keyword} — very relatable.",
                "Loved this — the {keyword} angle is spot on. Thanks for posting!",
            ],
            "Expert": [
                "Excellent analysis on {keyword}. For teams, I'd consider {takeaway} to complement this.",
                "Thought-provoking — {keyword} ties directly to {takeaway}, which often gets overlooked.",
            ],
        }

    def _extract_keyword(self, text: str) -> str:
        if not text:
            return "this"
        first = text.split(".\n")[0].split(".")[0]
        words = [w.strip(".,!?()[]") for w in first.split() if len(w) > 3]
        if not words:
            words = first.split()
        return words[0] if words else "this"

    def _make_takeaway(self, keyword: str) -> str:
        choices = [f"scaling {keyword}", f"measuring {keyword}", f"aligning teams around {keyword}", f"balancing speed and quality in {keyword}"]
        return random.choice(choices)

    def generate(self, post_text: str, tone: str = "Professional") -> str:
        """Return a short comment using chain if available else fallback."""
        if hasattr(self, "chain"):
            try:
                return self.chain.run({"post": post_text, "tone": tone}).strip()
            except Exception as e:
                print("[ai_engine_fixed] LLM chain failed, falling back to local generator:", e)

        keyword = self._extract_keyword(post_text)
        takeaway = self._make_takeaway(keyword)
        templates = self._fallback_templates.get(tone, self._fallback_templates["Professional"])
        template = random.choice(templates)
        comment = template.format(keyword=keyword, takeaway=takeaway)
        if len(comment) > 240:
            comment = comment[:237].rsplit(" ", 1)[0] + "..."
        return comment
