"""AI comment generator (single authoritative file).

Ordering: Ollama (local) -> OpenAI Chat -> deterministic templates.
This module is defensive so it runs even without optional packages.
"""
from __future__ import annotations

import os
import random

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class AICommentGenerator:
    def __init__(self, prefer_model: str = "ollama") -> None:
        self.backend_name = "local-fallback"
        self.llm = None
        self.chain = None
        self._setup_local_fallback()

        if prefer_model == "ollama":
            self._try_init_ollama()
            if self.llm is None and os.environ.get("OPENAI_API_KEY"):
                self._try_init_openai()
        else:
            if os.environ.get("OPENAI_API_KEY"):
                self._try_init_openai()
            if self.llm is None:
                self._try_init_ollama()

        print(f"[ai_engine] ready with backend={self.backend_name}")

    def _try_init_ollama(self) -> None:
        try:
            import requests

            requests.get(DEFAULT_OLLAMA_URL, timeout=1)
        except Exception:
            print("[ai_engine] Ollama daemon not reachable")
            return

        try:
            try:
                from langchain_community.llms import Ollama as LC_Ollama  # type: ignore
            except Exception:
                from langchain.llms import Ollama as LC_Ollama  # type: ignore

            from langchain.prompts import PromptTemplate  # type: ignore
            from langchain.chains import LLMChain  # type: ignore

            # instantiate a model you pulled into Ollama locally
            self.llm = LC_Ollama(model="llama3")
            self.template = PromptTemplate(
                input_variables=["post", "tone"],
                template=(
                    "You are a professional LinkedIn expert. Read the following post "
                    "and write a {tone} comment in under 2 sentences that adds genuine "
                    "insight and value. Avoid generic praise.\n\nPost:\n{post}\n\nComment:"
                ),
            )
            self.chain = LLMChain(prompt=self.template, llm=self.llm)
            self.backend_name = "ollama"
            print("[ai_engine] Ollama initialized via LangChain")
        except Exception as e:
            print("[ai_engine] Ollama init failed:", e)

    def _try_init_openai(self) -> None:
        try:
            from langchain.chat_models import ChatOpenAI  # type: ignore
            from langchain.prompts import PromptTemplate  # type: ignore
            from langchain.chains import LLMChain  # type: ignore

            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            self.template = PromptTemplate(
                input_variables=["post", "tone"],
                template=(
                    "You are a professional LinkedIn expert. Read the following post "
                    "and write a {tone} comment in under 2 sentences that adds genuine "
                    "insight and value. Avoid generic praise.\n\nPost:\n{post}\n\nComment:"
                ),
            )
            self.chain = LLMChain(prompt=self.template, llm=self.llm)
            self.backend_name = "openai"
            print("[ai_engine] OpenAI Chat initialized via LangChain")
        except Exception as e:
            print("[ai_engine] OpenAI init failed:", e)

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
        choices = [
            f"scaling {keyword}",
            f"measuring {keyword}",
            f"aligning teams around {keyword}",
            f"balancing speed and quality in {keyword}",
        ]
        return random.choice(choices)

    def generate(self, post_text: str, tone: str = "Professional") -> str:
        if getattr(self, "chain", None) is not None and getattr(self, "llm", None) is not None:
            try:
                out = self.chain.run({"post": post_text, "tone": tone})
                if isinstance(out, str):
                    return out.strip()
            except Exception as e:
                print("[ai_engine] LLM chain generation failed, falling back:", e)

        keyword = self._extract_keyword(post_text)
        takeaway = self._make_takeaway(keyword)
        template = random.choice(self._fallback_templates.get(tone, self._fallback_templates["Professional"]))
        comment = template.format(keyword=keyword, takeaway=takeaway)
        if len(comment) > 240:
            comment = comment[:237].rsplit(" ", 1)[0] + "..."
        return comment


if __name__ == "__main__":
    gen = AICommentGenerator()
    print("backend:", gen.backend_name)
    print("comment:", gen.generate("Quick post about improving onboarding flows for SMBs", tone="Professional"))
"""AI comment generator.

Single-file implementation: prefer local Ollama (if reachable), else
fall back to OpenAI Chat (if OPENAI_API_KEY is set), else deterministic
template responses. This file is meant to be the sole AI engine used by
the app (remove duplicates like `ai_engine_fixed.py`).
"""
from __future__ import annotations

import os
import random
import typing

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class AICommentGenerator:
    def __init__(self, prefer_model: str = "ollama") -> None:
        """Initialize generator.

        prefer_model: 'ollama' or 'openai' — preference only, availability
        determines actual backend.
        """
        self.backend_name: str = "local-fallback"
        self.llm = None
        self.chain = None
        self._setup_local_fallback()

        # Try Ollama first if preferred
        if prefer_model == "ollama":
            self._try_init_ollama()
            if self.llm is None and os.environ.get("OPENAI_API_KEY"):
                self._try_init_openai()
        else:
            if os.environ.get("OPENAI_API_KEY"):
                self._try_init_openai()
            if self.llm is None:
                self._try_init_ollama()

        print(f"[ai_engine] ready with backend={self.backend_name}")

    def _try_init_ollama(self) -> None:
        """Attempt to connect to a local Ollama daemon and wire a LangChain chain.

        Supports multiple LangChain wrappers if present.
        """
        try:
            import requests  # standard optional dependency

            requests.get(DEFAULT_OLLAMA_URL, timeout=1)
        except Exception:
            # Ollama not reachable — leave fallback in place
            print("[ai_engine] Ollama not reachable at http://localhost:11434")
            return

        try:
            # Try community package first, then main package
            try:
                from langchain_community.llms import Ollama as LC_Ollama  # type: ignore
            except Exception:
                from langchain.llms import Ollama as LC_Ollama  # type: ignore

            from langchain.prompts import PromptTemplate  # type: ignore
            from langchain.chains import LLMChain  # type: ignore

            # Use a local model name that the user will have pulled to Ollama
            self.llm = LC_Ollama(model="llama3")
            self.template = PromptTemplate(
                input_variables=["post", "tone"],
                template=(
                    "You are a professional LinkedIn expert. Read the following post "
                    "and write a {tone} comment in under 2 sentences that adds genuine "
                    "insight and value. Avoid generic praise.\n\nPost:\n{post}\n\nComment:"
                ),
            )
            self.chain = LLMChain(prompt=self.template, llm=self.llm)
            self.backend_name = "ollama"
            print("[ai_engine] Initialized Ollama via LangChain")
        except Exception as e:
            print("[ai_engine] Ollama init failed (LangChain or model):", e)

    def _try_init_openai(self) -> None:
        """Attempt to initialize ChatOpenAI via LangChain (requires OPENAI_API_KEY)."""
        try:
            from langchain.chat_models import ChatOpenAI  # type: ignore
            from langchain.prompts import PromptTemplate  # type: ignore
            from langchain.chains import LLMChain  # type: ignore

            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            self.template = PromptTemplate(
                input_variables=["post", "tone"],
                template=(
                    "You are a professional LinkedIn expert. Read the following post "
                    "and write a {tone} comment in under 2 sentences that adds genuine "
                    "insight and value. Avoid generic praise.\n\nPost:\n{post}\n\nComment:"
                ),
            )
            self.chain = LLMChain(prompt=self.template, llm=self.llm)
            self.backend_name = "openai"
            print("[ai_engine] Initialized OpenAI Chat via LangChain")
        except Exception as e:
            print("[ai_engine] OpenAI init failed (LangChain or key missing):", e)

    # ----------------
    # Local deterministic fallback
    # ----------------
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
        choices = [
            f"scaling {keyword}",
            f"measuring {keyword}",
            f"aligning teams around {keyword}",
            f"balancing speed and quality in {keyword}",
        ]
        return random.choice(choices)

    # ----------------
    # Public
    # ----------------
    def generate(self, post_text: str, tone: str = "Professional") -> str:
        """Return a short comment (LLM chain -> deterministic fallback)."""
        if getattr(self, "chain", None) is not None and getattr(self, "llm", None) is not None:
            try:
                out = self.chain.run({"post": post_text, "tone": tone})
                if isinstance(out, str):
                    return out.strip()
            except Exception as e:
                print("[ai_engine] LLM chain generation failed, falling back:", e)

        keyword = self._extract_keyword(post_text)
        takeaway = self._make_takeaway(keyword)
        template = random.choice(self._fallback_templates.get(tone, self._fallback_templates["Professional"]))
        comment = template.format(keyword=keyword, takeaway=takeaway)
        if len(comment) > 240:
            comment = comment[:237].rsplit(" ", 1)[0] + "..."
        return comment


if __name__ == "__main__":
    g = AICommentGenerator()
    print("backend:", g.backend_name)
    print("comment:", g.generate("Quick post about improving onboarding flows for SMBs", tone="Professional"))

"""AI comment generator.

Behavior:
- Prefer a local Ollama daemon (if reachable at http://localhost:11434) using
  LangChain's Ollama wrapper when available.
- If Ollama isn't reachable and OPENAI_API_KEY is present, initialize
  LangChain's ChatOpenAI (gpt-3.5-turbo) as a fallback.
- If no LLM is available, use deterministic template-based comments.

This module is defensive: it never leaves `self.llm` undefined and it
keeps a simple template fallback so the app works offline.
"""
from __future__ import annotations

import os
import random
from typing import Optional

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class AICommentGenerator:
    def __init__(self, prefer_model: str = "ollama") -> None:
        """Create the generator.

        prefer_model: 'ollama' or 'openai' - only a preference; actual backend
        will be chosen based on availability and environment variables.
        """
        self.backend_name: str = "local-fallback"
        self.llm = None  # type: ignore
        self.chain = None

        # Prepare always-available fallback templates first
        self._setup_local_fallback()

        # Try Ollama first if preferred
        tried_ollama = False
        if prefer_model == "ollama" or prefer_model is None:
            tried_ollama = self._try_init_ollama()

        # If Ollama not initialized, and an OpenAI key exists, try OpenAI
        if self.llm is None and os.environ.get("OPENAI_API_KEY"):
            self._try_init_openai()

        # If user preferred OpenAI first, allow that ordering
        if prefer_model == "openai" and self.llm is None:
            # If OpenAI wasn't tried above, try it now
            if os.environ.get("OPENAI_API_KEY"):
                self._try_init_openai()
            # Finally, try Ollama if still none
            if self.llm is None and not tried_ollama:
                self._try_init_ollama()

        print(f"[ai_engine] Ready with backend: {self.backend_name}")

    # ----------------
    # Backend initializers
    # ----------------
    def _try_init_ollama(self) -> bool:
        """Attempt to initialize an Ollama client (langchain wrapper).

        Returns True if an attempt was made (regardless of success).
        """
        try:
            import requests

            # quick availability check
            resp = requests.get(DEFAULT_OLLAMA_URL, timeout=1)
            if resp.status_code >= 400:
                print("[ai_engine] Ollama responded but returned error code; skipping")
                return True
        except Exception:
            print(f"[ai_engine] Ollama daemon not reachable at {DEFAULT_OLLAMA_URL}")
            return True

        # If we got here, Ollama seems reachable; try to use LangChain's wrapper
        try:
            try:
                # community package name used in some envs
                from langchain_community.llms import Ollama  # type: ignore
            except Exception:
                # newer langchain variants may expose different names
                from langchain.llms import Ollama  # type: ignore

            from langchain.prompts import PromptTemplate  # type: ignore
            from langchain.chains import LLMChain  # type: ignore

            # Use a sensible default; users can change models locally via Ollama
            self.llm = Ollama(model="llama2")
            self.template = PromptTemplate(
                input_variables=["post", "tone"],
                template=(
                    "You are a professional LinkedIn expert. Read the following post "
                    "and write a {tone} comment in under 2 sentences that adds genuine "
                    "insight and value. Avoid generic praise.\n\nPost:\n{post}\n\nComment:"
                ),
            )
            self.chain = LLMChain(prompt=self.template, llm=self.llm)
            self.backend_name = "ollama"
            print("[ai_engine] Connected to Ollama via LangChain")
            return True
        except Exception as e:
            print(f"[ai_engine] Ollama init failed (LangChain missing or error): {e}")
            # Leave llm as None so fallback is used
            return True

    def _try_init_openai(self) -> bool:
        """Attempt to initialize LangChain ChatOpenAI (requires OPENAI_API_KEY).

        Returns True if attempted.
        """
        try:
            from langchain.chat_models import ChatOpenAI  # type: ignore
            from langchain.prompts import PromptTemplate  # type: ignore
            from langchain.chains import LLMChain  # type: ignore

            # Use deterministic settings by default
            self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            self.template = PromptTemplate(
                input_variables=["post", "tone"],
                template=(
                    "You are a professional LinkedIn expert. Read the following post "
                    "and write a {tone} comment in under 2 sentences that adds genuine "
                    "insight and value. Avoid generic praise.\n\nPost:\n{post}\n\nComment:"
                ),
            )
            self.chain = LLMChain(prompt=self.template, llm=self.llm)
            self.backend_name = "openai"
            print("[ai_engine] Connected to OpenAI Chat model via LangChain")
            return True
        except Exception as e:
            print(f"[ai_engine] OpenAI Chat initialization failed: {e}")
            return True

    # ----------------
    # Fallback templates + helpers
    # ----------------
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
        choices = [
            f"scaling {keyword}",
            f"measuring {keyword}",
            f"aligning teams around {keyword}",
            f"balancing speed and quality in {keyword}",
        ]
        return random.choice(choices)

    # ----------------
    # Public generation API
    # ----------------
    def generate(self, post_text: str, tone: str = "Professional") -> str:
        """Generate a short comment.

        Priority: chain.run (if chain and llm present) -> template fallback.
        """
        # If we have a chain wired up, prefer using it
        if getattr(self, "chain", None) is not None and getattr(self, "llm", None) is not None:
            try:
                # LangChain's LLMChain.run may exist; guard with getattr
                result = self.chain.run({"post": post_text, "tone": tone})
                if isinstance(result, str):
                    return result.strip()
            except Exception as e:
                print(f"[ai_engine] LLM chain generation failed, falling back to templates: {e}")

        # Template fallback
        keyword = self._extract_keyword(post_text)
        takeaway = self._make_takeaway(keyword)
        templates = self._fallback_templates.get(tone, self._fallback_templates["Professional"])
        template = random.choice(templates)
        comment = template.format(keyword=keyword, takeaway=takeaway)
        if len(comment) > 240:
            comment = comment[:237].rsplit(" ", 1)[0] + "..."
        return comment


if __name__ == "__main__":
    # Quick manual smoke run when executed directly
    gen = AICommentGenerator()
    print("backend:", gen.backend_name)
    print("comment:", gen.generate("This post is about scaling machine learning teams.", "Professional"))
"""AI comment generator.

Behavior:
- Prefer a local Ollama daemon (if reachable at http://localhost:11434) using
  LangChain's Ollama wrapper when available.
- If Ollama isn't reachable and OPENAI_API_KEY is present, initialize
  LangChain's ChatOpenAI (gpt-3.5-turbo) as a fallback.
- If no LLM is available, use deterministic template-based comments.

This module is defensive: it never leaves `self.llm` undefined and it
keeps a simple template fallback so the app works offline.
"""
from __future__ import annotations

import os
import random
from typing import Optional

DEFAULT_OLLAMA_URL = "http://localhost:11434"


class AICommentGenerator:
    def __init__(self, prefer_model: str = "ollama") -> None:
        """Create the generator.

        prefer_model: 'ollama' or 'openai' - only a preference; actual backend
        will be chosen based on availability and environment variables.
        """
        self.backend_name: str = "local-fallback"
        self.llm = None  # type: ignore
        self.chain = None

        # Prepare always-available fallback templates first
        self._setup_local_fallback()

        # Try Ollama first if preferred
        tried_ollama = False
        if prefer_model == "ollama" or prefer_model is None:
            tried_ollama = self._try_init_ollama()

        # If Ollama not initialized, and an OpenAI key exists, try OpenAI
        if self.llm is None and os.environ.get("OPENAI_API_KEY"):
            self._try_init_openai()

        # If user preferred OpenAI first, allow that ordering
        if prefer_model == "openai" and self.llm is None:
            # If OpenAI wasn't tried above, try it now
            if os.environ.get("OPENAI_API_KEY"):
                self._try_init_openai()
            # Finally, try Ollama if still none
            if self.llm is None and not tried_ollama:
                self._try_init_ollama()

        print(f"[ai_engine] Ready with backend: {self.backend_name}")

    # ----------------
    # Backend initializers
    # ----------------
    def _try_init_ollama(self) -> bool:
        """Attempt to initialize an Ollama client (langchain wrapper).

        Returns True if an attempt was made (regardless of success).
        """
        try:
            import requests

            # quick availability check
            resp = requests.get(DEFAULT_OLLAMA_URL, timeout=1)
            if resp.status_code >= 400:
                print("[ai_engine] Ollama responded but returned error code; skipping")
                return True
        except Exception:
            print(f"[ai_engine] Ollama daemon not reachable at {DEFAULT_OLLAMA_URL}")
            return True

        # If we got here, Ollama seems reachable; try to use LangChain's wrapper
        try:
            try:
                # community package name used in some envs
                from langchain_community.llms import Ollama  # type: ignore
            except Exception:
                # newer langchain variants may expose different names
                from langchain.llms import Ollama  # type: ignore

            from langchain.prompts import PromptTemplate  # type: ignore
            from langchain.chains import LLMChain  # type: ignore

            # Use a sensible default; users can change models locally via Ollama
            self.llm = Ollama(model="llama2")
            self.template = PromptTemplate(
                input_variables=["post", "tone"],
                template=(
                    "You are a professional LinkedIn expert. Read the following post "
                    "and write a {tone} comment in under 2 sentences that adds genuine "
                    "insight and value. Avoid generic praise.\n\nPost:\n{post}\n\nComment:"
                ),
            )
            self.chain = LLMChain(prompt=self.template, llm=self.llm)
            self.backend_name = "ollama"
            print("[ai_engine] Connected to Ollama via LangChain")
            return True
        except Exception as e:
            print(f"[ai_engine] Ollama init failed (LangChain missing or error): {e}")
            # Leave llm as None so fallback is used
            return True

    def _try_init_openai(self) -> bool:
        """Attempt to initialize LangChain ChatOpenAI (requires OPENAI_API_KEY).

        Returns True if attempted.


                        # Wire up the chain since we have Ollama            from langchain.prompts import PromptTemplate  # type: ignoreclass AICommentGenerator:

                        self.template = self._PromptTemplate(

                            input_variables=["post", "tone"],            from langchain.chains import LLMChain  # type: ignore

                            template=(

                                "You are a professional LinkedIn expert. Read the following post "class AICommentGenerator:    def __init__(self, model_type: str = "ollama"):

                                "and write a {tone} comment in under 2 sentences that adds genuine "

                                "insight and value. Avoid generic praise.\n\n"            self._Ollama = Ollama

                                "Post:\n{post}\n\nComment:"

                            ),            self._HuggingFaceHub = HuggingFaceHub    def __init__(self, model_type: str = "ollama"):        # Try imports for langchain-backed clients. If they fail we'll use the

                        )

                        self.chain = self._LLMChain(prompt=self.template, llm=self.llm)            self._PromptTemplate = PromptTemplate

                    except Exception as e:

                        print(f"[ai_engine] Ollama init failed - using templates: {e}")            self._LLMChain = LLMChain        self.backend_name = "local-fallback"        # local fallback implementation below.

                        self._reset_to_fallback()

                except Exception:            have_langchain = True

                    print("[ai_engine] Ollama not running - using templates")

                    self._reset_to_fallback()        except Exception:        have_langchain = False        have_langchain = True

            except Exception:

                print("[ai_engine] Could not check Ollama - using templates")            have_langchain = False

                self._reset_to_fallback()

            print("[ai_engine] LangChain components not available - will use simple fallback templates.")        try:

    def _reset_to_fallback(self) -> None:

        """Reset state to use fallback templates."""

        self.llm = None

        self.backend_name = "local-fallback"        # Initialize the fallback templates first, so we always have them ready        # Try lazy imports for langchain-based wrappers. If they exist we'll            from langchain_community.llms import Ollama, HuggingFaceHub  # type: ignore



    def _setup_local_fallback(self) -> None:        self._setup_local_fallback()

        """Initialize the local template-based fallback generator."""

        self._fallback_templates = {        # attempt to initialize a real LLM client; otherwise we'll keep the            from langchain.prompts import PromptTemplate  # type: ignore

            "Professional": [

                "Sharp insight — particularly around {keyword}. Thanks for sharing.",        # If langchain helpers are available, prefer Ollama (local daemon) first

                "Great point on {keyword}. I think this also highlights {takeaway}.",

            ],        if have_langchain and model_type != "huggingface":        # deterministic local fallback.            from langchain.chains import LLMChain  # type: ignore

            "Friendly": [

                "Nice share! I especially liked the bit about {keyword} — very relatable.",            try:

                "Loved this — the {keyword} angle is spot on. Thanks for posting!",

            ],                import requests  # type: ignore        try:

            "Expert": [

                "Excellent analysis on {keyword}. For teams, I'd consider {takeaway} to complement this.",                print("[ai_engine] Checking Ollama availability...")

                "Thought-provoking — {keyword} ties directly to {takeaway}, which often gets overlooked.",

            ],                try:            from langchain_community.llms import Ollama, HuggingFaceHub  # type: ignore            self._Ollama = Ollama

        }

                    requests.get("http://localhost:11434", timeout=1)

    def _extract_keyword(self, text: str) -> str:

        """Extract a meaningful keyword from the post text."""                    # Try to initialize Ollama client            from langchain.prompts import PromptTemplate  # type: ignore            self._HuggingFaceHub = HuggingFaceHub

        if not text:

            return "this"                    try:

        first = text.split(".\n")[0].split(".")[0]

        words = [w.strip(".,!?()[]") for w in first.split() if len(w) > 3]                        self.llm = self._Ollama(model="llama3")            from langchain.chains import LLMChain  # type: ignore            self._PromptTemplate = PromptTemplate

        if not words:

            words = first.split()                        self.backend_name = "ollama"

        return words[0] if words else "this"

                        print("[ai_engine] Successfully connected to Ollama")            self._LLMChain = LLMChain

    def _make_takeaway(self, keyword: str) -> str:

        """Generate a relevant insight/takeaway from the keyword."""                    except Exception as e:

        choices = [

            f"scaling {keyword}",                        print(f"[ai_engine] Ollama client init failed (using fallback): {e}")            self._Ollama = Ollama        """AI comment generator wrapper.

            f"measuring {keyword}",

            f"aligning teams around {keyword}",                except Exception:

            f"balancing speed and quality in {keyword}",

        ]                    print("[ai_engine] Ollama not running at http://localhost:11434 (using fallback)")            self._HuggingFaceHub = HuggingFaceHub

        return random.choice(choices)

            except Exception:

    def generate(self, post_text: str, tone: str = "Professional") -> str:

        """Generate a comment using Ollama if available, else use templates."""                print("[ai_engine] Could not check Ollama (using fallback)")            self._PromptTemplate = PromptTemplate        This module attempts lazy imports of optional LangChain helpers and prefers a

        # Try Ollama chain first if we have it

        if hasattr(self, "chain") and self.llm is not None:

            try:

                return self.chain.run({"post": post_text, "tone": tone}).strip()        # If still no backend and HF token present, try HuggingFaceHub            self._LLMChain = LLMChain        local Ollama daemon, falls back to HuggingFaceHub when a token is configured,

            except Exception as e:

                print(f"[ai_engine] Ollama generation failed - using templates: {e}")        if have_langchain and self.backend_name == "local-fallback":



        # Local template fallback (always available)            token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")            have_langchain = True        and otherwise uses a deterministic local generator so the app remains usable

        keyword = self._extract_keyword(post_text)

        takeaway = self._make_takeaway(keyword)            if token:

        templates = self._fallback_templates.get(tone, self._fallback_templates["Professional"])

        template = random.choice(templates)                try:        except Exception:        without external LLMs.

        comment = template.format(keyword=keyword, takeaway=takeaway)

        if len(comment) > 240:                    self.llm = self._HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct", model_kwargs={"temperature": 0.7})

            comment = comment[:237].rsplit(" ", 1)[0] + "..."

        return comment                    self.backend_name = "huggingface"            have_langchain = False        """

                    print("[ai_engine] Successfully connected to HuggingFaceHub")

                except Exception as e:            # We intentionally don't surface the full exception here; the app

                    print(f"[ai_engine] HuggingFaceHub init failed (using fallback): {e}")

            # should still work with the fallback generator.        import os

        # If we have a real LLM client, wire up the chain

        if self.llm is not None and have_langchain:            print("[ai_engine] langchain components not available; using local fallback.")        import random

            try:

                self.template = self._PromptTemplate(        from typing import Optional

                    input_variables=["post", "tone"],

                    template=(        # If langchain helpers are available, prefer Ollama (local daemon) first

                        "You are a professional LinkedIn expert. Read the following post and write a {tone} "

                        "comment in under 2 sentences that adds genuine insight and value. Avoid generic praise.\n\n"        if have_langchain and model_type != "huggingface":

                        "Post:\n{post}\n\nComment:"

                    ),            try:        class AICommentGenerator:

                )

                self.chain = self._LLMChain(prompt=self.template, llm=self.llm)                import requests  # type: ignore            def __init__(self, model_type: str = "ollama"):

                print(f"[ai_engine] Ready with {self.backend_name} backend")

            except Exception as e:                self.backend_name = "local-fallback"

                print(f"[ai_engine] Could not create LLM chain (using fallback): {e}")

                self._setup_local_fallback()                try:                have_langchain = False

                self.llm = None  # Reset llm since chain init failed

                    requests.get("http://localhost:11434", timeout=1)

    def _setup_local_fallback(self) -> None:

        """Initialize the local template-based fallback generator."""                    # Try to initialize Ollama client                # Try lazy imports for langchain-based wrappers. If they exist we'll

        self.backend_name = "local-fallback"

        self._fallback_templates = {                    try:                # attempt to initialize a real LLM client; otherwise we'll keep the

            "Professional": [

                "Sharp insight — particularly around {keyword}. Thanks for sharing.",                        self.llm = self._Ollama(model="llama3")                # deterministic local fallback.

                "Great point on {keyword}. I think this also highlights {takeaway}.",

            ],                        self.backend_name = "ollama"                try:

            "Friendly": [

                "Nice share! I especially liked the bit about {keyword} — very relatable.",                        print("[ai_engine] using Ollama backend")                    from langchain_community.llms import Ollama, HuggingFaceHub  # type: ignore

                "Loved this — the {keyword} angle is spot on. Thanks for posting!",

            ],                    except Exception as e:                    from langchain.prompts import PromptTemplate  # type: ignore

            "Expert": [

                "Excellent analysis on {keyword}. For teams, I'd consider {takeaway} to complement this.",                        print("[ai_engine] Ollama client init failed:", e)                    from langchain.chains import LLMChain  # type: ignore

                "Thought-provoking — {keyword} ties directly to {takeaway}, which often gets overlooked.",

            ],                except Exception:

        }

                    print("[ai_engine] Ollama daemon not reachable at http://localhost:11434")                    self._Ollama = Ollama

    def _extract_keyword(self, text: str) -> str:

        """Extract a meaningful keyword from the post text."""            except Exception:                    self._HuggingFaceHub = HuggingFaceHub

        if not text:

            return "this"                print("[ai_engine] requests not available; skipping Ollama check")                    self._PromptTemplate = PromptTemplate

        first = text.split(".\n")[0].split(".")[0]

        words = [w.strip(".,!?()[]") for w in first.split() if len(w) > 3]                    self._LLMChain = LLMChain

        if not words:

            words = first.split()        # If still no backend and HF token present, try HuggingFaceHub                    have_langchain = True

        return words[0] if words else "this"

        if have_langchain and self.backend_name == "local-fallback":                except Exception:

    def _make_takeaway(self, keyword: str) -> str:

        """Generate a relevant insight/takeaway from the keyword."""            if os.environ.get("HUGGINGFACEHUB_API_TOKEN"):                    have_langchain = False

        choices = [f"scaling {keyword}", f"measuring {keyword}", f"aligning teams around {keyword}", f"balancing speed and quality in {keyword}"]

        return random.choice(choices)                try:                    # We intentionally don't surface the full exception here; the app



    def generate(self, post_text: str, tone: str = "Professional") -> str:                    self.llm = self._HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct", model_kwargs={"temperature": 0.7})                    # should still work with the fallback generator.

        """Generate a short comment using LLM chain if available, else use local templates."""

        # Try LLM chain first (if we have one)                    self.backend_name = "huggingface"                    print("[ai_engine] langchain components not available; using local fallback.")

        if hasattr(self, "chain") and self.llm is not None:

            try:                    print("[ai_engine] using HuggingFaceHub backend")

                return self.chain.run({"post": post_text, "tone": tone}).strip()

            except Exception as e:                except Exception as e:                # If langchain helpers are available, prefer Ollama (local daemon) first

                print("[ai_engine] LLM chain failed (using fallback):", e)

                    print("[ai_engine] HuggingFaceHub init failed:", e)                if have_langchain and model_type != "huggingface":

        # Local template fallback (always available)

        keyword = self._extract_keyword(post_text)                    try:

        takeaway = self._make_takeaway(keyword)

        templates = self._fallback_templates.get(tone, self._fallback_templates["Professional"])        # If we have a langchain-backed llm, wire up a simple LLMChain; otherwise                        import requests  # type: ignore

        template = random.choice(templates)

        comment = template.format(keyword=keyword, takeaway=takeaway)        # prepare the deterministic fallback templates.

        if len(comment) > 240:

            comment = comment[:237].rsplit(" ", 1)[0] + "..."        if hasattr(self, "llm") and have_langchain:                        try:

        return comment
            try:                            requests.get("http://localhost:11434", timeout=1)

                self.template = self._PromptTemplate(                            # Try to initialize Ollama client

                    input_variables=["post", "tone"],                            try:

                    template=(                                self.llm = self._Ollama(model="llama3")

                        "You are a professional LinkedIn expert. Read the following post and write a {tone} "                                self.backend_name = "ollama"

                        "comment in under 2 sentences that adds genuine insight and value. Avoid generic praise.\n\n"                                print("[ai_engine] using Ollama backend")

                        "Post:\n{post}\n\nComment:"                            except Exception as e:

                    ),                                print("[ai_engine] Ollama client init failed:", e)

                )                        except Exception:

                self.chain = self._LLMChain(prompt=self.template, llm=self.llm)                            print("[ai_engine] Ollama daemon not reachable at http://localhost:11434")

            except Exception as e:                    except Exception:

                print("[ai_engine] failed to wire LLMChain, falling back:", e)                        print("[ai_engine] requests not available; skipping Ollama check")

                self._setup_local_fallback()

        else:                # If still no backend and HF token present, try HuggingFaceHub

            self._setup_local_fallback()                if have_langchain and self.backend_name == "local-fallback":

                    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")

    def _setup_local_fallback(self) -> None:                    if hf_token:

        self.backend_name = "local-fallback"                        try:

        self._fallback_templates = {                            self.llm = self._HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct", model_kwargs={"temperature": 0.7})

            "Professional": [                            self.backend_name = "huggingface"

                "Sharp insight — particularly around {keyword}. Thanks for sharing.",                            print("[ai_engine] using HuggingFaceHub backend")

                "Great point on {keyword}. I think this also highlights {takeaway}.",                        except Exception as e:

            ],                            print("[ai_engine] HuggingFaceHub init failed:", e)

            "Friendly": [

                "Nice share! I especially liked the bit about {keyword} — very relatable.",                # If we have a langchain-backed llm, wire up a simple LLMChain; otherwise

                "Loved this — the {keyword} angle is spot on. Thanks for posting!",                # prepare the deterministic fallback templates.

            ],                if hasattr(self, "llm") and have_langchain:

            "Expert": [                    try:

                "Excellent analysis on {keyword}. For teams, I'd consider {takeaway} to complement this.",                        self.template = self._PromptTemplate(

                "Thought-provoking — {keyword} ties directly to {takeaway}, which often gets overlooked.",                            input_variables=["post", "tone"],

            ],                            template=(

        }                                "You are a professional LinkedIn expert. Read the following post and write a {tone} "

                                "comment in under 2 sentences that adds genuine insight and value. Avoid generic praise.\n\n"

    def _extract_keyword(self, text: str) -> str:                                "Post:\n{post}\n\nComment:"

        if not text:                            ),

            return "this"                        )

        first = text.split(".\n")[0].split(".")[0]                        self.chain = self._LLMChain(prompt=self.template, llm=self.llm)

        words = [w.strip(".,!?()[]") for w in first.split() if len(w) > 3]                    except Exception as e:

        if not words:                        print("[ai_engine] failed to wire LLMChain, falling back:", e)

            words = first.split()                        self._setup_local_fallback()

        return words[0] if words else "this"                else:

                    self._setup_local_fallback()

    def _make_takeaway(self, keyword: str) -> str:

        choices = [f"scaling {keyword}", f"measuring {keyword}", f"aligning teams around {keyword}", f"balancing speed and quality in {keyword}"]            def _setup_local_fallback(self) -> None:

        return random.choice(choices)                self.backend_name = "local-fallback"

                self._fallback_templates = {

    def generate(self, post_text: str, tone: str = "Professional") -> str:                    "Professional": [

        """Return a short comment using chain if available else fallback."""                        "Sharp insight — particularly around {keyword}. Thanks for sharing.",

        if hasattr(self, "chain"):                        "Great point on {keyword}. I think this also highlights {takeaway}.",

            try:                    ],

                return self.chain.run({"post": post_text, "tone": tone}).strip()                    "Friendly": [

            except Exception as e:                        "Nice share! I especially liked the bit about {keyword} — very relatable.",

                print("[ai_engine] LLM chain failed, falling back to local generator:", e)                        "Loved this — the {keyword} angle is spot on. Thanks for posting!",

                    ],

        keyword = self._extract_keyword(post_text)                    "Expert": [

        takeaway = self._make_takeaway(keyword)                        "Excellent analysis on {keyword}. For teams, I'd consider {takeaway} to complement this.",

        templates = self._fallback_templates.get(tone, self._fallback_templates["Professional"])                        "Thought-provoking — {keyword} ties directly to {takeaway}, which often gets overlooked.",

        template = random.choice(templates)                    ],

        comment = template.format(keyword=keyword, takeaway=takeaway)                }

        if len(comment) > 240:

            comment = comment[:237].rsplit(" ", 1)[0] + "..."            def _extract_keyword(self, text: str) -> str:

        return comment                if not text:
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
                # Use langchain chain if available
                if hasattr(self, "chain"):
                    try:
                        return self.chain.run({"post": post_text, "tone": tone}).strip()
                    except Exception as e:
                        print("[ai_engine] LLM chain failed, falling back to local generator:", e)

                # Local deterministic fallback
                keyword = self._extract_keyword(post_text)
                takeaway = self._make_takeaway(keyword)
                templates = self._fallback_templates.get(tone, self._fallback_templates["Professional"])
                template = random.choice(templates)
                comment = template.format(keyword=keyword, takeaway=takeaway)
                if len(comment) > 240:
                    comment = comment[:237].rsplit(" ", 1)[0] + "..."
                return comment
