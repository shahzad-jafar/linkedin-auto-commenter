
# from __future__ import annotations
# import json
# import datetime
# from pathlib import Path
# from typing import Optional


# class CommentDecisionAgent:
#     def __init__(self, ai_generator, niche: str):
#         self.ai = ai_generator
#         self.niche = niche.lower().strip()
#         self.decision_chain = None
#         self.use_invoke = True

#         try:
#             from langchain.prompts import PromptTemplate
#             from langchain_core.output_parsers import StrOutputParser
#             from langchain_core.runnables import RunnableSequence
#         except Exception as e:
#             raise ModuleNotFoundError(
#                 "Required package 'langchain' is missing. Install with:\n"
#                 "   pip install langchain langchain-ollama\n"
#                 f"Original error: {e}"
#             )

#         # 🧠 Strict yes/no decision prompt
#         self.decision_prompt = PromptTemplate.from_template(
#             """You are an AI assistant deciding whether to comment on a LinkedIn post.

# The user only comments if the post is relevant to the niche: "{niche}".

# Reply **strictly** with 'yes' or 'no'. No explanation.

# Post:
# {post}

# Decision:"""
#         )

#         # ✅ Build runnable chain if LLM is available
#         if self.ai and getattr(self.ai, "llm", None):
#             try:
#                 from langchain_core.runnables import RunnableSequence
#                 from langchain_core.output_parsers import StrOutputParser

#                 self.decision_chain = RunnableSequence(
#                     self.decision_prompt,
#                     self.ai.llm,
#                     StrOutputParser(),
#                 )
#                 self.use_invoke = True
#                 print("[decision_agent] ✅ RunnableSequence ready.")
#             except Exception as e:
#                 print("[decision_agent] ⚠️ RunnableSequence unavailable, fallback to LLMChain:", e)
#                 self._fallback_chain()
#         else:
#             print("[decision_agent] ⚠️ No LLM available, using fallback logic.")
#             self._fallback_chain()

#     # ------------------------------------------------------------------
#     def _fallback_chain(self):
#         """Fallback logic when no LLM or LangChain chain available."""
#         self.decision_chain = None
#         self.use_invoke = False

#     # ------------------------------------------------------------------
#     def should_comment(self, post_text: str) -> bool:
#         """Decide if the bot should comment on this post."""
#         post_text = (post_text or "").strip()
#         if not post_text:
#             return False

#         if not self.decision_chain:
#             # Local keyword-based fallback
#             keywords = ["linkedin", self.niche]
#             match = any(k.lower() in post_text.lower() for k in keywords)
#             print(f"[decision_agent] ⚙️ Fallback keyword check: {match}")
#             return match

#         try:
#             # Invoke modern LangChain pipeline
#             result = self.decision_chain.invoke({"post": post_text, "niche": self.niche})
#             decision = str(result).strip().lower()

#             # Normalize variations
#             clean_decision = (
#                 "yes" if any(word in decision for word in ["yes", "yeah", "yep", "sure"]) else
#                 "no"
#             )
#             print(f"[decision_agent] decision raw output: {decision} → parsed: {clean_decision}")

#             self._log_decision(post_text, clean_decision, decision)
#             return clean_decision == "yes"

#         except Exception as e:
#             print(f"⚠️ Decision LLM error: {e}")
#             self._log_decision(post_text, "error", str(e))
#             return False

#     # ------------------------------------------------------------------
#     def decide_and_generate(self, post_text: str, tone: str) -> Optional[str]:
#         """Combines decision + comment generation."""
#         if self.should_comment(post_text):
#             try:
#                 comment = self.ai.generate(post_text, tone)
#                 if comment:
#                     print(f"[decision_agent] 🗨️ Comment: {comment[:120]}...")
#                     self._log_generation(post_text, comment)
#                     return comment
#             except Exception as e:
#                 print("⚠️ LLM generation failed:", e)
#         print("⏭️ Skipped post (irrelevant).")
#         return None

#     # ------------------------------------------------------------------
#     def _log_decision(self, post_text: str, parsed: str, raw: str):
#         """Append decision log entry."""
#         try:
#             Path("data").mkdir(parents=True, exist_ok=True)
#             entry = {
#                 "timestamp": datetime.datetime.now().isoformat(),
#                 "type": "decision",
#                 "niche": self.niche,
#                 "parsed": parsed,
#                 "raw": raw,
#                 "post_snippet": post_text[:400],
#             }
#             with open("data/decisions.jsonl", "a", encoding="utf-8") as fh:
#                 fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
#         except Exception:
#             pass

#     def _log_generation(self, post_text: str, comment: str):
#         """Append comment generation entry."""
#         try:
#             entry = {
#                 "timestamp": datetime.datetime.now().isoformat(),
#                 "type": "generation",
#                 "niche": self.niche,
#                 "post_snippet": post_text[:400],
#                 "comment_preview": comment[:300],
#             }
#             with open("data/decisions.jsonl", "a", encoding="utf-8") as fh:
#                 fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
#         except Exception:
#             pass


# # ----------------------------------------------------------------------
# if __name__ == "__main__":
#     from ai_engine_clean import AICommentGenerator

#     ai = AICommentGenerator()
#     decider = CommentDecisionAgent(ai, "LinkedIn marketing")

#     post = "I just learned a new strategy for growing personal brands using authentic storytelling on LinkedIn."
#     print("Should comment?", decider.should_comment(post))
#     if decider.should_comment(post):
#         print("Comment:", decider.decide_and_generate(post, "Professional"))


# from typing import Optional
# import json
# import datetime
# from pathlib import Path


# class CommentDecisionAgent:
#     """
#     This agent decides if a LinkedIn post is relevant and, if so,
#     generates a short professional comment.
#     """

#     def __init__(self, ai_generator, niche: str):
#         self.ai = ai_generator
#         self.niche = niche.lower()

#         try:
#             from langchain.prompts import PromptTemplate
#         except Exception as e:
#             raise ModuleNotFoundError(
#                 "LangChain missing. Install it with:\n"
#                 "   pip install langchain langchain-community langchain-ollama\n"
#                 f"Original error: {e}"
#             )

#         # ✅ Smarter, balanced decision prompt
#         self.decision_prompt = PromptTemplate(
#             input_variables=["post", "niche"],
#             template=(
#                 "You are an assistant that helps decide if a LinkedIn post is relevant to a niche.\n"
#                 "Niche: {niche}\n"
#                 "Post:\n{post}\n\n"
#                 "If the post is even somewhat related or useful for someone in this niche, reply exactly with 'yes'.\n"
#                 "If it's totally unrelated, reply 'no'.\n"
#                 "Be inclusive — choose 'yes' for most business, tech, or marketing-related posts.\n"
#                 "Answer only 'yes' or 'no'."
#             ),
#         )

#         # ✅ Compatible with both new and old LangChain versions
#         try:
#             self.decision_chain = self.decision_prompt | self.ai.llm
#             self.use_invoke = True
#         except Exception:
#             from langchain.chains import LLMChain
#             self.decision_chain = LLMChain(prompt=self.decision_prompt, llm=self.ai.llm)
#             self.use_invoke = False

#     def should_comment(self, post_text: str) -> bool:
#         """Decide if this post deserves a comment."""
#         if not self.ai.llm:
#             print("⚠️ No LLM initialized; skipping.")
#             return False

#         try:
#             if self.use_invoke:
#                 result = self.decision_chain.invoke({"post": post_text, "niche": self.niche})
#                 decision_raw = str(result).strip().lower()
#             else:
#                 decision_raw = str(
#                     self.decision_chain.run({"post": post_text, "niche": self.niche})
#                 ).strip().lower()

#             print(f"[decision_agent] raw decision: {decision_raw}")

#             # ✅ Log decision
#             try:
#                 Path("data").mkdir(parents=True, exist_ok=True)
#                 with open("data/decisions.jsonl", "a", encoding="utf-8") as f:
#                     f.write(
#                         json.dumps(
#                             {
#                                 "timestamp": datetime.datetime.now().isoformat(),
#                                 "type": "decision",
#                                 "niche": self.niche,
#                                 "post_snippet": post_text[:500],
#                                 "decision_raw": decision_raw,
#                             },
#                             ensure_ascii=False,
#                         )
#                         + "\n"
#                     )
#             except Exception:
#                 pass

#             # ✅ Soft check
#             return "yes" in decision_raw
#         except Exception as e:
#             print("⚠️ Decision call failed:", e)
#             return False
#     def decide_and_generate(self, post_text: str, tone: str) -> Optional[str]:
#         """Generate a short, natural LinkedIn comment (max 2 lines, ~30 words)."""
#         if self.should_comment(post_text):
#             try:
#                 prompt = (
#                     f"Write a short LinkedIn comment (1–2 lines, max 30 words) "
#                     f"that sounds natural, {tone.lower()}, and human. "
#                     f"Focus on appreciation, insight, or relevance to the topic. "
#                     f"Do not start with phrases like 'Here's a comment', 'Great post', or 'I think'. "
#                     f"Make it sound authentic and contextually intelligent.\n\n"
#                     f"Post:\n{post_text}\n\n"
#                     "Write only the final comment, nothing else."
#                 )

#                 comment = self.ai.llm.invoke(prompt)
#                 if isinstance(comment, dict) and "text" in comment:
#                     comment = comment["text"]
#                 comment = str(comment).strip()

#                 # Clean up AI prefixes
#                 bad_starts = [
#                     "here's", "here is", "a possible", "comment:", "sure", "assistant:", "response:", "great post"
#                 ]
#                 for prefix in bad_starts:
#                     if comment.lower().startswith(prefix):
#                         comment = comment[len(prefix):].strip(" :-\n\t\"'")

#                 # Enforce 30-word limit
#                 words = comment.split()
#                 if len(words) > 30:
#                     comment = " ".join(words[:30]) + "..."

#                 comment = comment.replace("\n", " ").strip()
#                 print(f"[decision_agent] Final comment: {comment}")

#                 # Log
#                 try:
#                     entry = {
#                         "timestamp": datetime.datetime.now().isoformat(),
#                         "type": "generation",
#                         "post_snippet": post_text[:400],
#                         "comment_preview": comment,
#                         "niche": self.niche
#                     }
#                     with open("data/decisions.jsonl", "a", encoding="utf-8") as f:
#                         f.write(json.dumps(entry, ensure_ascii=False) + "\n")
#                 except Exception:
#                     pass

#                 return comment

#             except Exception as e:
#                 print("⚠️ Comment generation failed:", e)
#                 return None
#         else:
#             print("⏭️ Post not relevant, skipping.")
#             return None


from typing import Optional
import json
import datetime
from pathlib import Path

class CommentDecisionAgent:
    """
    Decides if a LinkedIn post is relevant and generates short professional comments.
    """

    def __init__(self, ai_generator, niche: str):
        self.ai = ai_generator
        self.niche = niche.lower().strip()

    def should_comment(self, post_text: str) -> bool:
        """Decide if the post is relevant based on niche keywords."""
        post_text = (post_text or "").strip()
        if not post_text:
            return False

        # Simple keyword-based relevance check
        match = any(keyword.strip() in post_text.lower() for keyword in self.niche.split(","))
        print(f"[decision_agent] Relevance check: {match}")

        # Log decision
        try:
            Path("data").mkdir(parents=True, exist_ok=True)
            entry = {
                "timestamp": datetime.datetime.now().isoformat(),
                "type": "decision",
                "niche": self.niche,
                "post_snippet": post_text[:400],
                "decision_raw": "yes" if match else "no",
            }
            with open("data/decisions.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return match

    def decide_and_generate(self, post_text: str, tone: str = "Professional") -> Optional[str]:
        """
        Generate a short LinkedIn comment (1–2 lines, ~30 words) for relevant posts.
        Avoid AI-like prefixes and make it sound human.
        """
        if not self.should_comment(post_text):
            print("⏭️ Post not relevant, skipping.")
            return None

        try:
            prompt = (
                f"Write a short LinkedIn comment (1–2 lines, max 30 words) "
                f"that sounds natural, {tone.lower()}, and human. "
                f"Focus on appreciation, insight, or relevance to the topic. "
                f"Do not start with phrases like 'Here's a comment', 'Great post', or 'I think'.\n\n"
                f"Post:\n{post_text}\n\n"
                "Write only the final comment, nothing else."
            )

            comment = self.ai.llm.invoke(prompt)
            if isinstance(comment, dict) and "text" in comment:
                comment = comment["text"]
            comment = str(comment).strip()

            # Remove unwanted AI-like prefixes
            bad_starts = ["here's", "here is", "a possible", "comment:", "sure",
                          "assistant:", "response:", "great post"]
            for prefix in bad_starts:
                if comment.lower().startswith(prefix):
                    comment = comment[len(prefix):].strip(" :-\n\t\"'")

            # Limit to 30 words
            words = comment.split()
            if len(words) > 30:
                comment = " ".join(words[:30]) + "..."

            comment = comment.replace("\n", " ").strip()
            print(f"[decision_agent] Final comment: {comment}")

            # Log comment generation
            try:
                entry = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "type": "generation",
                    "post_snippet": post_text[:400],
                    "comment_preview": comment,
                    "niche": self.niche
                }
                with open("data/decisions.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except Exception:
                pass

            return comment

        except Exception as e:
            print("⚠️ Comment generation failed:", e)
            return None
