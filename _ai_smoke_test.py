from linkedin_bot.ai_engine_clean import AICommentGenerator

g = AICommentGenerator()
print("backend:", getattr(g, "backend_name", "unknown"))
print("comment:", g.generate("Quick post about improving onboarding flows for SMBs", tone="Professional"))
