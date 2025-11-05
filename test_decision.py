from linkedin_bot.ai_engine_clean import AICommentGenerator
from linkedin_bot.decision_agent import CommentDecisionAgent

ai = AICommentGenerator()
agent = CommentDecisionAgent(ai, niche='SEO, Marketing')

posts = [
    "We improved onboarding by 30% using a new checklist and cross-team reviews.",
    "Beautiful sunset today! #life",
    "Quick tip: optimize your content for user intent and watch traffic grow.",
]

for p in posts:
    print('---')
    print('post:', p)
    should = agent.should_comment(p)
    print('should_comment:', should)
    c = agent.decide_and_generate(p, tone='Professional')
    print('generated:', c)
