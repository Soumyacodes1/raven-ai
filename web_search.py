"""
Raven AI — Web Search Module
Uses DuckDuckGo for free web search (no API key needed).
"""

from duckduckgo_search import DDGS


def should_search(user_message, intent):
    """Decide if we should search the web for this query."""
    # Always search for certain intents
    if intent in ("question", "learning"):
        # Check for time-sensitive or factual indicators
        triggers = [
            "latest", "today", "current", "news", "price", "2024", "2025", "2026",
            "who is", "what is", "when did", "where is", "how many", "how much",
            "stock", "weather", "score", "result", "update", "recent",
            "new release", "launched", "announced", "trending", "popular",
            "best", "top 10", "top 5", "compare", "vs", "versus",
            "review", "specs", "specification", "features of",
        ]
        msg_lower = user_message.lower()
        return any(trigger in msg_lower for trigger in triggers)
    return False


def search_web(query, max_results=5):
    """Search DuckDuckGo and return formatted results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return None, []

        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
                "url": r.get("href", ""),
            })

        # Build context string for the LLM
        context_parts = ["Here are relevant web search results:\n"]
        for i, r in enumerate(formatted, 1):
            context_parts.append(f"{i}. **{r['title']}**")
            context_parts.append(f"   {r['snippet']}")
            context_parts.append(f"   Source: {r['url']}\n")

        context_string = "\n".join(context_parts)
        return context_string, formatted
    except Exception:
        return None, []
