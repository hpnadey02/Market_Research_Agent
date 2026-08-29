# ==============================================================================
# prompts.py — Analyst-Grade Prompt Templates
# ==============================================================================
# Defines the system prompts and user templates for generating high-quality
# market research insights.
# ==============================================================================

# ── Supply Insight Prompt ────────────────────────────────────────────────────
SUPPLY_PROMPT_TEMPLATE = """
As a Senior Commodity Market Analyst with 20+ years of experience, write a SINGLE professional paragraph on the SUPPLY-SIDE dynamics for {metal} in {country}.

CONTEXT (Retrieved News):
{context}

MARKET DATA:
- Current Price: {price}
- Weekly Change: {change}%

INSTRUCTIONS:
1. Explain the SUPPLY drivers causing or reacting to this price movement.
2. Synthesize the provided news context into a coherent narrative.
3. Use a natural, authoritative human tone.
4. Integrate the weekly percentage change logically.
5. Mention specific production figures, mine closures, or export/import data if present in context.
6. NO AI phrases like "Based on the provided text" or "In conclusion".
7. Focus ONLY on supply (production, inventory, exports, logistics).

OUTPUT:
(A single, dense, professional paragraph)
"""

# ── Demand Insight Prompt ────────────────────────────────────────────────────
DEMAND_PROMPT_TEMPLATE = """
As a Senior Commodity Market Analyst, write a SINGLE professional paragraph on the DEMAND-SIDE dynamics for {metal} in {country}.

CONTEXT (Retrieved News):
{context}

MARKET DATA:
- Current Price: {price}
- Weekly Change: {change}%

INSTRUCTIONS:
1. Explain the DEMAND drivers (consumption, downstream sectors like auto/infra).
2. Synthesize the news context.
3. Maintain narrative continuity with a supply analysis (implicit).
4. Integrate the price and percentage change.
5. Mention specific buying trends, sector performance, or economic indicators from the context.
6. NO AI phrases.
7. Focus ONLY on demand.

OUTPUT:
(A single, dense, professional paragraph)
"""

# ── Summary Prompt ───────────────────────────────────────────────────────────
SUMMARY_PROMPT_TEMPLATE = """
Write a concise 150-word Executive Summary for this {metal} market report ({country}).

SUPPLY INSIGHT:
{supply_insight}

DEMAND INSIGHT:
{demand_insight}

MARKET DATA:
- Price: {price}
- Change: {change}%

INSTRUCTIONS:
1. Create a "monthly-style" narrative summary.
2. Combine the key supply and demand signals.
3. Provide a clear forward-looking directional takeaway.
4. Do NOT simply list the % values again unless critical for the narrative.
5. Strict limit: 150 words.
6. Professional, decision-maker focused tone.

OUTPUT:
(An executive summary paragraph)
"""
