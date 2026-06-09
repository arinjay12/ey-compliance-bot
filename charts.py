"""
charts.py
---------
Level 4: turn a numeric answer into a chart.

After an answer is generated, if it contains chartable numeric data (a timeline of
days, amounts, percentages, counts), we ask the LLM to pull out a small data series
and render it with Plotly. A cheap text pre-check (looks_chartable) means the extra
extraction call only runs when the answer plausibly has numbers worth charting — so
ordinary answers stay fast.
"""
import json
import re

# Cues that suggest the answer has quantitative content worth charting.
_CHART_CUES = re.compile(
    r"₹|rs\.?\s*\d|lakh|crore|%|per\s*cent|t\s*\+\s*\d|\bdays?\b|\bmonths?\b|"
    r"\btimeline\b|\blimit\b", re.I)


def looks_chartable(answer: str) -> bool:
    """Cheap pre-check: does the answer plausibly contain chartable numbers?
    Requires at least two numbers and a quantitative cue, so we only spend the
    extraction LLM call when it's worth it."""
    if not answer:
        return False
    numbers = re.findall(r"\d[\d,]*\.?\d*", answer)
    return len(numbers) >= 2 and bool(_CHART_CUES.search(answer))


_EXTRACT_PROMPT = """From the compliance question and answer below, extract numeric data
that would be clearer as a chart (e.g. a timeline of days, amounts, percentages, counts).

Return ONLY JSON, no prose:
{{"chart_type": "bar" or "line" or "none",
  "title": "<short title>",
  "x_label": "<category axis label>",
  "y_label": "<value axis label, include the unit, e.g. Days or Rs lakh>",
  "data": [{{"label": "<category>", "value": <number>}}]}}

Rules:
- "value" must be a plain number only (no units, no commas).
- Include at least 2 data points, otherwise return {{"chart_type": "none"}}.
- Use "line" only for a progression over time; otherwise "bar".
- Only use numbers that appear in the answer. Do not invent data.

Question: {question}
Answer: {answer}
JSON:"""


def _parse_json(raw: str):
    if not raw:
        return None
    m = re.search(r"\{.*\}", raw, re.S)   # first {...} block, even inside ``` fences
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def extract_chart_data(question: str, answer: str, llm) -> dict | None:
    """Ask the LLM to pull a small data series from the answer. Returns a spec
    dict {chart_type,title,x_label,y_label,data:[{label,value}]} or None."""
    try:
        raw = llm.invoke(_EXTRACT_PROMPT.format(question=question, answer=answer))
    except Exception:
        return None
    spec = _parse_json(raw)
    if not spec or spec.get("chart_type") not in ("bar", "line"):
        return None
    clean = []
    for d in spec.get("data") or []:
        try:
            clean.append({"label": str(d["label"])[:40], "value": float(d["value"])})
        except (KeyError, TypeError, ValueError):
            continue
    if len(clean) < 2:
        return None
    spec["data"] = clean
    spec["title"] = str(spec.get("title", ""))[:80]
    return spec


def build_figure(spec: dict):
    """Build a Plotly figure from a chart spec (EY colours)."""
    import plotly.graph_objects as go
    labels = [d["label"] for d in spec["data"]]
    values = [d["value"] for d in spec["data"]]
    if spec.get("chart_type") == "line":
        trace = go.Scatter(x=labels, y=values, mode="lines+markers",
                           line=dict(color="#2e2e38", width=2),
                           marker=dict(color="#ffe600", size=9,
                                       line=dict(color="#2e2e38", width=1)))
    else:
        trace = go.Bar(x=labels, y=values, marker_color="#ffe600",
                       marker_line_color="#2e2e38", marker_line_width=1)
    fig = go.Figure(trace)
    fig.update_layout(
        title=spec.get("title", ""),
        xaxis_title=spec.get("x_label", ""),
        yaxis_title=spec.get("y_label", ""),
        margin=dict(l=10, r=10, t=44, b=10),
        height=380,
        plot_bgcolor="white",
        font=dict(color="#2e2e38"),
    )
    return fig
