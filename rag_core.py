"""
rag_core.py
-----------
The retrieval-augmented generation (RAG) pipeline, extracted from app.py so it
can be used WITHOUT Streamlit — by the UI, the Level 2 monitor, and the
evaluation harness alike. One source of truth for retrieval and prompting.

Why this module exists:
  - app.py used to re-instantiate ChromaDB on every single query (slow).
    Here the embedding model and vector store are loaded once and cached.
  - The evaluation harness needs to call the exact same retrieval the UI uses,
    so we don't end up testing different code than we ship.
  - Keeps the LLM backend swappable: change LLM_BACKEND/LLM_MODEL in one place
    and everything (UI, monitor, eval) follows. If EY later approves a hosted
    model, only get_llm() changes.

Public API:
  retrieve(query, k=4)        -> list[Document]      (hybrid retrieval)
  build_prompt(docs, query)   -> str
  answer_rag(query)           -> (answer, source_docs)
  answer_vanilla(query)       -> answer             (no retrieval; baseline)
"""

import re
from pathlib import Path

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama

# ── Config ──────────────────────────────────────────────────────────────────
CHROMA_DIR  = "./chroma_db"
DOCS_DIR    = "./docs"
EMBED_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL   = "llama3"          # swap here to change the LLM everywhere
LLM_TEMPERATURE = 0.1

# Chunking (used by ingest.py and the app's upload path)
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 150

# Retrieval params
MMR_K          = 4              # final chunks from semantic search
MMR_FETCH_K    = 30             # candidates considered before MMR picks K
INJECT_PER_DOC = 5              # max chunks injected from a single matched doc
INJECT_MAX     = 8              # max chunks injected across all matched docs
INJECT_FETCH   = 8              # candidates pulled from each targeted document

# Relevance gate (loose backstop): if the nearest chunk's L2 distance exceeds
# this, refuse outright. Calibration (evaluation/calibrate_gate.py) showed that
# in this regulatory domain the embeddings are compressed — out-of-scope queries
# (~0.72) overlap with legitimate in-scope ones (up to ~1.02) — so distance alone
# CANNOT cleanly separate them. We therefore keep the gate loose (catches only
# clearly-unrelated queries) and rely on a strict grounding instruction in the
# prompt as the primary out-of-scope guard.
REFUSE_DISTANCE = 1.15

REFUSAL_LINE = "This information is not available in the uploaded documents."

# Verified document-level metadata, keyed by PDF filename. These are facts that
# live only in a document's header (reference number, issue date) — header text
# is a poor semantic match for content queries, so those chunks are rarely
# retrieved. We attach this metadata to EVERY chunk's source tag at prompt-build
# time, so questions like "what is the circular reference number?" are
# answerable from any retrieved chunk and the model stops guessing.
# Values are hand-verified against the source PDFs' page-1 headers.
DOC_META = {
    "18__SEBI_Circular_dated_June_05_2025.pdf": {
        "title": "SEBI Circular (LODR Reg 58 relaxation)",
        "ref":   "SEBI/HO/DDHS/DDHS-PoD-1/P/CIR/2025/83",
        "date":  "05 June 2025",
    },
    "1749641449497.pdf": {
        "title": "SEBI Circular (UPI IDs for registered intermediaries)",
        "ref":   "SEBI/HO/DEPA-II/DEPA-II_SRG/P/CIR/2025/86",
        "date":  "11 June 2025",
    },
    "39_SEBI_Circular_dated_October_15_2025.pdf": {
        "title": "SEBI Master Circular (issue & listing of non-convertible securities)",
        "ref":   "SEBI/HO/DDHS/DDHS-PoD/P/CIR/2025/0000000137",
        "date":  "15 October 2025",
    },
    "SEBI_Master_Circular_LODR_NCS_July2025.pdf": {
        "title": "SEBI Master Circular (LODR for non-convertible securities)",
        "ref":   "SEBI/HO/DDHS/DDHS-PoD-1/P/CIR/2025/0000000103",
        "date":  "11 July 2025",
    },
    "SEBI_Master_Circular_LODR_Listed_Entities_Jan2026.pdf": {
        "title": "SEBI Master Circular (LODR for listed entities)",
        "ref":   "HO/49/14/14(7)2025-CFD-POD2/I/3762/2026",
        "date":  "last updated 30 January 2026 (originally issued 11 July 2023)",
    },
}

# Maps query keywords -> specific document filenames for targeted retrieval.
# NOTE: this hardcoded map is a known limitation (it does not generalize to
# unseen documents). The evaluation harness measures how much we actually
# depend on it; the goal is to replace it with metadata-driven retrieval.
DOC_KEYWORD_MAP = {
    r"june[\s\w,]{0,12}2025|jun\s*2025|lodr relaxation|regulation\s*58|reg\s*58|58\(1\)":
        "18__SEBI_Circular_dated_June_05_2025.pdf",
    r"upi.*intermediar|intermediar.*upi|upi payment|payment.*upi|upi id|upi handle|sebi check":
        "1749641449497.pdf",
    r"october\s*2025|oct\s*2025|october 15":
        "39_SEBI_Circular_dated_October_15_2025.pdf",
    r"july\s*2025|jul\s*2025|master circular.*ncs|ncs.*master circular":
        "SEBI_Master_Circular_LODR_NCS_July2025.pdf",
    r"jan\w*\s*2026|january\s*2026":
        "SEBI_Master_Circular_LODR_Listed_Entities_Jan2026.pdf",
}

# ── Document metadata auto-extraction (for non-curated docs) ────────────────
_MONTHS = ("January|February|March|April|May|June|July|August|September|"
           "October|November|December")
_RE_DATE = re.compile(rf"({_MONTHS})\s+\d{{1,2}},?\s+\d{{4}}", re.I)
_RE_SEBI = re.compile(r"SEBI/[A-Z0-9][A-Z0-9\-_/().]+/\d{4}/\d+")
_RE_RBI  = re.compile(r"RBI/[\w\-./]+?/\d+")
_RE_RBI_MD = re.compile(r"(DoR|DOR|DBR|DNBR|DPSS|FED|RPCD)[.\w]*\s*No\.?\s*[\w./\-]+", re.I)


def auto_extract_meta(page1_text: str, fname: str) -> dict:
    """Best-effort extraction of ref/date/title from a document's first page.
    Used for any document not in the curated DOC_META map (e.g. RBI corpus,
    user-uploaded PDFs)."""
    text = page1_text or ""
    ref = ""
    for rx in (_RE_SEBI, _RE_RBI, _RE_RBI_MD):
        m = rx.search(text)
        if m:
            ref = m.group(0).strip()
            break
    md = _RE_DATE.search(text)
    date = md.group(0).strip() if md else ""
    title = ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:20]:
        if re.search(r"master direction|master circular|directions,\s*\d{4}|"
                     r"^circular|direction\b", ln, re.I) and len(ln) > 12:
            title = ln[:90]
            break
    if not title:
        for ln in lines[:10]:
            if len(ln) > 25:
                title = ln[:90]
                break
    if not title:
        title = fname.replace(".pdf", "").replace(".PDF", "")
    return {"title": title, "ref": ref or "not stated", "date": date or "not stated"}


def doc_meta_for(fname: str, page1_text: str = "") -> dict:
    """Curated metadata if known, else auto-extracted."""
    return DOC_META.get(fname) or auto_extract_meta(page1_text, fname)


# ── Cached singletons (loaded once, reused) ─────────────────────────────────
_embeddings = None
_vectorstore = None
_llm = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return _embeddings


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=CHROMA_DIR,
            embedding_function=get_embeddings(),
        )
    return _vectorstore


def reset_vectorstore():
    """Drop the cached vector store handle (call after adding new documents)."""
    global _vectorstore
    _vectorstore = None


def get_llm(temperature: float = LLM_TEMPERATURE):
    global _llm
    if _llm is None:
        _llm = Ollama(model=LLM_MODEL, temperature=temperature)
    return _llm


# ── Retrieval ───────────────────────────────────────────────────────────────
def retrieve(query: str, k: int = MMR_K, use_hybrid: bool = True) -> list:
    """
    Hybrid retrieval:
      1. MMR semantic search for broadly relevant, diverse chunks.
      2. If the query references a specific document (keyword map), inject
         chunks directly from that document so large docs can't drown it out.

    Set use_hybrid=False to get pure semantic search (used by the eval harness
    to measure how much the keyword injection actually contributes).
    """
    vs = get_vectorstore()
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": MMR_FETCH_K},
    )
    docs = list(retriever.invoke(query))

    if use_hybrid:
        query_lower = query.lower()
        existing = {d.page_content for d in docs}
        total_injected = 0
        # Inject from EVERY matched document (not just the first). A query like
        # "old UPI IDs after the June 2025 circular" matches both the June and the
        # UPI patterns — first-match-wins used to inject the wrong document.
        for pattern, filename in DOC_KEYWORD_MAP.items():
            if total_injected >= INJECT_MAX:
                break
            if re.search(pattern, query_lower):
                src_path = f"./docs\\{filename}"
                targeted = vs.similarity_search(
                    query, k=INJECT_FETCH, filter={"source": {"$eq": src_path}}
                )
                per_doc = 0
                for td in targeted:
                    if (td.page_content not in existing
                            and per_doc < INJECT_PER_DOC
                            and total_injected < INJECT_MAX):
                        docs.append(td)
                        existing.add(td.page_content)
                        per_doc += 1
                        total_injected += 1
    return docs


def source_tag(doc) -> str:
    """Build a rich, verified source label for a retrieved chunk."""
    fname = Path(doc.metadata.get("source", "Unknown")).name
    page = doc.metadata.get("page", "?")
    # Prefer per-chunk metadata written at ingest time; fall back to curated map.
    title = doc.metadata.get("doc_title")
    ref   = doc.metadata.get("doc_ref")
    date  = doc.metadata.get("doc_date")
    if not title:
        m = DOC_META.get(fname)
        if m:
            title, ref, date = m["title"], m["ref"], m["date"]
    if title:
        return f"[Source: {title} | Ref No: {ref} | Dated: {date} | Page {page}]"
    return f"[Source: {fname}, Page {page}]"


def min_distance(query: str) -> float:
    """L2 distance of the single nearest chunk — used by the relevance gate."""
    try:
        scored = get_vectorstore().similarity_search_with_score(query, k=1)
        return scored[0][1] if scored else 99.0
    except Exception:
        return 0.0   # fail open: don't block answering if scoring breaks


def build_prompt(docs: list, query: str) -> str:
    """Assemble the grounded prompt sent to the LLM."""
    context = "\n\n".join(
        f"{source_tag(doc)}\n{doc.page_content}" for doc in docs
    )

    return f"""You are a SEBI/RBI compliance expert assistant working at EY (Ernst & Young).
Answer the question using ONLY the context below, extracted from official SEBI and
RBI circulars. Each chunk is labelled with its source document, reference and page.

Rules:
- Answer using ONLY the facts stated in the context above. Do NOT use outside or
  prior knowledge, and do NOT guess.
- If the context contains the answer, state it directly and cite the source.
- A passing mention of a regulation's NAME is not the same as the context stating
  the specific fact asked for. If the precise fact requested (e.g. a penalty, a
  number, a date) is not explicitly stated in the context, do not answer.
- Only if the context does NOT contain the answer — for example it merely names a
  related regulation, or is about a different subject entirely — respond with
  EXACTLY this line and nothing else:
  "{REFUSAL_LINE}"
- Cite specific regulation numbers or circular references when available.
- Each chunk's "Source" label gives the document's verified title, official reference
  number ("Ref No") and date. When asked for a circular's reference number or date,
  use the "Ref No"/"Dated" fields from the Source label of the relevant chunk. Never
  invent a reference number and never quote a file name.

Context:
{context}

Question: {query}

Answer:"""


# ── Answer generation ───────────────────────────────────────────────────────
def answer_rag(query: str, use_hybrid: bool = True) -> tuple:
    """
    Full RAG answer. Returns (answer_text, source_docs).

    Relevance gate: if nothing in the corpus is genuinely close to the query,
    refuse immediately rather than letting the LLM answer from loosely-related
    chunks. This is the out-of-scope safety guard for a compliance tool.
    """
    if min_distance(query) > REFUSE_DISTANCE:
        return REFUSAL_LINE, []

    docs = retrieve(query, use_hybrid=use_hybrid)
    prompt = build_prompt(docs, query)
    answer = get_llm().invoke(prompt)
    return answer, docs


VANILLA_PROMPT = """You are a SEBI compliance expert assistant working at EY (Ernst & Young).
Answer the question accurately and concisely.
Cite specific SEBI regulation numbers or circular references when available.

Question: {query}

Answer:"""


def answer_vanilla(query: str) -> str:
    """
    Baseline: the SAME LLM with the SAME expert persona, but NO retrieved
    context. This isolates the value added by retrieval — the head-to-head
    that proves the RAG pipeline beats using the LLM directly.
    """
    return get_llm().invoke(VANILLA_PROMPT.format(query=query))
