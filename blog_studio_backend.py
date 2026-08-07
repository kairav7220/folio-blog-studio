from __future__ import annotations

import json
import operator
import os
import re
import sqlite3
import threading
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Annotated, List, Literal, Optional, TypedDict

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_mistralai import ChatMistralAI
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, Field
from tavily import TavilyClient

load_dotenv()

try:
    import streamlit as st
    for _k, _v in dict(st.secrets).items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'blogs.db'
IMAGES_DIR = BASE_DIR / 'images'


# ---------------------------------------------------------------------------
# Pipeline: models
# ---------------------------------------------------------------------------

class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(..., description='One sentence describing what the reader should be able to do/understand after this section.')
    bullets: List[str] = Field(..., min_length=3, max_length=6, description='3-6 concrete, non-overlapping subpoints to cover in this section.')
    target_words: int = Field(..., description='Target word count for this section (120-550).')
    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False
    section_type: Literal['intro', 'core', 'examples', 'checklist', 'common_mistakes', 'conclusion'] = Field(...,
        description="Use 'common_mistakes' exactly once in the plan."
    )


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal['explainer', 'tutorial', 'news_roundup', 'comparison', 'system_design'] = 'explainer'
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal['closed_book', 'hybrid', 'open_book']
    queries: List[str] = Field(default_factory=list)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


class ImageSpec(BaseModel):
    anchor: str = Field(..., description="Exact section heading (including leading '## ', e.g. '## Self-Attention') after which this image is inserted.")
    filename: str = Field(..., description='Save under images/, e.g. qkv_cache.png')
    alt: str
    caption: str
    prompt: str = Field(..., description='Prompt to send to the image model.')
    size: Literal['1024x1024', '1024x1536', '1536x1024'] = '1024x1024'
    quality: Literal['low', 'medium', 'high'] = 'medium'


class GlobalImagePlan(BaseModel):
    images: List[ImageSpec] = Field(default_factory=list)


class State(TypedDict):
    topic: str
    mode: str
    needs_research: bool
    force_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str
    user_prefs: str


# ---------------------------------------------------------------------------
# Pipeline: LLM plumbing (dual-key failover + throttle)
# ---------------------------------------------------------------------------

_LLM_MODEL = 'mistral-large-latest'
_LLM_TEMPERATURE = 0.3
_LLM_MAX_TOKENS = 8192

_MISTRAL_KEYS = [k for k in (os.getenv('MISTRAL_API_KEY'), os.getenv('MISTRAL_API_KEY_2')) if k]
if not _MISTRAL_KEYS:
    raise RuntimeError('No MISTRAL_API_KEY / MISTRAL_API_KEY_2 found in environment')

_key_counter = 0


def _rotate_key() -> str:
    global _key_counter
    key = _MISTRAL_KEYS[_key_counter % len(_MISTRAL_KEYS)]
    _key_counter += 1
    return key


def _base_llm(api_key: Optional[str] = None) -> ChatMistralAI:
    return ChatMistralAI(
        model=_LLM_MODEL,
        temperature=_LLM_TEMPERATURE,
        max_tokens=_LLM_MAX_TOKENS,
        api_key=api_key or _rotate_key(),
    )


def _is_rate_limit(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(m in msg for m in ('429', 'rate limit', 'too many requests', 'quota'))


_llm_lock = threading.Lock()
_last_llm_call = 0.0
_LLM_MIN_INTERVAL = 2.0


def _throttle():
    global _last_llm_call
    with _llm_lock:
        wait = _LLM_MIN_INTERVAL - (time.monotonic() - _last_llm_call)
        if wait > 0:
            time.sleep(wait)
        _last_llm_call = time.monotonic()


def _invoke_with_failover(runnable_factory, messages, attempts_per_key: int = 3, base_wait: float = 2.0, rate_limit_max_wait: float = 60.0):
    last_error: Optional[BaseException] = None
    total = len(_MISTRAL_KEYS) * attempts_per_key
    for i in range(total):
        _throttle()
        try:
            result = runnable_factory(_rotate_key()).invoke(messages)
            if result is None:
                raise RuntimeError('model returned empty/unparseable output; retrying')
            return result
        except Exception as e:
            last_error = e
            if _is_rate_limit(e):
                wait = min(base_wait * (2 ** i), rate_limit_max_wait)
            else:
                wait = base_wait * 0.5
            if i < total - 1:
                time.sleep(wait)
    raise RuntimeError(f'LLM call failed after {total} attempts across all keys') from last_error


def _throttled_llm_invoke(messages):
    return _invoke_with_failover(_base_llm, messages)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------

_STEP_ORDER = ['Route', 'Research', 'Outline', 'Draft', 'Revise', 'Illustrate', 'Bind']

_PROGRESS_SINK = None


def _emit(event: dict):
    if _PROGRESS_SINK is not None:
        try:
            _PROGRESS_SINK.append(event)
        except Exception:
            pass


def _step(status: str, name: str, detail: str = ''):
    _emit({
        'type': 'step',
        'step': _STEP_ORDER.index(name),
        'name': name,
        'status': status,
        'detail': detail,
    })


def _info(message: str):
    _emit({'type': 'info', 'message': message})


# ---------------------------------------------------------------------------
# Pipeline: nodes
# ---------------------------------------------------------------------------

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3-10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- If user asked for "last week/this week/latest", reflect that constraint IN THE QUERIES.
"""


def router_node(state: State) -> dict:
    _step('running', 'Route', 'Reading the brief…')
    decision = _invoke_with_failover(
        lambda key: _base_llm(key).with_structured_output(RouterDecision),
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f'Topic: {state["topic"]}'),
        ],
    )
    _step('done', 'Route', f'Mode: {decision.mode}')
    return {
        'needs_research': decision.needs_research,
        'mode': decision.mode,
        'queries': decision.queries
    }


def route_next(state: State) -> str:
    if state.get('force_research') or state.get('needs_research'):
        return 'research'
    return 'orchestrator'


def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    response = TavilyClient().search(
        query=query,
        max_results=max_results,
        search_depth='advanced',
    )
    normalized: List[dict] = []

    for r in response.get('results') or []:
        normalized.append({
            'title': r.get('title') or '',
            'url': r.get('url') or '',
            'snippet': r.get('content') or r.get('raw_content') or '',
            'published_at': r.get('published_date'),
            'source': r.get('source'),
        })
    return normalized


RESEARCH_SYSTEM = """You are a research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources (company blogs, docs, reputable outlets).
- If a published date is explicitly present in the result payload, keep it as YYYY-MM-DD.
  If missing or unclear, set published_at=null. Do NOT guess.
- Keep snippets short.
- Deduplicate by URL.
"""


def research_node(state: State) -> dict:
    queries = (state.get('queries', []) or [])
    max_results = 6
    raw_results: List[dict] = []

    _step('running', 'Research', f'Searching {len(queries)} queries via Tavily…')
    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
        _step('done', 'Research', 'No sources found')
        return {'evidence': []}

    pack = _invoke_with_failover(
        lambda key: _base_llm(key).with_structured_output(EvidencePack),
        [
            SystemMessage(content=RESEARCH_SYSTEM),
            HumanMessage(content=f'Raw results:\n{raw_results}')
        ],
    )

    dedup = {}
    for e in pack.evidence:
        if e.url:
            dedup[e.url] = e
    _step('done', 'Research', f'{len(dedup)} sources kept')
    return {'evidence': list(dedup.values())}


ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5-9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3-6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120-550)

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book:
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections unless user explicitly asked for that.
  - If evidence is empty or insufficient, create a plan that transparently says "insufficient sources"
    and includes only what can be supported.

Output must strictly match the Plan schema.
"""


def orchestrator_node(state: State) -> dict:
    evidence = state.get('evidence', [])
    mode = state.get('mode', 'closed_book')
    prefs = state.get('user_prefs') or ''

    _step('running', 'Outline', 'Structuring the sections…')
    human_msg = (
        f"Topic: {state['topic']}\n"
        f'Mode: {mode}\n\n'
        f'Evidence (ONLY use for fresh claims; may be empty):\n'
        f'{[e.model_dump() for e in evidence][:16]}'
    )
    if prefs:
        human_msg += f'\n\nUser preferences (respect them):\n{prefs}'

    plan = _invoke_with_failover(
        lambda key: _base_llm(key).with_structured_output(Plan),
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(content=human_msg),
        ],
    )
    _step('done', 'Outline', f'{len(plan.tasks)} sections planned')
    return {'plan': plan}


def fanout(state: State):
    return [Send('worker', {
        'task': task.model_dump(),
        'topic': state['topic'],
        'mode': state.get('mode', 'closed_book'),
        'plan': state['plan'].model_dump(),
        'evidence': [e.model_dump() for e in state.get('evidence', [])[:8]],
    }) for task in state['plan'].tasks]


WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (+-15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": do NOT turn this into a tutorial/how-to guide.
  Do NOT teach web scraping, RSS, automation, or "how to fetch news" unless bullets explicitly ask for it.
  Focus on summarizing events and implications.

Grounding policy:
- If mode == open_book:
  - Do NOT introduce any specific event/company/model/funding/policy claim unless it is supported by provided Evidence URL.
  - For each event claim, attach a source as a Markdown link: ([Source](URL)).
  - Only use URLs provided in Evidence. If not supported, write: "Not found in provided sources."
- If requires_citations == true:
  - For outside-world claims, cite Evidence URLs the same way.
- Evergreen reasoning is OK without citations unless requires_citations is true.

Code:
- If requires_code == true, include at least one minimal, correct code snippet relevant to the bullets.

Style:
- Short paragraphs, bullets where helpful, code fences for code.
- Avoid fluff/marketing. Be precise and implementation-oriented.
"""


def worker_node(payload: dict) -> dict:
    task = Task(**payload['task'])
    plan = Plan(**payload['plan'])
    evidence = [EvidenceItem(**e) for e in payload.get('evidence', [])]
    topic = payload['topic']
    mode = payload.get('mode', 'closed_book')

    _step('running', 'Draft', f'Writing: {task.title}')
    bullets_text = '\n- ' + '\n- '.join(task.bullets)

    evidence_text = ''
    if evidence:
        evidence_text = '\n'.join(
            f"- {e.title} | {e.url} | {e.published_at or 'date:unknown'}".strip()
            for e in evidence[:20]
        )

    section_md = _throttled_llm_invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f'Blog title: {plan.blog_title}\n'
            f'Audience: {plan.audience}\n'
            f'Tone: {plan.tone}\n'
            f'Blog kind: {plan.blog_kind}\n'
            f'Constraints: {plan.constraints}\n'
            f'Topic: {topic}\n'
            f'Mode: {mode}\n\n'
            f'Section title: {task.title}\n'
            f'Goal: {task.goal}\n'
            f'Target words: {task.target_words}\n'
            f'Tags: {task.tags}\n'
            f'requires_research: {task.requires_research}\n'
            f'requires_citations: {task.requires_citations}\n'
            f'requires_code: {task.requires_code}\n'
            f'Bullets: {bullets_text}\n'
            f'Evidence (only use for fresh claims; may be empty):\n{evidence_text}\n'
        )),
    ]).content.strip()

    _step('done', 'Draft', f'{task.title}')
    return {'sections': [(task.id, section_md)]}


def merger_node(state: State) -> dict:
    _step('running', 'Revise', 'Merging sections…')
    plan = state['plan']
    ordered_sections = [md for _, md in sorted(state['sections'], key=lambda x: x[0])]
    body = '\n\n'.join(ordered_sections).strip()
    merged_md = f'# {plan.blog_title}\n\n{body}\n'
    _step('done', 'Revise', 'Sections bound into a draft')
    return {'merged_md': merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Max 3 images total.
- Each image must materially improve understanding (diagram/flow/table-like visual).
- For each image, set anchor to the EXACT heading line (starting with '## ') from the provided
  section list after which the image belongs. Do not invent headings.
- If no images needed: images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""


def decide_images(state: State) -> dict:
    merged_md = state['merged_md']
    plan = state['plan']
    assert plan is not None
    headings = [ln for ln in merged_md.splitlines() if ln.startswith('## ')]
    excerpt = '\n'.join(merged_md.splitlines()[:40])

    _step('running', 'Illustrate', 'Choosing diagrams…')
    image_plan = _invoke_with_failover(
        lambda key: _base_llm(key).with_structured_output(GlobalImagePlan),
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(content=(
                f'Blog kind: {plan.blog_kind}\n'
                f'Topic: {state["topic"]}\n\n'
                f'Section headings:\n' + ('\n'.join(headings) or '(none)') + '\n\n'
                f'Blog excerpt (first 40 lines):\n{excerpt}'
            )),
        ],
    )
    specs = [img.model_dump() for img in image_plan.images]
    _info(f'{len(specs)} diagram{"s" if len(specs) != 1 else ""} planned')
    return {'image_specs': specs}


def _generate_images_bytes(prompt: str) -> bytes:
    api_key = os.getenv('HF_API_KEY')
    if not api_key:
        raise RuntimeError('HF_API_KEY not set')

    client = InferenceClient(model='black-forest-labs/FLUX.1-schnell', token=api_key)
    image = client.text_to_image(prompt)
    buf = BytesIO()
    image.save(buf, format='PNG')
    return buf.getvalue()


def _sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '_', name).strip()


def _insert_image_after_heading(md: str, anchor: str, img_md: str) -> str:
    idx = md.find(anchor)
    if idx == -1:
        for line in md.splitlines():
            if line.startswith('## ') and anchor.strip('#').strip() in line:
                idx = md.find(line)
                anchor = line
                break
    if idx == -1:
        return md
    nxt = md.find('\n## ', idx + len(anchor))
    insert_at = nxt if nxt != -1 else len(md)
    return md[:insert_at] + '\n\n' + img_md + '\n\n' + md[insert_at:].lstrip('\n')


def generate_and_place_images(state: State) -> dict:
    plan = state['plan']
    assert plan is not None
    md = state['merged_md']
    image_specs = state.get('image_specs', []) or []

    if image_specs:
        IMAGES_DIR.mkdir(exist_ok=True)
        _step('running', 'Illustrate', f'Generating {len(image_specs)} image(s) with FLUX.1-schnell…')

        for spec in image_specs:
            anchor = spec.get('anchor') or ''
            filename = spec['filename']
            out_path = IMAGES_DIR / filename

            if not out_path.exists():
                try:
                    img_bytes = _generate_images_bytes(spec['prompt'])
                    out_path.write_bytes(img_bytes)
                except RuntimeError as e:
                    prompt_block = (
                        f"> ** [IMAGE GENERATION FAILED] ** {spec.get('caption', '')}\n>\n"
                        f"> **Alt text**: {spec.get('alt', '')}\n>\n"
                        f"> **Prompt**: {spec.get('prompt', '')}\n>\n"
                        f"> **Error**: {e}\n"
                    )
                    md = _insert_image_after_heading(md, anchor, prompt_block)
                    continue
            img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
            md = _insert_image_after_heading(md, anchor, img_md)

        _step('done', 'Illustrate', 'Diagrams placed')
    else:
        _step('done', 'Illustrate', 'No diagrams needed')

    return {'final': md}


# ---------------------------------------------------------------------------
# Pipeline: graph
# ---------------------------------------------------------------------------

def _build_graph():
    g = StateGraph(State)
    g.add_node('router', router_node)
    g.add_node('research', research_node)
    g.add_node('orchestrator', orchestrator_node)
    g.add_node('worker', worker_node)
    g.add_node('merge_content', merger_node)
    g.add_node('decide_images', decide_images)
    g.add_node('generate_and_place_images', generate_and_place_images)

    g.add_edge(START, 'router')
    g.add_conditional_edges('router', route_next, {'research': 'research', 'orchestrator': 'orchestrator'})
    g.add_edge('research', 'orchestrator')
    g.add_conditional_edges('orchestrator', fanout, ['worker'])
    g.add_edge('worker', 'merge_content')
    g.add_edge('merge_content', 'decide_images')
    g.add_edge('decide_images', 'generate_and_place_images')
    g.add_edge('generate_and_place_images', END)
    return g.compile()


app = _build_graph()


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

def _connect():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with _connect() as con:
        con.execute('''
            CREATE TABLE IF NOT EXISTS blogs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                topic TEXT NOT NULL,
                options_json TEXT NOT NULL DEFAULT '{}',
                md TEXT NOT NULL,
                images_json TEXT NOT NULL DEFAULT '[]',
                word_count INTEGER NOT NULL DEFAULT 0,
                duration_seconds REAL NOT NULL DEFAULT 0,
                version INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                edited_at TEXT
            )
        ''')
        con.commit()


init_db()


def _word_count(md: str) -> int:
    code_lines = 0
    in_fence = False
    for line in md.splitlines():
        if line.strip().startswith('```'):
            in_fence = not in_fence
            continue
        if in_fence:
            code_lines += len(line.split())
    total = len(md.split())
    return max(total - code_lines, 0)


def _extract_images(md: str) -> List[str]:
    refs = re.findall(r'\]\((images/[^)]+)\)', md)
    return sorted(set(refs))


def _row_to_blog(row) -> dict:
    return {
        'id': row['id'],
        'title': row['title'],
        'topic': row['topic'],
        'options': json.loads(row['options_json'] or '{}'),
        'md': row['md'],
        'images': json.loads(row['images_json'] or '[]'),
        'word_count': row['word_count'],
        'duration_seconds': row['duration_seconds'],
        'version': row['version'],
        'created_at': row['created_at'],
        'edited_at': row['edited_at'],
    }


def list_blogs() -> List[dict]:
    with _connect() as con:
        rows = con.execute(
            'SELECT id, title, topic, word_count, version, created_at, images_json FROM blogs ORDER BY id DESC'
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            'id': r['id'],
            'title': r['title'],
            'topic': r['topic'],
            'word_count': r['word_count'],
            'version': r['version'],
            'created_at': r['created_at'],
            'images': json.loads(r['images_json'] or '[]'),
        })
    return out


def get_blog(blog_id: int) -> Optional[dict]:
    with _connect() as con:
        row = con.execute('SELECT * FROM blogs WHERE id = ?', (blog_id,)).fetchone()
    return _row_to_blog(row) if row else None


def delete_blog(blog_id: int) -> bool:
    with _connect() as con:
        cur = con.execute('DELETE FROM blogs WHERE id = ?', (blog_id,))
        con.commit()
    return cur.rowcount > 0


def _save_new_blog(title: str, topic: str, options: dict, md: str, images: List[str],
                   duration: float) -> dict:
    now = datetime.now().isoformat(timespec='seconds')
    with _connect() as con:
        cur = con.execute(
            'INSERT INTO blogs (title, topic, options_json, md, images_json, word_count, duration_seconds, version, created_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?)',
            (title, topic, json.dumps(options), md, json.dumps(images), _word_count(md), duration, now),
        )
        con.commit()
        blog_id = cur.lastrowid
    return get_blog(blog_id)


def update_blog_md(blog_id: int, md: str) -> Optional[dict]:
    now = datetime.now().isoformat(timespec='seconds')
    images = _extract_images(md)
    with _connect() as con:
        cur = con.execute(
            'UPDATE blogs SET md = ?, images_json = ?, word_count = ?, version = version + 1, edited_at = ? WHERE id = ?',
            (md, json.dumps(images), _word_count(md), now, blog_id),
        )
        con.commit()
    if cur.rowcount == 0:
        return None
    return get_blog(blog_id)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _build_prefs(options: dict) -> str:
    parts = []
    if options.get('audience'):
        parts.append(f"Audience: {options['audience']}")
    if options.get('tone'):
        parts.append(f"Tone: {options['tone']}")
    if options.get('kind') and options['kind'] != 'Auto':
        parts.append(f"Blog kind: {options['kind']}")
    if options.get('research') == 'Deep':
        parts.append('Do thorough web research: rely on recent sources and cite them.')
    return ' | '.join(parts)


def create_blog(topic: str, options: Optional[dict] = None, progress: Optional[callable] = None) -> dict:
    options = options or {}
    state = {
        'topic': topic.strip(),
        'mode': '',
        'needs_research': False,
        'force_research': options.get('research') == 'Deep',
        'queries': [],
        'evidence': [],
        'plan': None,
        'as_of': '',
        'recency_days': 0,
        'sections': [],
        'merged_md': '',
        'md_with_placeholders': '',
        'image_specs': [],
        'final': '',
        'user_prefs': _build_prefs(options),
    }

    global _PROGRESS_SINK
    _PROGRESS_SINK = progress if progress is not None else None
    t0 = time.perf_counter()
    try:
        _step('running', 'Bind', 'Starting the press run…')
        out = app.invoke(state)
        duration = time.perf_counter() - t0

        plan = out.get('plan')
        final_md = out.get('final') or out.get('merged_md') or ''
        title = plan.blog_title if plan else _sanitize_filename(topic) or 'Untitled'
        images = _extract_images(final_md)

        blog = _save_new_blog(
            title=title,
            topic=topic.strip(),
            options=options,
            md=final_md,
            images=images,
            duration=duration,
        )
        _info(f'Bound as "{title}" ({_word_count(final_md)} words, {len(images)} image(s), {duration:.1f}s)')
        _step('done', 'Bind', 'Finished')
        return blog
    finally:
        _PROGRESS_SINK = None


def regenerate_blog(blog_id: int, progress: Optional[callable] = None) -> dict:
    blog = get_blog(blog_id)
    if not blog:
        raise ValueError(f'Blog {blog_id} not found')
    return create_blog(blog['topic'], blog['options'], progress=progress)
