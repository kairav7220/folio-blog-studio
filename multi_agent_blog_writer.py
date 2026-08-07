from __future__ import annotations

import operator
import os
import re
import threading
import time
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
from typing import Annotated, TypedDict, List, Literal, Optional

from huggingface_hub import InferenceClient

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_mistralai import ChatMistralAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults

from dotenv import load_dotenv
load_dotenv()

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
    published_at: Optional[str] = None # Do not rely on it
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
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]
    as_of: str
    recency_days: int
    sections: Annotated[List[tuple[int, str]], operator.add] # (task_id, section_md)
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]
    final: str

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
    """Invoke a runnable, retrying on failure (esp. 429) while rotating through API keys."""
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
    topic = state['topic']
    decision = _invoke_with_failover(
        lambda key: _base_llm(key).with_structured_output(RouterDecision),
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f'Topic: {topic}'),
        ],
    )
    return {
        'needs_research': decision.needs_research,
        'mode': decision.mode,
        'queries': decision.queries
    }

def route_next(state: State) -> str:
    return 'research' if state['needs_research'] else 'orchestrator'

def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    tool = TavilySearchResults(max_results=max_results)
    results = tool.invoke({'query': query})
    normalized: List[dict] = []

    for r in results or []:
        normalized.append({
            'title': r.get('title') or '',
            'url': r.get('url') or '',
            'snippet': r.get('content') or r.get('snippet') or '',
            'published_at': r.get('published_at') or r.get('date'),
            'source': r.get('source')
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

    for q in queries:
        raw_results.extend(_tavily_search(q, max_results=max_results))

    if not raw_results:
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

    plan = _invoke_with_failover(
        lambda key: _base_llm(key).with_structured_output(Plan),
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(content=
            f"Topic: {state['topic']}\n"
            f'Mode: {mode}\n\n'
            f'Evidence (ONLY use for fresh claims; may be empty):\n'
            f'{[e.model_dump() for e in evidence][:16]}'
            ),
        ],
    )

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
- Stay close to Target words (±15%).
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

    return {'sections': [(task.id, section_md)]}

def merger_node(state: State) -> dict:
    plan = state['plan']
    ordered_sections = [md for _, md in sorted(state['sections'], key=lambda x: x[0])]
    body = '\n\n'.join(ordered_sections).strip()
    merged_md = f'# {plan.blog_title}\n\n{body}\n'
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
    return {'image_specs': [img.model_dump() for img in image_plan.images]}

def _generate_images_bytes(prompt: str) -> bytes:
    """
    Return raw PNG bytes generated via Hugging Face serverless inference.
    Env Var: HF_API_KEY
    """
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

    # If images requested, generate and insert after their anchor headings.
    if image_specs:
        images_dir = Path('images')
        images_dir.mkdir(exist_ok=True)

        for spec in image_specs:
            anchor = spec.get('anchor') or ''
            filename = spec['filename']
            out_path = images_dir / filename

            # generate only if needed
            if not out_path.exists():
                try:
                    img_bytes = _generate_images_bytes(spec['prompt'])
                    out_path.write_bytes(img_bytes)
                except RuntimeError as e:
                    prompt_block = (
                        f"> ** [IMAGE GENERATION FAILED] ** {spec.get('caption','')}\n>\n"
                        f"> **Alt text**: {spec.get('alt', '')}\n>\n"
                        f"> **Prompt**: {spec.get('prompt','')}\n>\n"
                        f"> **Error**: {e}\n"
                    )
                    md = _insert_image_after_heading(md, anchor, prompt_block)
                    continue
            img_md = f"![{spec['alt']}](images/{filename})\n*{spec['caption']}*"
            md = _insert_image_after_heading(md, anchor, img_md)

    filename = _sanitize_filename(f'{plan.blog_title}.md')
    Path(filename).write_text(md, encoding='utf-8')
    return {'final': md}

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

app = g.compile()
try:
    with open("multi_agent_blog_writer.png", "wb") as f:
        f.write(app.get_graph().draw_mermaid_png())
    print("Graph image saved successfully as 'multi_agent_blog_writer.png'!")
except Exception as e:
    print(f"Could not save graph image: {e}")


def run(topic: str):
    out = app.invoke({
        'topic': topic,
        'mode': '',
        'needs_research': False,
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
    })
    return out

out = run('Write a blog on Self Attention')

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
print(out)