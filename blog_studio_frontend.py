from __future__ import annotations

import re
import threading
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

import blog_studio_backend as backend

_PAGE_TITLE = 'Folio'

st.set_page_config(page_title=_PAGE_TITLE, layout='wide', initial_sidebar_state='expanded')

for _k, _v in {
    'view': 'shelf',          # 'shelf' | 'new' | 'reader'
    'current_id': None,
    'edit_mode': False,
    'press_running': False,
    'press_events': [],
    'press_result': [],
    'press_thread': None,
    'confirm_delete_id': None,
}.items():
    st.session_state.setdefault(_k, _v)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

def _inject_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;600;700;800&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;1,8..60,400&family=JetBrains+Mono:wght@400;500;600&display=swap');

    [data-testid="stAppViewContainer"] { background: #FBFAF6; }
    [data-testid="stHeader"] { background: rgba(251, 250, 246, 0.92); }
    [data-testid="stSidebar"] { background: #F2F0E9; border-right: 1px solid #E5E0D6; }
    [data-testid="stSidebar"] .stMarkdown { font-family: 'Source Serif 4', Georgia, serif; color: #26292F; }

    .stMarkdown { font-family: 'Source Serif 4', Georgia, serif; color: #26292F; }
    .stMarkdown h1 { font-family: 'Playfair Display', Georgia, serif; color: #1F2430; letter-spacing: -0.015em; }
    .stMarkdown h2 { font-family: 'Playfair Display', Georgia, serif; color: #1F2430; border-bottom: 1px solid #E5E0D6; padding-bottom: .35rem; margin-top: 1.6rem; }
    .stMarkdown h3 { font-family: 'Playfair Display', Georgia, serif; color: #1F2430; }
    .stMarkdown p, .stMarkdown li, .stMarkdown td, .stMarkdown th { font-size: 1.04rem; line-height: 1.75; }
    .stMarkdown a { color: #E07A5F; }
    .stMarkdown blockquote { border-left: 3px solid #E07A5F; background: #F6F3EC; padding: .6rem 1rem; border-radius: 0 8px 8px 0; }
    .stMarkdown img { border-radius: 10px; border: 1px solid #E5E0D6; margin: .6rem 0; max-width: 100%; }
    pre { font-family: 'JetBrains Mono', Consolas, monospace; background: #F2F0E9 !important; border: 1px solid #E5E0D6; border-radius: 8px; }
    code { font-family: 'JetBrains Mono', Consolas, monospace; }
    .stMarkdown code:not(pre code) { background: #F2F0E9; border: 1px solid #E5E0D6; border-radius: 4px; padding: .1em .3em; }

    .stButton > button, .stDownloadButton > button { border-radius: 8px; font-family: 'JetBrains Mono', Consolas, monospace; }
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] { background: #E07A5F; border-color: #E07A5F; }
    .stButton > button:hover { border-color: #E07A5F; }
    .stTextInput input, .stTextArea textarea { border-radius: 8px; border-color: #E5E0D6; }
    [data-baseweb="select"] > div { border-radius: 8px; border-color: #E5E0D6; }

    .folio-kicker { font-family: 'JetBrains Mono', Consolas, monospace; font-size: .72rem; letter-spacing: .28em; text-transform: uppercase; color: #E07A5F; margin-bottom: .4rem; }
    .folio-title { font-family: 'Playfair Display', Georgia, serif; font-size: 4rem; font-weight: 700; color: #1F2430; letter-spacing: -0.02em; line-height: 1; }
    .folio-sub { font-family: 'Source Serif 4', Georgia, serif; font-style: italic; color: #6B7280; margin-top: .6rem; }
    .folio-tile { height: 150px; background: #1F2430; color: #FBFAF6; font-family: 'Playfair Display', Georgia, serif; font-size: 5rem; display: flex; align-items: center; justify-content: center; border-radius: 10px; margin-bottom: .5rem; }
    .folio-rule { border-top: 1px solid #E5E0D6; margin: 1.2rem 0; }

    [data-testid="stSidebar"] .stButton > button[kind="secondary"] {
      justify-content: flex-start; background: transparent; border: none; box-shadow: none;
      padding: .45rem .6rem; color: #26292F; text-align: left;
      font-family: 'Source Serif 4', Georgia, serif; font-size: 1rem;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] > div {
      flex: 1 1 auto; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    [data-testid="stSidebar"] .stButton > button[kind="secondary"] p {
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin: 0;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] {
      gap: 0 !important; border-radius: 8px; transition: background .15s ease;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div { padding: 0 !important; }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover { background: rgba(31, 36, 48, 0.06); }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover > div:first-child button { background: transparent; }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:last-child button {
      opacity: 0; transition: opacity .15s ease; background: transparent; border: none; box-shadow: none;
      width: 1.9rem; min-width: 1.9rem; padding: .3rem 0; font-size: 1.15rem; line-height: 1; color: #6B7280;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:hover > div:last-child button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(> div:first-child button:hover) > div:last-child button,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:last-child button:focus-visible {
      opacity: 1;
    }
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:last-child button:hover {
      background: rgba(31, 36, 48, 0.08); border: none;
    }
    @media (hover: none) {
      [data-testid="stSidebar"] [data-testid="stHorizontalBlock"] > div:last-child button { opacity: 1; }
    }
    [data-testid="stPopoverButton"] [aria-hidden="true"] { display: none; }
    </style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime('%b %d')
    except Exception:
        return iso or ''


def _fmt_dur(sec: float) -> str:
    sec = int(sec)
    if sec < 60:
        return f'{sec}s'
    return f'{sec // 60}m {sec % 60:02d}s'


def _meta(blog: dict) -> str:
    parts = [_fmt_date(blog['created_at'])]
    if blog.get('word_count'):
        parts.append(f"{blog['word_count']:,} words")
    if blog.get('version', 1) > 1:
        parts.append(f"v{blog['version']}")
    if blog.get('images'):
        parts.append(f"{len(blog['images'])} image(s)")
    return ' · '.join(parts)


def _strip_h1(md: str) -> str:
    lines = md.splitlines()
    if lines and lines[0].startswith('# '):
        lines = lines[1:]
    return '\n'.join(lines).lstrip('\n')


def _image_path(img_ref: str) -> Path:
    return backend.IMAGES_DIR / Path(img_ref).name


_IMG_LINE = re.compile(r'^!\[([^\]]*)\]\((images/[^)]+)\)\s*$')


def _render_article(md: str):
    md = _strip_h1(md)
    buffer: list = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _IMG_LINE.match(line.strip())
        if m:
            if buffer:
                st.markdown('\n'.join(buffer).strip())
                buffer = []
            alt, ref = m.group(1), m.group(2)
            caption = ''
            if i + 1 < len(lines):
                cand = lines[i + 1].strip()
                if cand.startswith('*') and cand.endswith('*') and len(cand) > 2:
                    caption = cand.strip('*').strip()
                    i += 1
            path = _image_path(ref)
            with st.container(border=True):
                if path.exists():
                    st.image(str(path), width='stretch')
                else:
                    st.markdown(f'*{alt or "Image unavailable"}*')
                if caption:
                    st.caption(caption)
        else:
            buffer.append(line)
        i += 1
    if buffer:
        st.markdown('\n'.join(buffer).strip())


def _render_delete_confirm():
    blog_id = st.session_state.confirm_delete_id
    if not blog_id:
        return
    blog = backend.get_blog(blog_id)
    if blog is None:
        st.session_state.confirm_delete_id = None
        return
    st.warning(f'Remove **{blog["title"]}** from the shelf? This cannot be undone.')
    c_yes, c_no = st.columns(2)
    with c_yes:
        if st.button('Confirm delete', type='primary', width='stretch', key='confirm_delete_yes'):
            backend.delete_blog(blog_id)
            st.session_state.confirm_delete_id = None
            if st.session_state.current_id == blog_id:
                st.session_state.current_id = None
                st.session_state.view = 'shelf'
            st.rerun()
    with c_no:
        if st.button('Keep it', width='stretch', key='confirm_delete_no'):
            st.session_state.confirm_delete_id = None
            st.rerun()


# ---------------------------------------------------------------------------
# Press run (background generation + live progress)
# ---------------------------------------------------------------------------

def _start_press(topic: str, options: dict):
    events: list = []
    result: list = []

    def worker():
        try:
            blog = backend.create_blog(topic, options, progress=events.append)
            result.append({'blog': blog})
        except Exception as exc:  # noqa: BLE001
            result.append({'error': str(exc)})

    st.session_state.press_events = events
    st.session_state.press_result = result
    st.session_state.press_thread = threading.Thread(target=worker, daemon=True)
    st.session_state.press_running = True
    st.session_state.press_thread.start()


def _slug_lines(events: list) -> list:
    lines = []
    for i, name in enumerate(backend._STEP_ORDER):
        step_events = [e for e in events if e.get('type') == 'step' and e.get('step') == i]
        if step_events:
            last = step_events[-1]
            mark = '\u2713' if last['status'] == 'done' else ('\u27f3' if last['status'] == 'running' else '\u25e6')
            detail = f" \u2014 {last['detail']}" if last.get('detail') else ''
            lines.append(f'**{mark} {name}**{detail}')
        else:
            lines.append(f'\u25e6 {name}')
    return lines


def _progress_fraction(events: list) -> float:
    done = running = 0
    for i in range(len(backend._STEP_ORDER)):
        step_events = [e for e in events if e.get('type') == 'step' and e.get('step') == i]
        if not step_events:
            continue
        if step_events[-1]['status'] == 'done':
            done += 1
        elif step_events[-1]['status'] == 'running':
            running += 1
    total = len(backend._STEP_ORDER)
    return (done + 0.5 * running) / total


def _status_line(events: list) -> str:
    for i in range(len(backend._STEP_ORDER)):
        step_events = [e for e in events if e.get('type') == 'step' and e.get('step') == i]
        if step_events and step_events[-1]['status'] == 'running':
            return f"Press run: {backend._STEP_ORDER[i]} (step {i + 1} of {len(backend._STEP_ORDER)})"
    done = sum(1 for e in events if e.get('type') == 'step' and e.get('status') == 'done')
    if done >= len(backend._STEP_ORDER):
        return 'Press run finished'
    return f'Press run starting (step 1 of {len(backend._STEP_ORDER)})'


def _render_press_run():
    st.markdown('<div class="folio-kicker">THE PRESS RUN</div>', unsafe_allow_html=True)
    with st.status('Press run in progress', expanded=True) as status:
        slug_area = st.empty()
        bar = st.progress(0.0, text='Starting\u2026')
        thread = st.session_state.press_thread
        events = st.session_state.press_events
        while thread.is_alive():
            time.sleep(0.25)
            slug_area.markdown('\n\n'.join(_slug_lines(events)))
            bar.progress(min(_progress_fraction(events), 1.0), text=_status_line(events))
            status.update(label=_status_line(events), state='running')
        slug_area.markdown('\n\n'.join(_slug_lines(events)))
        bar.progress(1.0, text='Press run finished')
        status.update(label='Press run finished', state='complete')

    result = st.session_state.press_result
    st.session_state.press_running = False
    if result and 'error' in result[0]:
        st.error(f"**The press run broke down.**\n\n{result[0]['error']}")
        st.session_state.view = 'new'
        st.session_state.press_events = []
        st.session_state.press_result = []
        st.rerun()

    blog = result[0]['blog'] if result else None
    if blog:
        st.session_state.press_events = []
        st.session_state.press_result = []
        st.session_state.current_id = blog['id']
        st.session_state.view = 'reader'
        st.session_state.edit_mode = False
        st.rerun()


# ---------------------------------------------------------------------------
# Sidebar (the shelf)
# ---------------------------------------------------------------------------

def _render_sidebar():
    with st.sidebar:
        st.markdown('<div class="folio-kicker">FOLIO</div>', unsafe_allow_html=True)
        st.markdown('**AI blog studio**')
        if st.button('New blog', type='primary', width='stretch'):
            st.session_state.view = 'new'
            st.session_state.edit_mode = False
            st.rerun()
        st.divider()

        query = st.text_input('Search the shelf', placeholder='Title or topic\u2026', label_visibility='collapsed')
        blogs = backend.list_blogs()
        if query:
            blogs = [b for b in blogs if query.lower() in (f"{b['title']} {b['topic']}").lower()]

        if not blogs:
            st.caption('The shelf is empty. Start a press run.')
        for b in blogs:
            _sidebar_item(b)


def _sidebar_item(blog: dict):
    with st.container():
        c_title, c_dots = st.columns([6, 1], vertical_alignment='center')
        with c_title:
            if st.button(blog['title'], key=f'open_{blog["id"]}', width='stretch'):
                st.session_state.current_id = blog['id']
                st.session_state.view = 'reader'
                st.session_state.edit_mode = False
        with c_dots:
            with st.popover('\u22ee', help='Blog actions'):
                if st.button('Delete from the shelf', key=f'menu_del_{blog["id"]}', width='stretch'):
                    st.session_state.confirm_delete_id = blog['id']
                    st.rerun()


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def _render_masthead():
    st.markdown(
        '<div class="folio-kicker">AI BLOG STUDIO</div>'
        '<div class="folio-title">Folio</div>'
        '<div class="folio-sub">Long-form writing, run through an editorial press.</div>'
        '<div class="folio-rule"></div>',
        unsafe_allow_html=True,
    )


def _render_card(blog: dict):
    with st.container(border=True):
        thumb = blog['images'][0] if blog.get('images') else None
        if thumb and _image_path(thumb).exists():
            st.image(str(_image_path(thumb)), width='stretch')
        else:
            initial = (blog['title'] or '?')[0].upper()
            st.markdown(f'<div class="folio-tile">{initial}</div>', unsafe_allow_html=True)
        st.markdown(f'### {blog["title"]}')
        st.caption(_meta(blog))
        if st.button('Open \u2192', key=f'gopen_{blog["id"]}', width='stretch'):
            st.session_state.current_id = blog['id']
            st.session_state.view = 'reader'
            st.session_state.edit_mode = False


def _render_shelf():
    _render_masthead()
    blogs = backend.list_blogs()
    if not blogs:
        st.markdown('No stories on the shelf yet.')
        if st.button('Start your first press run', type='primary'):
            st.session_state.view = 'new'
            st.session_state.edit_mode = False
            st.rerun()
        return
    st.markdown(f'**{len(blogs)} story{"s" if len(blogs) != 1 else ""} on the shelf**')
    cols = st.columns(3)
    for i, blog in enumerate(blogs):
        with cols[i % 3]:
            _render_card(blog)


def _render_new():
    if st.session_state.press_running:
        _render_press_run()
        return

    st.markdown('<div class="folio-kicker">THE BRIEF</div>', unsafe_allow_html=True)
    st.markdown('### What should we write?')

    topic = st.text_area(
        'Topic',
        placeholder='e.g. Understanding self-attention in transformers',
        height=110,
        label_visibility='collapsed',
    )

    c1, c2 = st.columns(2)
    with c1:
        research = st.segmented_control(
            'Research',
            options=['Auto', 'Deep'],
            default='Auto',
            help='Auto lets the router decide; Deep always searches the web and cites sources.',
        )
        audience = st.selectbox('Audience', ['Auto', 'Developers', 'Data scientists', 'General readers', 'Beginners'])
    with c2:
        tone = st.segmented_control('Tone', options=['Practical', 'Deep dive', 'Friendly', 'Formal'], default='Practical')
        kind = st.selectbox('Blog kind', ['Auto', 'Explainer', 'Tutorial', 'Comparison', 'News roundup', 'System design'])

    ready = bool(topic.strip())
    if st.button('Start the press run', type='primary', width='stretch', disabled=not ready):
        options = {
            'research': research,
            'audience': None if audience == 'Auto' else audience,
            'tone': tone,
            'kind': None if kind == 'Auto' else kind,
        }
        _start_press(topic.strip(), options)
        st.rerun()
    if not ready:
        st.caption('Give the press a topic to set it running.')


def _render_reader():
    blog = backend.get_blog(st.session_state.current_id)
    if blog is None:
        st.session_state.current_id = None
        st.session_state.view = 'shelf'
        st.rerun()

    if st.button('\u2190 The Shelf', key='back_to_shelf'):
        st.session_state.current_id = None
        st.session_state.view = 'shelf'
        st.rerun()

    st.markdown(f'# {blog["title"]}')
    meta = _meta(blog)
    if blog.get('duration_seconds'):
        meta += f' \u00b7 {_fmt_dur(blog["duration_seconds"])}'
    st.caption(meta)
    st.markdown('<div class="folio-rule"></div>', unsafe_allow_html=True)

    c_dl, c_edit, c_regen, c_del, _spacer = st.columns([1, 1, 1, 1, 2])
    with c_dl:
        st.download_button(
            'Download.md',
            data=blog['md'],
            file_name=f"{blog['title']}.md",
            mime='text/markdown',
            width='stretch',
        )
    with c_edit:
        if st.button('Edit', width='stretch'):
            st.session_state.edit_mode = not st.session_state.edit_mode
    with c_regen:
        if st.button('Regenerate', width='stretch'):
            _start_press(blog['topic'], blog['options'])
            st.session_state.view = 'new'
            st.rerun()
    with c_del:
        if st.button('Delete', width='stretch', key='reader_delete', help='Delete this story from the shelf'):
            st.session_state.confirm_delete_id = blog['id']
            st.rerun()

    st.markdown('<div class="folio-rule"></div>', unsafe_allow_html=True)

    if st.session_state.edit_mode:
        st.markdown('**Manuscript \u2014 make your edits, then set a new revision.**')
        new_md = st.text_area('Manuscript', value=blog['md'], height=480, key=f'manuscript_{blog["id"]}')
        c_save, c_cancel = st.columns(2)
        with c_save:
            if st.button('Set revision', type='primary', width='stretch'):
                if new_md.strip():
                    updated = backend.update_blog_md(blog['id'], new_md)
                    if updated:
                        st.session_state.edit_mode = False
                        st.success(f"Revision v{updated['version']} saved.")
                        st.rerun()
                else:
                    st.warning('An empty manuscript cannot be saved.')
        with c_cancel:
            if st.button('Cancel', width='stretch'):
                st.session_state.edit_mode = False
                st.rerun()
        return

    _render_article(blog['md'])


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

_inject_css()
_render_sidebar()
_render_delete_confirm()

view = st.session_state.view
if st.session_state.press_running and view != 'new':
    view = 'new'

if view == 'reader':
    _render_reader()
elif view == 'new':
    _render_new()
else:
    _render_shelf()
