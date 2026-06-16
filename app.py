"""
Alpie Vision Tester — Streamlit app for testing the vision endpoint.
Supports image & video uploads via the OpenAI-compatible chat/completions API.
"""

import base64
import io
import json
import os
import tempfile
import time

import cv2
import requests
import streamlit as st
import yt_dlp
from PIL import Image

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Alpie · Vision Tester",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    :root {
        --bg-primary: #0a0a0f;
        --bg-card: rgba(18, 18, 28, 0.85);
        --bg-glass: rgba(255,255,255,0.04);
        --accent-1: #7c3aed;
        --accent-2: #06b6d4;
        --accent-3: #f472b6;
        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --border: rgba(255,255,255,0.08);
        --glow: 0 0 30px rgba(124, 58, 237, .25);
    }

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif !important;
        background: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f0f1a 0%, #141425 100%) !important;
        border-right: 1px solid var(--border) !important;
    }

    /* Hero */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1.5rem;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--accent-1), var(--accent-2), var(--accent-3));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -1px;
    }
    .hero p {
        color: var(--text-secondary);
        font-size: 1.05rem;
        margin-top: .4rem;
        font-weight: 400;
    }

    /* Section label */
    .section-label {
        font-weight: 600;
        font-size: .85rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: var(--accent-2);
        margin-bottom: .5rem;
    }

    /* Think block */
    .think-block {
        background: rgba(124, 58, 237, .08);
        border-left: 3px solid var(--accent-1);
        border-radius: 0 10px 10px 0;
        padding: 1rem 1.2rem;
        font-size: .92rem;
        line-height: 1.7;
        color: #c4b5fd;
        margin-bottom: 1rem;
        white-space: pre-wrap;
    }

    /* Answer block */
    .answer-block {
        background: var(--bg-glass);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        font-size: 1rem;
        line-height: 1.8;
        color: var(--text-primary);
        box-shadow: var(--glow);
        white-space: pre-wrap;
    }

    /* Stat cards */
    .stat-row {
        display: flex;
        gap: .8rem;
        margin-top: .8rem;
        flex-wrap: wrap;
    }
    .stat-card {
        flex: 1;
        min-width: 120px;
        background: var(--bg-glass);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: .6rem .9rem;
        text-align: center;
    }
    .stat-card .val {
        font-size: 1.15rem;
        font-weight: 700;
        background: linear-gradient(90deg, var(--accent-2), var(--accent-1));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .stat-card .lbl {
        font-size: .7rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 2px;
    }

    /* Tabs */
    [data-testid="stTabs"] button {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: .95rem !important;
        color: var(--text-secondary) !important;
        border-bottom: 2px solid transparent !important;
        transition: all .2s ease !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: var(--accent-2) !important;
        border-bottom: 2px solid var(--accent-2) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-1), var(--accent-2)) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: .55rem 1.4rem !important;
        transition: transform .15s, box-shadow .15s !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 20px rgba(124,58,237,.35) !important;
    }

    /* Text area */
    textarea {
        background: var(--bg-glass) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--border) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        transition: border-color .2s !important;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-1) !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(124,58,237,.35); border-radius: 3px; }

    /* Image preview */
    .preview-grid {
        display: flex;
        gap: .6rem;
        flex-wrap: wrap;
        margin: .8rem 0;
    }
    .preview-grid img {
        border-radius: 10px;
        border: 1px solid var(--border);
        max-height: 200px;
        object-fit: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ────────────────────────────────────────────────────────────────

API_URL = "http://20.245.200.125:8000/v1/chat/completions"
MAX_FRAMES = 8

# Auto-detect model name from the server
if "model_name" not in st.session_state:
    st.session_state.model_name = ""
    # Try to auto-fetch on first load
    try:
        _r = requests.get(
            API_URL.replace("/v1/chat/completions", "/v1/models"),
            timeout=5,
        )
        if _r.status_code == 200:
            _data = _r.json()
            _models = _data.get("data", [])
            if _models:
                st.session_state.model_name = _models[0].get("id", "")
    except Exception:
        pass

# ── Helpers ──────────────────────────────────────────────────────────────────


def image_to_base64_url(img: Image.Image, fmt: str = "PNG") -> str:
    """Convert a PIL Image to a base64 data-URL string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode()
    mime = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{b64}"


def download_youtube_video(url: str) -> str | None:
    """Download lowest resolution stream of a YouTube video to a temp file."""
    temp_dir = tempfile.gettempdir()
    output_template = os.path.join(temp_dir, 'yt_download_%(id)s.%(ext)s')
    
    ydl_opts = {
        'format': 'worst[ext=mp4]/lowest[ext=mp4]/worst/lowest',
        'outtmpl': output_template,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)
            if os.path.exists(filepath):
                return filepath
            base_path = os.path.splitext(filepath)[0]
            for ext in ['mp4', 'mkv', 'webm', '3gp', 'flv']:
                p = f"{base_path}.{ext}"
                if os.path.exists(p):
                    return p
    except Exception as e:
        st.error(f"❌ Failed to download YouTube video: {e}")
    return None


def extract_frames(path: str, n: int = MAX_FRAMES) -> list[Image.Image]:
    """Extract *n* evenly-spaced frames from a video file."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []
    idxs = [int(i * total / n) for i in range(n)]
    frames: list[Image.Image] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames


def parse_output(raw: str) -> tuple[str | None, str]:
    """Split <think>…</think> from the final answer."""
    raw = raw.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if "<think>" in raw and "</think>" in raw:
        think = raw.split("<think>")[1].split("</think>")[0].strip()
        answer = raw.split("</think>")[-1].strip()
        return think or None, answer
    return None, raw


def call_api(
    messages: list[dict],
    model: str = "",
    max_tokens: int = 8192,
    temperature: float = 0.7,
    stream: bool = True,
) -> requests.Response | dict:
    """
    Call the OpenAI-compatible chat/completions endpoint.
    Returns a streaming Response when stream=True, or parsed JSON otherwise.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": max(temperature, 0.01),
        "stream": stream,
    }
    headers = {"Content-Type": "application/json"}

    if stream:
        resp = requests.post(API_URL, json=payload, headers=headers, stream=True, timeout=300)
        resp.raise_for_status()
        return resp
    else:
        resp = requests.post(API_URL, json=payload, headers=headers, timeout=300)
        resp.raise_for_status()
        return resp.json()


def stream_response(resp: requests.Response):
    """Yield incremental text from an SSE streaming response."""
    full_text = ""
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        if line.startswith("data: "):
            data_str = line[6:]
            if data_str.strip() == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                token = delta.get("content", "")
                if token:
                    full_text += token
                    yield full_text
            except json.JSONDecodeError:
                continue
    yield full_text


def build_image_messages(prompt: str, images: list[Image.Image]) -> list[dict]:
    """Build a multimodal user message with text + images."""
    content: list[dict] = []
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_base64_url(img)},
        })
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def build_video_messages(prompt: str, frames: list[Image.Image]) -> list[dict]:
    """Build a multimodal message from extracted video frames."""
    content: list[dict] = []
    for i, frame in enumerate(frames):
        content.append({
            "type": "image_url",
            "image_url": {"url": image_to_base64_url(frame)},
        })
    content.append({
        "type": "text",
        "text": f"[These are {len(frames)} frames extracted from a video]\n\n{prompt}",
    })
    return [{"role": "user", "content": content}]


def render_stream(messages: list[dict], max_tokens: int, temperature: float, model: str = ""):
    """Call the API with streaming and render results live."""
    think_label_ph = st.empty()
    think_ph = st.empty()
    ans_label_ph = st.empty()
    ans_ph = st.empty()
    status = st.empty()

    status.markdown(
        '<div class="section-label">⏳ Generating…</div>',
        unsafe_allow_html=True,
    )

    t0 = time.time()

    try:
        resp = call_api(messages, model=model, max_tokens=max_tokens, temperature=temperature, stream=True)
        final_text = ""
        token_count = 0

        for full_text in stream_response(resp):
            final_text = full_text
            token_count += 1
            thinking, answer = parse_output(full_text)

            if thinking:
                think_label_ph.markdown(
                    '<div class="section-label">🔮 Chain of Thought</div>',
                    unsafe_allow_html=True,
                )
                think_ph.markdown(
                    f'<div class="think-block">{thinking}</div>',
                    unsafe_allow_html=True,
                )

            if answer:
                ans_label_ph.markdown(
                    '<div class="section-label">✦ Answer</div>',
                    unsafe_allow_html=True,
                )
                ans_ph.markdown(
                    f'<div class="answer-block">{answer}</div>',
                    unsafe_allow_html=True,
                )

        elapsed = time.time() - t0
        status.empty()

        # Stats bar
        st.markdown(
            f"""
            <div class="stat-row">
                <div class="stat-card">
                    <div class="val">{elapsed:.1f}s</div>
                    <div class="lbl">Latency</div>
                </div>
                <div class="stat-card">
                    <div class="val">{len(final_text)}</div>
                    <div class="lbl">Characters</div>
                </div>
                <div class="stat-card">
                    <div class="val">{token_count}</div>
                    <div class="lbl">Chunks</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except requests.exceptions.ConnectionError:
        status.empty()
        st.error("❌ Cannot connect to the API server. Is it running?")
    except requests.exceptions.Timeout:
        status.empty()
        st.error("❌ Request timed out after 300 seconds.")
    except requests.exceptions.HTTPError as e:
        status.empty()
        st.error(f"❌ HTTP Error: {e.response.status_code} — {e.response.text[:500]}")
    except Exception as e:
        status.empty()
        st.error(f"❌ {type(e).__name__}: {e}")


def render_non_stream(messages: list[dict], max_tokens: int, temperature: float, model: str = ""):
    """Call the API without streaming and render the full result."""
    status = st.empty()
    status.markdown(
        '<div class="section-label">⏳ Generating…</div>',
        unsafe_allow_html=True,
    )

    t0 = time.time()

    try:
        result = call_api(messages, model=model, max_tokens=max_tokens, temperature=temperature, stream=False)
        elapsed = time.time() - t0
        status.empty()

        raw = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage = result.get("usage", {})
        thinking, answer = parse_output(raw)

        if thinking:
            st.markdown(
                '<div class="section-label">🔮 Chain of Thought</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="think-block">{thinking}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<div class="section-label">✦ Answer</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="answer-block">{answer}</div>',
            unsafe_allow_html=True,
        )

        # Stats
        prompt_tok = usage.get("prompt_tokens", "—")
        compl_tok = usage.get("completion_tokens", "—")
        st.markdown(
            f"""
            <div class="stat-row">
                <div class="stat-card">
                    <div class="val">{elapsed:.1f}s</div>
                    <div class="lbl">Latency</div>
                </div>
                <div class="stat-card">
                    <div class="val">{prompt_tok}</div>
                    <div class="lbl">Prompt Tokens</div>
                </div>
                <div class="stat-card">
                    <div class="val">{compl_tok}</div>
                    <div class="lbl">Completion Tokens</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except requests.exceptions.ConnectionError:
        status.empty()
        st.error("❌ Cannot connect to the API server. Is it running?")
    except requests.exceptions.Timeout:
        status.empty()
        st.error("❌ Request timed out after 300 seconds.")
    except requests.exceptions.HTTPError as e:
        status.empty()
        st.error(f"❌ HTTP Error: {e.response.status_code} — {e.response.text[:500]}")
    except Exception as e:
        status.empty()
        st.error(f"❌ {type(e).__name__}: {e}")


# ── Hero ─────────────────────────────────────────────────────────────────────

st.markdown(
    '<div class="hero">'
    "<h1>🧠 ALPIE</h1>"
    "<p>Vision Tester · Image · Video · Multimodal Reasoning</p>"
    "</div>",
    unsafe_allow_html=True,
)

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<div class="section-label">🔗 Endpoint</div>',
        unsafe_allow_html=True,
    )
    st.code(API_URL, language=None)

    st.markdown("---")

    st.markdown(
        '<div class="section-label">🤖 Model</div>',
        unsafe_allow_html=True,
    )
    model_name = st.text_input(
        "Model name",
        value=st.session_state.model_name,
        placeholder="e.g. Qwen/Qwen2.5-VL-7B-Instruct",
        label_visibility="collapsed",
        help="The model ID served by vLLM. Click 'Check API Health' to auto-detect.",
    )
    st.session_state.model_name = model_name

    st.markdown("---")

    st.markdown(
        '<div class="section-label">⚙ Settings</div>',
        unsafe_allow_html=True,
    )

    use_streaming = st.toggle(
        "🌊 Stream response",
        value=True,
        help="Stream tokens as they are generated",
    )

    st.markdown("---")

    st.markdown(
        '<div class="section-label">🎛 Generation</div>',
        unsafe_allow_html=True,
    )

    max_tokens = st.slider("Max New Tokens", 256, 32768, 8192, 256)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.7, 0.05)

    st.markdown("---")

    st.markdown(
        '<div class="section-label">🎬 Video</div>',
        unsafe_allow_html=True,
    )
    max_frames = st.slider("Max Frames to Extract", 1, 16, MAX_FRAMES, 1)

    st.markdown("---")

    # Health check
    if st.button("🩺 Check API Health", use_container_width=True):
        try:
            r = requests.get(
                API_URL.replace("/v1/chat/completions", "/v1/models"),
                timeout=10,
            )
            if r.status_code == 200:
                models_data = r.json()
                st.success("✅ API is online!")
                st.json(models_data, expanded=False)
                # Auto-fill model name if empty
                available = models_data.get("data", [])
                if available and not model_name.strip():
                    st.session_state.model_name = available[0].get("id", "")
                    st.rerun()
            else:
                st.warning(f"⚠ Status {r.status_code}")
        except Exception as e:
            st.error(f"❌ {e}")

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_image, tab_video, tab_text = st.tabs(
    ["🖼 Image", "🎬 Video", "💬 Text-Only"]
)

# ── Image Tab ────────────────────────────────────────────────────────────────

with tab_image:
    st.markdown(
        '<div class="section-label">Upload Images</div>',
        unsafe_allow_html=True,
    )

    uploaded_images = st.file_uploader(
        "Drop images here",
        type=["png", "jpg", "jpeg", "webp", "bmp", "gif"],
        accept_multiple_files=True,
        key="img_upload",
        label_visibility="collapsed",
    )

    if uploaded_images:
        cols = st.columns(min(len(uploaded_images), 4))
        pil_images: list[Image.Image] = []
        for i, f in enumerate(uploaded_images):
            img = Image.open(f).convert("RGB")
            pil_images.append(img)
            with cols[i % len(cols)]:
                st.image(img, caption=f.name, use_container_width=True)

    img_prompt = st.text_area(
        "Image prompt",
        placeholder="Describe what you want to know about the image(s)…",
        height=100,
        key="img_prompt",
        label_visibility="collapsed",
    )

    if st.button("🔍 Analyze Image(s)", key="img_run", use_container_width=True):
        if not uploaded_images:
            st.warning("⚠ Please upload at least one image.")
        elif not img_prompt.strip():
            st.warning("⚠ Please enter a prompt.")
        else:
            messages = build_image_messages(img_prompt, pil_images)
            if use_streaming:
                render_stream(messages, max_tokens, temperature, model=model_name)
            else:
                render_non_stream(messages, max_tokens, temperature, model=model_name)

# ── Video Tab ────────────────────────────────────────────────────────────────

with tab_video:
    video_source = st.radio(
        "Select Video Source",
        ["📁 Upload Video File", "🔗 YouTube URL"],
        horizontal=True,
        key="vid_source",
    )

    uploaded_video = None
    youtube_url = ""

    if video_source == "📁 Upload Video File":
        st.markdown(
            '<div class="section-label">Upload Video</div>',
            unsafe_allow_html=True,
        )
        uploaded_video = st.file_uploader(
            "Drop a video here",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            accept_multiple_files=False,
            key="vid_upload",
            label_visibility="collapsed",
        )
        if uploaded_video:
            if uploaded_video.size > 500 * 1024 * 1024:
                st.error("❌ File must be 500 MB or smaller for video")
                uploaded_video = None
            else:
                st.video(uploaded_video)
    else:
        st.markdown(
            '<div class="section-label">YouTube URL</div>',
            unsafe_allow_html=True,
        )
        youtube_url = st.text_input(
            "Paste YouTube Video URL",
            placeholder="e.g. https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            key="yt_url",
            label_visibility="collapsed",
        )
        if youtube_url.strip():
            if "youtube.com" in youtube_url or "youtu.be" in youtube_url:
                try:
                    st.video(youtube_url)
                except Exception as e:
                    st.error(f"❌ Failed to load video: {e}")
            else:
                st.warning("⚠ Please enter a valid YouTube video URL.")

    vid_prompt = st.text_area(
        "Video prompt",
        placeholder="What do you want to know about this video?",
        height=100,
        key="vid_prompt",
        label_visibility="collapsed",
    )

    if st.button("🎬 Analyze Video", key="vid_run", use_container_width=True):
        if video_source == "📁 Upload Video File" and not uploaded_video:
            st.warning("⚠ Please upload a video.")
        elif video_source == "🔗 YouTube URL" and not youtube_url.strip():
            st.warning("⚠ Please enter a YouTube URL.")
        elif not vid_prompt.strip():
            st.warning("⚠ Please enter a prompt.")
        else:
            frames = []
            tmp_path = None
            
            if video_source == "📁 Upload Video File":
                with st.spinner("Preparing uploaded video…"):
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".mp4"
                    ) as tmp:
                        tmp.write(uploaded_video.read())
                        tmp_path = tmp.name
            else:
                with st.spinner("Downloading YouTube video (lowest resolution)…"):
                    tmp_path = download_youtube_video(youtube_url)

            if tmp_path:
                with st.spinner(f"Extracting up to {max_frames} frames…"):
                    frames = extract_frames(tmp_path, max_frames)
                
                # Clean up temporary video file
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

            if not frames:
                st.error("❌ Could not extract any frames from the video.")
            else:
                st.markdown(
                    f'<div class="section-label">📸 Extracted {len(frames)} Frames</div>',
                    unsafe_allow_html=True,
                )
                frame_cols = st.columns(min(len(frames), 4))
                for i, frame in enumerate(frames):
                    with frame_cols[i % len(frame_cols)]:
                        st.image(frame, caption=f"Frame {i+1}", use_container_width=True)

                messages = build_video_messages(vid_prompt, frames)
                if use_streaming:
                    render_stream(messages, max_tokens, temperature, model=model_name)
                else:
                    render_non_stream(messages, max_tokens, temperature, model=model_name)

# ── Text Tab ─────────────────────────────────────────────────────────────────

with tab_text:
    st.markdown(
        '<div class="section-label">Text Prompt</div>',
        unsafe_allow_html=True,
    )

    text_prompt = st.text_area(
        "Text prompt",
        placeholder="Ask Alpie anything (text-only, no vision)…",
        height=130,
        key="text_prompt",
        label_visibility="collapsed",
    )

    if st.button("🚀 Run", key="text_run", use_container_width=True):
        if not text_prompt.strip():
            st.warning("⚠ Please enter a prompt.")
        else:
            messages = [{"role": "user", "content": text_prompt}]
            if use_streaming:
                render_stream(messages, max_tokens, temperature, model=model_name)
            else:
                render_non_stream(messages, max_tokens, temperature, model=model_name)

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    '<p style="text-align:center; color: var(--text-secondary); font-size:.8rem;">'
    "Alpie Vision Tester · Powered by 169Pi · "
    f'Endpoint: <code>{API_URL}</code></p>',
    unsafe_allow_html=True,
)
