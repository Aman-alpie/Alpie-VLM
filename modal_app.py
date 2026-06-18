import os
import subprocess
import modal

# Create Modal App
app = modal.App("alpie-vision-tester")

# Define container image for Streamlit frontend
streamlit_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("curl", "unzip", "libgl1-mesa-glx", "libglib2.0-0", "nodejs", "npm")
    # Install Deno to allow yt-dlp to run YouTube token decryption JS
    .run_commands("curl -fsSL https://deno.land/install.sh | sh")
    .env({
        "PATH": "/root/.deno/bin:/usr/local/bin:/usr/bin:/bin",
        # Bake Streamlit settings into the image — these take highest priority
        # and are available before Streamlit even starts.
        "STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION": "false",
        "STREAMLIT_SERVER_ENABLE_CORS": "false",
        "STREAMLIT_SERVER_MAX_UPLOAD_SIZE": "2048",
        "STREAMLIT_SERVER_MAX_MESSAGE_SIZE": "2048",
        "STREAMLIT_SERVER_HEADLESS": "true",
        "STREAMLIT_SERVER_FILE_WATCHER_TYPE": "none",
        "STREAMLIT_SERVER_RUN_ON_SAVE": "false",
        "STREAMLIT_BROWSER_GATHER_USAGE_STATS": "false",
        "STREAMLIT_SERVER_ENABLE_STATIC_SERVING": "true",
    })
    # Install curl_cffi, latest yt-dlp from master branch, and bgutil pot provider for YouTube bot bypass
    .pip_install(
        "curl_cffi",
        "https://github.com/yt-dlp/yt-dlp/archive/master.tar.gz",
        "bgutil-ytdlp-pot-provider",
    )
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/root")
)

# Define container image for vLLM VLM server
# Note: Qwen2.5-VL models require vLLM >= 0.7.2 and transformers >= 4.49.0
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-base-ubuntu22.04", add_python="3.10")
    .pip_install(
        "vllm>=0.7.2",
        "transformers>=4.49.0",
        "qwen-vl-utils",
        "huggingface_hub",
        "hf-transfer",
        "protobuf"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

# ── VLM GPU Server Endpoint ──────────────────────────────────────────────────
@app.function(
    image=vllm_image,
    gpu="A100",  # Using A100 (40GB or 80GB) to support large reasoning model weights
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
    scaledown_window=300,
)
@modal.web_server(port=8000)
def vllm_server():
    """Start vLLM serving the private Alpie VLM model."""
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", "169Pi/Alpie_learn_sft_merged",
        "--port", "8000",
        "--host", "0.0.0.0",
        "--trust-remote-code",
        "--max-model-len", "65536"
    ]
    subprocess.run(cmd)

# ── Streamlit Web Application ────────────────────────────────────────────────
@app.function(
    image=streamlit_image,
    secrets=[
        modal.Secret.from_name("huggingface"),
        # To make cookies available to everyone without putting cookies.txt in Git:
        # 1. Create a Modal secret named `youtube-cookies` on the Modal dashboard with key `YOUTUBE_COOKIES`.
        modal.Secret.from_name("youtube-cookies"),
    ],
    scaledown_window=600,  # Keep container alive 10 min to prevent session loss
)
@modal.web_server(port=8501)
def run_streamlit():
    """Start the Streamlit application interface."""
    # Attempt to dynamically resolve the deployed vLLM server web URL
    try:
        from modal import Function
        try:
            vllm_func = Function.from_name("alpie-vision-tester", "vllm_server")
        except AttributeError:
            vllm_func = Function.lookup("alpie-vision-tester", "vllm_server")
            
        try:
            vllm_url = vllm_func.web_url
        except AttributeError:
            vllm_url = vllm_func.get_web_url()

        print(f"Auto-resolved vLLM web URL: {vllm_url}")
        # Inject API_URL into the Streamlit subprocess environment
        env = {**os.environ, "API_URL": f"{vllm_url}/v1/chat/completions"}
    except Exception as e:
        print(f"Could not auto-resolve local vLLM URL: {e}")
        env = dict(os.environ)

    # All server settings (XSRF, CORS, upload size, etc.) are baked into the
    # Docker image env vars — no need to duplicate them here.  CLI flags are
    # kept as an additional safety net.
    subprocess.Popen(
        "streamlit run app.py "
        "--server.port 8501 "
        "--server.address 0.0.0.0 "
        "--server.enableCORS false "
        "--server.enableXsrfProtection false "
        "--server.maxUploadSize 2048 "
        "--server.maxMessageSize 2048 "
        "--server.headless true "
        "--server.fileWatcherType none "
        "--server.runOnSave false "
        "--browser.gatherUsageStats false "
        "--server.enableStaticServing true",
        shell=True,
        env=env,
        cwd="/root",
    )

