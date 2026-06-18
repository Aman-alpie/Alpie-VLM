"""Custom file uploader that bypasses Streamlit's PUT-based /_stcore/upload_file
endpoint, which is broken behind Modal's reverse proxy (Tornado 6.5 + proxy
session routing issues cause "400: Invalid session_id").

Files are read as base64 in the browser via JavaScript FileReader and sent
through the WebSocket using Streamlit's component postMessage protocol.
"""

import streamlit.components.v1 as components
import os
import base64
import io

from PIL import Image

_component_func = components.declare_component(
    "base64_uploader", url="/app/static/base64_uploader/index.html"
)


def base64_uploader(label, accept=None, multiple=True, key=None):
    """Drop-in replacement for st.file_uploader that works behind reverse proxies.

    Returns a list of dicts: [{"name": str, "type": str, "size": int, "data": str}, ...]
    or None if no files have been uploaded yet.
    """
    if accept is None:
        accept = "*"
    elif isinstance(accept, list):
        accept = ",".join(
            ext if ext.startswith(".") else f".{ext}" for ext in accept
        )

    result = _component_func(
        label=label,
        accept=accept,
        multiple=multiple,
        key=key,
        default=None,
    )
    return result


def uploaded_files_to_images(upload_data):
    """Convert base64 upload data to a list of (PIL.Image, filename) tuples."""
    images = []
    if not upload_data:
        return images
    for file_info in upload_data:
        try:
            b64_str = file_info["data"]
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            raw = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            images.append((img, file_info.get("name", "unknown")))
        except Exception:
            pass
    return images


def uploaded_file_to_bytes(file_info):
    """Convert a single base64 file dict to raw bytes."""
    if not file_info:
        return None
    b64_str = file_info["data"]
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    return base64.b64decode(b64_str)
