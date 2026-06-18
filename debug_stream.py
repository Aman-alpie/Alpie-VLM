"""Quick debug script to inspect raw SSE chunks from the API."""
import requests
import json

API_URL = "http://20.245.200.125:8000/v1/chat/completions"
MODEL = "/home/azureuser/.cache/huggingface/hub/models--169Pi--Alpie_learn_sft_merged"

payload = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "What is 2+2? Explain briefly."}],
    "max_tokens": 512,
    "temperature": 0.7,
    "stream": True,
    "chat_template_kwargs": {"enable_thinking": True},
}

print("=== Sending request with enable_thinking=True, stream=True ===\n")
resp = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"}, stream=True, timeout=60)
resp.raise_for_status()

chunk_count = 0
for line in resp.iter_lines(decode_unicode=True):
    if not line or not line.startswith("data: "):
        continue
    data_str = line[6:]
    if data_str.strip() == "[DONE]":
        print("\n--- [DONE] ---")
        break
    chunk = json.loads(data_str)
    delta = chunk.get("choices", [{}])[0].get("delta", {})
    # Print first 10 chunks in full, then just the keys
    chunk_count += 1
    if chunk_count <= 15:
        print(f"Chunk {chunk_count}: delta keys={list(delta.keys())}  delta={json.dumps(delta)}")
    elif chunk_count == 16:
        print("... (truncating further chunks, showing summary at end) ...")

# Also do a non-streaming request to see the full structure
print("\n\n=== Non-streaming request ===\n")
payload["stream"] = False
resp2 = requests.post(API_URL, json=payload, headers={"Content-Type": "application/json"}, timeout=60)
result = resp2.json()
msg = result.get("choices", [{}])[0].get("message", {})
print(f"message keys: {list(msg.keys())}")
print(f"message.role: {msg.get('role')}")
print(f"message.reasoning_content: {repr(msg.get('reasoning_content', 'NOT PRESENT')[:200])}")
print(f"message.content (first 300 chars): {repr(msg.get('content', '')[:300])}")
