"""Serve a small browser chat UI for the MiniLLaVA vLLM OpenAI server.

Start `vllm_openai_server.py` first, then run this script and open the printed
URL in a browser. The UI accepts image uploads and sends OpenAI-compatible
multimodal chat requests through this local proxy.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import mimetypes
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_UPLOAD_DIR = ROOT / "week18_minillava_hf_vllm" / "outputs" / "chat_uploads"
DEFAULT_VLLM_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "minillava"


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MiniLLaVA Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --panel-soft: #eef2f6;
      --text: #17202a;
      --muted: #667085;
      --line: #d8dee7;
      --accent: #1f7a5c;
      --accent-strong: #155f47;
      --danger: #b42318;
      --shadow: 0 16px 40px rgba(18, 31, 53, 0.10);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }

    .app {
      min-height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }

    header {
      height: 64px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 0 24px;
      background: rgba(255, 255, 255, 0.92);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(12px);
      position: sticky;
      top: 0;
      z-index: 10;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .mark {
      width: 34px;
      height: 34px;
      display: grid;
      place-items: center;
      border-radius: 8px;
      background: #173f35;
      color: #ffffff;
      font-weight: 700;
      flex: 0 0 auto;
    }

    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 700;
    }

    .status {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }

    .dot {
      width: 9px;
      height: 9px;
      border-radius: 999px;
      background: #98a2b3;
    }

    .dot.ready { background: #16a34a; }
    .dot.busy { background: #d97706; }
    .dot.error { background: var(--danger); }

    main {
      width: min(1040px, 100%);
      margin: 0 auto;
      padding: 28px 18px 22px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }

    .messages {
      display: flex;
      flex-direction: column;
      gap: 16px;
    }

    .empty {
      min-height: 44vh;
      display: grid;
      place-items: center;
      color: var(--muted);
      text-align: center;
      padding: 32px 16px;
    }

    .empty strong {
      display: block;
      color: var(--text);
      font-size: clamp(24px, 4vw, 40px);
      letter-spacing: 0;
      margin-bottom: 10px;
    }

    .msg {
      display: grid;
      grid-template-columns: 42px minmax(0, 1fr);
      gap: 12px;
      align-items: start;
    }

    .avatar {
      width: 42px;
      height: 42px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: #243447;
      color: #ffffff;
      font-weight: 700;
      font-size: 13px;
    }

    .msg.user .avatar { background: var(--accent); }

    .bubble {
      max-width: 820px;
      width: fit-content;
      padding: 13px 15px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      box-shadow: 0 4px 18px rgba(18, 31, 53, 0.04);
      line-height: 1.65;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }

    .msg.user .bubble {
      background: #eef7f3;
      border-color: #b9dfd1;
    }

    .thumb {
      width: min(320px, 100%);
      max-height: 260px;
      object-fit: contain;
      display: block;
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 10px;
      background: #ffffff;
    }

    .composer-wrap {
      position: sticky;
      bottom: 0;
      background: linear-gradient(180deg, rgba(246, 247, 249, 0), var(--bg) 18px);
      padding: 22px 18px 18px;
    }

    .composer {
      width: min(1040px, 100%);
      margin: 0 auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      padding: 12px;
      display: grid;
      grid-template-columns: auto 1fr auto;
      gap: 10px;
      align-items: end;
    }

    .file-btn,
    .send-btn,
    .clear-btn {
      border: 1px solid var(--line);
      background: var(--panel-soft);
      color: var(--text);
      border-radius: 8px;
      height: 42px;
      min-width: 42px;
      display: inline-grid;
      place-items: center;
      cursor: pointer;
      font: inherit;
    }

    .send-btn {
      min-width: 88px;
      padding: 0 16px;
      border-color: var(--accent);
      background: var(--accent);
      color: #ffffff;
      font-weight: 700;
    }

    .send-btn:disabled,
    .file-btn:disabled,
    .clear-btn:disabled {
      cursor: not-allowed;
      opacity: 0.55;
    }

    .send-btn:not(:disabled):hover { background: var(--accent-strong); }
    .file-btn:hover, .clear-btn:hover { background: #e4e9f0; }

    .input-area {
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }

    textarea {
      width: 100%;
      max-height: 180px;
      min-height: 42px;
      resize: none;
      border: 0;
      outline: 0;
      font: inherit;
      line-height: 1.5;
      color: var(--text);
      background: transparent;
      padding: 8px 2px;
    }

    .preview {
      display: none;
      align-items: center;
      gap: 10px;
      min-width: 0;
      color: var(--muted);
      font-size: 13px;
    }

    .preview.active { display: flex; }

    .preview img {
      width: 52px;
      height: 52px;
      object-fit: cover;
      border-radius: 8px;
      border: 1px solid var(--line);
      flex: 0 0 auto;
    }

    .preview span {
      min-width: 0;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }

    input[type="file"] {
      position: absolute;
      inline-size: 1px;
      block-size: 1px;
      opacity: 0;
      pointer-events: none;
    }

    @media (max-width: 640px) {
      header { padding: 0 14px; }
      h1 { font-size: 16px; }
      .status span { display: none; }
      main { padding-inline: 12px; }
      .msg { grid-template-columns: 34px minmax(0, 1fr); }
      .avatar { width: 34px; height: 34px; font-size: 12px; }
      .bubble { width: 100%; }
      .composer-wrap { padding-inline: 10px; }
      .composer { grid-template-columns: auto minmax(0, 1fr) auto; padding: 9px; }
      .send-btn { min-width: 54px; padding-inline: 12px; }
      .send-label { display: none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header>
      <div class="brand">
        <div class="mark">ML</div>
        <h1>MiniLLaVA Chat</h1>
      </div>
      <div class="status" aria-live="polite">
        <span id="status-dot" class="dot ready"></span>
        <span id="status-text">Ready</span>
      </div>
    </header>

    <main>
      <section id="messages" class="messages" aria-live="polite">
        <div id="empty" class="empty">
          <div>
            <strong>上传图片并开始对话</strong>
            <div>支持单张图片，后续消息会保留在当前对话上下文中。</div>
          </div>
        </div>
      </section>
    </main>

    <div class="composer-wrap">
      <form id="composer" class="composer">
        <label class="file-btn" title="上传图片" aria-label="上传图片">
          <input id="image-input" type="file" accept="image/*">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M17 8 12 3 7 8" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M12 3v12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        </label>
        <div class="input-area">
          <div id="preview" class="preview">
            <img id="preview-img" alt="">
            <span id="preview-name"></span>
            <button id="clear-image" class="clear-btn" type="button" title="移除图片" aria-label="移除图片">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="M18 6 6 18M6 6l12 12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
              </svg>
            </button>
          </div>
          <textarea id="prompt" rows="1" placeholder="输入问题，Enter 发送，Shift+Enter 换行"></textarea>
        </div>
        <button id="send" class="send-btn" type="submit">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="m22 2-7 20-4-9-9-4Z" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/>
            <path d="M22 2 11 13" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
          </svg>
          <span class="send-label">发送</span>
        </button>
      </form>
    </div>
  </div>

  <script>
    const messagesEl = document.querySelector("#messages");
    const emptyEl = document.querySelector("#empty");
    const form = document.querySelector("#composer");
    const promptEl = document.querySelector("#prompt");
    const imageInput = document.querySelector("#image-input");
    const preview = document.querySelector("#preview");
    const previewImg = document.querySelector("#preview-img");
    const previewName = document.querySelector("#preview-name");
    const clearImage = document.querySelector("#clear-image");
    const sendBtn = document.querySelector("#send");
    const statusDot = document.querySelector("#status-dot");
    const statusText = document.querySelector("#status-text");

    let activeImage = null;
    let pendingImage = null;
    let isSending = false;

    function setStatus(text, state) {
      statusText.textContent = text;
      statusDot.className = `dot ${state || "ready"}`;
    }

    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, (ch) => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;",
      }[ch]));
    }

    function scrollToBottom() {
      requestAnimationFrame(() => window.scrollTo({ top: document.body.scrollHeight, behavior: "smooth" }));
    }

    function autoSize() {
      promptEl.style.height = "auto";
      promptEl.style.height = `${Math.min(promptEl.scrollHeight, 180)}px`;
    }

    function renderMessage(role, text, imageDataUrl) {
      emptyEl.style.display = "none";
      const row = document.createElement("article");
      row.className = `msg ${role}`;
      const who = role === "user" ? "我" : "AI";
      row.innerHTML = `
        <div class="avatar">${who}</div>
        <div class="bubble">
          ${imageDataUrl ? `<img class="thumb" src="${imageDataUrl}" alt="uploaded image">` : ""}
          <div>${escapeHtml(text || "")}</div>
        </div>
      `;
      messagesEl.appendChild(row);
      scrollToBottom();
      return row.querySelector(".bubble div");
    }

    function fileToDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });
    }

    function clearPendingImage() {
      pendingImage = null;
      imageInput.value = "";
      preview.classList.remove("active");
      previewImg.removeAttribute("src");
      previewName.textContent = "";
    }

    imageInput.addEventListener("change", async () => {
      const file = imageInput.files && imageInput.files[0];
      if (!file) return;
      pendingImage = {
        name: file.name,
        dataUrl: await fileToDataUrl(file),
      };
      previewImg.src = pendingImage.dataUrl;
      previewName.textContent = file.name;
      preview.classList.add("active");
    });

    clearImage.addEventListener("click", clearPendingImage);

    promptEl.addEventListener("input", autoSize);
    promptEl.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        form.requestSubmit();
      }
    });

    async function sendMessage(event) {
      event.preventDefault();
      if (isSending) return;

      const text = promptEl.value.trim();
      if (!text && !pendingImage) return;

      const imageForTurn = pendingImage;
      if (imageForTurn) {
        activeImage = imageForTurn;
      }
      const imageForRequest = imageForTurn || activeImage;
      const asksAboutImage = /图|图片|照片|image|photo|picture/i.test(text);
      if (asksAboutImage && !imageForRequest) {
        renderMessage("user", text, null);
        renderMessage("assistant", "请先上传一张图片，再让我描述。", null);
        promptEl.value = "";
        autoSize();
        promptEl.focus();
        return;
      }
      const content = [];
      if (text) {
        content.push({ type: "text", text });
      }
      if (imageForRequest) {
        content.push({
          type: "image_url",
          image_url: { url: imageForRequest.dataUrl },
        });
      }

      const userMessage = {
        role: "user",
        content: content.length === 1 && content[0].type === "text" ? text : content,
      };
      renderMessage("user", text, imageForTurn && imageForTurn.dataUrl);

      promptEl.value = "";
      autoSize();
      clearPendingImage();
      isSending = true;
      sendBtn.disabled = true;
      imageInput.disabled = true;
      setStatus("Generating", "busy");

      const assistantTarget = renderMessage("assistant", "思考中...", null);

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: [userMessage] }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || `HTTP ${response.status}`);
        }
        const answer = data.choices?.[0]?.message?.content || "";
        assistantTarget.textContent = answer;
        setStatus("Ready", "ready");
      } catch (error) {
        assistantTarget.textContent = `请求失败：${error.message}`;
        setStatus("Error", "error");
      } finally {
        isSending = false;
        sendBtn.disabled = false;
        imageInput.disabled = false;
        promptEl.focus();
        scrollToBottom();
      }
    }

    form.addEventListener("submit", sendMessage);
    promptEl.focus();
  </script>
</body>
</html>
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Serve a web chat UI for MiniLLaVA vLLM.")
    parser.add_argument("--host", default="127.0.0.1", help="UI server host.")
    parser.add_argument("--port", type=int, default=7860, help="UI server port.")
    parser.add_argument("--vllm-base-url", default=DEFAULT_VLLM_BASE_URL, help="Base URL of the vLLM OpenAI server.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI served model name.")
    parser.add_argument("--upload-dir", default=str(DEFAULT_UPLOAD_DIR), help="Directory used to store uploaded images.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    return parser.parse_args()


def make_handler(vllm_base_url: str, model: str, upload_dir: Path, max_tokens: int, temperature: float):
    class ChatUIHandler(BaseHTTPRequestHandler):
        server_version = "MiniLLaVAChatUI/1.0"

        def do_GET(self):
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            self._send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/api/chat":
                self.send_error(404)
                return

            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                body = self.rfile.read(content_length)
                payload = json.loads(body.decode("utf-8"))
                messages = self._materialize_images(payload["messages"])
                upstream_payload = {
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "frequency_penalty": 0.2,
                    "stop": ["\nUSER:", "\nASSISTANT:", "USER:", "ASSISTANT:"],
                }
                upstream = self._post_to_vllm(upstream_payload)
                self._send_json(upstream)
            except KeyError as exc:
                self._send_json({"error": f"Missing field: {exc}"}, status=400)
            except json.JSONDecodeError:
                self._send_json({"error": "Invalid JSON request body."}, status=400)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                self._send_json({"error": detail or str(exc)}, status=exc.code)
            except urllib.error.URLError as exc:
                self._send_json({"error": f"Cannot connect to vLLM server: {exc.reason}"}, status=502)

        def _post_to_vllm(self, payload):
            url = f"{vllm_base_url.rstrip('/')}/v1/chat/completions"
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.loads(response.read().decode("utf-8"))

        def _materialize_images(self, messages):
            materialized = copy.deepcopy(messages)
            upload_dir.mkdir(parents=True, exist_ok=True)

            for message in materialized:
                content = message.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if block.get("type") != "image_url":
                        continue
                    image_url = block.get("image_url")
                    if not isinstance(image_url, dict):
                        continue
                    url = image_url.get("url", "")
                    if not url.startswith("data:"):
                        continue
                    image_url["url"], image_id = self._save_data_url(url)
                    block.setdefault("uuid", image_id)
            return materialized

        def _save_data_url(self, url):
            header, encoded = url.split(",", 1)
            media_type = header[5:].split(";", 1)[0] or "image/png"
            extension = mimetypes.guess_extension(media_type) or ".png"
            image_bytes = base64.b64decode(encoded)
            image_id = uuid.uuid4().hex
            image_path = upload_dir / f"{image_id}{extension}"
            image_path.write_bytes(image_bytes)
            return image_path.resolve().as_uri(), image_id

        def _send_json(self, payload, status=200):
            self._send_bytes(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status=status,
            )

        def _send_bytes(self, body, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt, *args):
            print(f"{self.address_string()} - {fmt % args}")

    return ChatUIHandler


def main():
    args = parse_args()
    upload_dir = Path(args.upload_dir).expanduser().resolve()
    handler = make_handler(args.vllm_base_url, args.model, upload_dir, args.max_tokens, args.temperature)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"MiniLLaVA chat UI: http://{args.host}:{args.port}")
    print(f"Proxying chat requests to: {args.vllm_base_url.rstrip('/')}/v1/chat/completions")
    print(f"Saving uploaded images to: {upload_dir}")
    server.serve_forever()


if __name__ == "__main__":
    main()
