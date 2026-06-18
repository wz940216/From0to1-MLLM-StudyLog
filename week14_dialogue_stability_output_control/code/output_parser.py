import json
import re
from dataclasses import dataclass


@dataclass
class ParseResult:
    ok: bool
    data: object = None
    text: str = ""
    error: str = ""


JSON_REPAIR_PROMPT = """上一轮回答不是合法 JSON。请只根据原始问题重新输出一个合法 JSON 对象。
不要输出 markdown 代码块，不要输出解释文字。
固定格式：
{"answer": "这里填写对用户问题的回答"}"""


def strip_code_fence(text):
    text = str(text).strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_json_candidate(text):
    """从模型输出中提取最可能的 JSON 对象片段。"""
    text = strip_code_fence(text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return text.strip()
    return text[start:end + 1].strip()


def parse_json_output(text, required_keys=None):
    """校验 JSON 输出，并可选检查必需字段。"""
    candidate = extract_json_candidate(text)
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return ParseResult(False, text=candidate, error=str(exc))

    if not isinstance(data, dict):
        return ParseResult(False, data=data, text=candidate, error="JSON 顶层必须是对象。")

    missing = [key for key in (required_keys or []) if key not in data]
    if missing:
        return ParseResult(False, data=data, text=candidate, error=f"JSON 缺少字段: {', '.join(missing)}")

    return ParseResult(True, data=data, text=json.dumps(data, ensure_ascii=False))


def normalize_model_text(text, prompt=""):
    """尽量只保留当前轮 assistant 的回答。"""
    text = str(text).strip()
    if prompt and text.startswith(prompt):
        text = text[len(prompt):].strip()

    if "ASSISTANT:" in text:
        text = text.split("ASSISTANT:")[-1].strip()
    if "USER:" in text:
        text = text.split("USER:", 1)[0].strip()

    return strip_code_fence(text)


def looks_like_json_request(text):
    return re.search(r"\bjson\b|结构化|字段|格式", str(text), flags=re.IGNORECASE) is not None
