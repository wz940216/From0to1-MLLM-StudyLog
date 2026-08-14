"""Prompt templates for common document QA scenarios."""

from __future__ import annotations


PROMPTS = {
    "general": """你是一名严谨的中文文档问答助手。请只根据给定 OCR 文本和页面图片回答。回答必须使用中文，并在关键结论后标注页码依据，例如：依据第 3 页。若文档没有明确依据，请说明未在文档中找到明确依据。""",
    "table": """你是一名擅长表格和数字核对的中文文档问答助手。请优先检查表格、数字、单位、列名和行名，回答必须使用中文，并标注页码依据，例如：依据第 3 页。涉及计算时写出简要计算过程。若依据不足，请说明未在文档中找到明确依据。""",
    "heading": """你是一名擅长标题、章节和目录定位的中文文档问答助手。请优先识别标题层级、章节编号、页眉页脚和目录信息，回答必须使用中文，并标注页码依据，例如：依据第 3 页。若无法定位，请说明未在文档中找到明确依据。""",
    "summary": """你是一名中文文档总结助手。请只根据给定内容总结，按要点输出，并尽量为每个核心要点标注页码依据，例如：依据第 3 页。不要引入文档外信息。""",
}


def infer_prompt_type(question: str) -> str:
    text = question.lower()
    table_keywords = {"表", "表格", "数字", "金额", "占比", "增长", "下降", "total", "table", "ratio"}
    heading_keywords = {"标题", "章节", "目录", "小节", "heading", "section", "title"}
    summary_keywords = {"总结", "概括", "归纳", "创新点", "摘要", "summary"}
    if any(keyword in text for keyword in table_keywords):
        return "table"
    if any(keyword in text for keyword in heading_keywords):
        return "heading"
    if any(keyword in text for keyword in summary_keywords):
        return "summary"
    return "general"


def resolve_prompt(prompt_type: str, question: str) -> tuple[str, str]:
    resolved = infer_prompt_type(question) if prompt_type == "auto" else prompt_type
    if resolved not in PROMPTS:
        resolved = "general"
    return resolved, PROMPTS[resolved]
