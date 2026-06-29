IMAGE_TOKEN = "<image>"


JSON_FEW_SHOT_PROMPT = """你需要按固定业务 JSON 格式回答。
要求：
- 只输出一个合法 JSON 对象。
- 不要输出 markdown 代码块。
- 不要输出额外解释。
- 字段固定为 answer、confidence、evidence。

示例 1：
用户：图中主要物体是什么？
助手：{"answer": "一只狗", "confidence": "medium", "evidence": "画面中能看到狗的身体和草地"}

示例 2：
用户：图片里的天气如何？
助手：{"answer": "晴朗", "confidence": "low", "evidence": "画面光线较亮，但不能直接确认天气"}"""


def clean_user_text(text):
    return str(text).replace(IMAGE_TOKEN, "").strip()


def build_json_few_shot_question(question):
    question = clean_user_text(question)
    if not question:
        return JSON_FEW_SHOT_PROMPT
    return f"{JSON_FEW_SHOT_PROMPT}\n\n当前用户问题：{question}"
