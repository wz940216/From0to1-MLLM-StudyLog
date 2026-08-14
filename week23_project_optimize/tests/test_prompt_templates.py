from week23_project_optimize.core.prompt_templates import infer_prompt_type, resolve_prompt


def test_infer_prompt_type_table():
    assert infer_prompt_type("表格里的增长是多少") == "table"


def test_resolve_prompt_unknown_fallback():
    prompt_type, prompt = resolve_prompt("unknown", "请总结")
    assert prompt_type == "general"
    assert "中文" in prompt
