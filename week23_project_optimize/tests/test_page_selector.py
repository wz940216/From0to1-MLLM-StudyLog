from pathlib import Path

from week23_project_optimize.core.schemas import OCRPage
from week23_project_optimize.core.vlm_engine import select_relevant_pages


def page(num, text):
    return OCRPage(page_number=num, image_path=Path(f"p{num}.png"), width=10, height=10, blocks=[]).__class__(
        page_number=num,
        image_path=Path(f"p{num}.png"),
        width=10,
        height=10,
        blocks=[],
    )


def test_select_relevant_pages_falls_back_to_first_pages():
    pages = [page(1, ""), page(2, "")]
    selected = select_relevant_pages(pages, "不存在的关键词", 1)
    assert [p.page_number for p in selected] == [1]
