"""Qwen3-VL document question answering."""

from __future__ import annotations

import re
from pathlib import Path

from .prompt_templates import resolve_prompt
from .schemas import DocumentArtifacts, OCRPage


WORD_RE = re.compile(r"[\w\u4e00-\u9fff]+")


def _keywords(question: str) -> set[str]:
    return {token.lower() for token in WORD_RE.findall(question) if len(token.strip()) >= 2}


def select_relevant_pages(ocr_pages: list[OCRPage], question: str, max_images: int) -> list[OCRPage]:
    """Select pages by simple lexical overlap, falling back to the first pages."""
    if max_images <= 0:
        return []
    keywords = _keywords(question)
    scored: list[tuple[int, int, OCRPage]] = []
    for page in ocr_pages:
        text = page.text.lower()
        score = sum(1 for keyword in keywords if keyword in text)
        scored.append((score, -page.page_number, page))
    relevant = [item[2] for item in sorted(scored, reverse=True) if item[0] > 0]
    if len(relevant) < max_images:
        seen = {page.page_number for page in relevant}
        relevant.extend(page for page in ocr_pages if page.page_number not in seen)
    return sorted(relevant[:max_images], key=lambda page: page.page_number)


class QwenVLDocumentQA:
    def __init__(self, model_path: Path, device_map: str = "auto") -> None:
        self.model_path = model_path
        self.device_map = device_map
        self.processor = None
        self.model = None

    def load(self) -> None:
        if self.model is not None and self.processor is not None:
            return
        from transformers import AutoProcessor

        try:
            from transformers import Qwen3VLForConditionalGeneration as Qwen3VLModel
        except ImportError:
            from transformers import AutoModelForImageTextToText as Qwen3VLModel

        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = Qwen3VLModel.from_pretrained(
            self.model_path,
            dtype="auto",
            device_map=self.device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    def build_messages(
        self,
        artifacts: DocumentArtifacts,
        question: str,
        max_images: int,
        max_chars: int,
        prompt_type: str = "auto",
    ) -> tuple[list[dict], list[int], str]:
        selected_pages = select_relevant_pages(artifacts.ocr_pages, question, max_images)
        resolved_prompt_type, system_prompt = resolve_prompt(prompt_type, question)
        text = artifacts.full_text
        if max_chars > 0 and len(text) > max_chars:
            text = text[:max_chars] + "\n\n[文本已按 DOC_QA_MAX_INPUT_CHARS 截断]"

        selected_note = ", ".join(str(page.page_number) for page in selected_pages) or "未附带页面图片"
        content = [
            {
                "type": "text",
                "text": (
                    f"{system_prompt}\n\n"
                    f"已附带页面图片页码：{selected_note}。\n"
                    "OCR 文本按 [Page n] 标记页码，请回答时引用这些页码。\n\n"
                    f"OCR 文本：\n{text}"
                ),
            }
        ]
        for page in selected_pages:
            content.append({"type": "image", "image": str(page.image_path)})
        content.append({"type": "text", "text": f"问题：{question}"})
        return [{"role": "user", "content": content}], [page.page_number for page in selected_pages], resolved_prompt_type

    def answer(
        self,
        artifacts: DocumentArtifacts,
        question: str,
        max_images: int,
        max_input_chars: int,
        max_new_tokens: int,
        temperature: float = 0.0,
        prompt_type: str = "auto",
    ) -> dict:
        self.load()
        messages, selected_page_numbers, resolved_prompt_type = self.build_messages(
            artifacts, question, max_images, max_input_chars, prompt_type
        )
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            generation_kwargs["temperature"] = temperature

        import torch

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids_trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return {"answer": output_text, "selected_pages": selected_page_numbers, "prompt_type": resolved_prompt_type}
