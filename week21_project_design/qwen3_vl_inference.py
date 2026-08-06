import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import torch
from transformers import AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration as Qwen3VLModel
except ImportError:
    from transformers import AutoModelForImageTextToText as Qwen3VLModel


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = ROOT / "models" / "Qwen3-VL-8B-Instruct"
DEFAULT_INPUT_PATH = ROOT / "docs" / "notes" / "BLIP.pdf"
DEFAULT_MARKDOWN_PATH = Path(__file__).resolve().parent / "output" / "BLIP.md"
DEFAULT_QUESTION = "根据这篇文章，总结文章创新点"

MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(([^)\s]+)(?:\s+['\"][^'\"]*['\"])?\)")
HTML_IMAGE_RE = re.compile(r"<img[^>]+src=[\"']([^\"']+)[\"'][^>]*>", re.IGNORECASE)


@dataclass
class DocumentInput:
    text: str
    images: list[Path]


def resolve_image_path(raw_path: str, base_dir: Path) -> Path | None:
    raw_path = unquote(raw_path.strip().strip("<>"))
    if not raw_path or raw_path.startswith(("http://", "https://", "data:")):
        return None

    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    path = path.resolve()
    return path if path.exists() else None


def extract_markdown_images(text: str, base_dir: Path, max_images: int | None) -> list[Path]:
    image_paths = []
    seen = set()
    raw_paths = MARKDOWN_IMAGE_RE.findall(text) + HTML_IMAGE_RE.findall(text)

    for raw_path in raw_paths:
        image_path = resolve_image_path(raw_path, base_dir)
        if image_path is None or image_path in seen:
            continue
        seen.add(image_path)
        image_paths.append(image_path)
        if max_images is not None and len(image_paths) >= max_images:
            break

    return image_paths


def parse_markdown(
    path: Path,
    max_chars: int | None = None,
    max_images: int | None = None,
) -> DocumentInput:
    """Read markdown text and collect local images referenced by markdown/html tags."""
    text = path.read_text(encoding="utf-8")
    images = extract_markdown_images(text, path.parent, max_images)

    text = re.sub(r"<div[^>]*>.*?</div>", "\n", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "\n", text)
    text = re.sub(r"<img[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return DocumentInput(text=text, images=images)


def parse_pdf(path: Path, max_chars: int | None = None) -> DocumentInput:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "PDF input requires pypdfium2. Install it with: pip install pypdfium2"
        ) from exc

    pdf = pdfium.PdfDocument(path)
    pages = []
    for index in range(len(pdf)):
        page = pdf[index]
        textpage = page.get_textpage()
        pages.append(textpage.get_text_range())
        textpage.close()
        page.close()

    text = "\n\n".join(pages)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    return DocumentInput(text=text, images=[])


def parse_document(
    path: Path,
    max_chars: int | None = None,
    max_images: int | None = None,
) -> DocumentInput:
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt"}:
        return parse_markdown(path, max_chars, max_images)
    if suffix == ".pdf":
        return parse_pdf(path, max_chars)
    raise ValueError(f"Unsupported input file type: {suffix}. Use markdown, txt, or pdf.")


def build_messages(document: DocumentInput, question: str) -> list[dict]:
    image_note = ""
    if document.images:
        image_note = f"\n\n文档中同时附带了 {len(document.images)} 张按原文顺序抽取的图片，请结合这些图片理解图表、模型结构和实验结果。"

    content = [
        {
            "type": "text",
            "text": (
                "你是一名多模态论文阅读助手。请只根据给定文章内容作答，"
                "优先提炼方法层面的创新点，并用中文分点总结。"
                f"{image_note}\n\n文章文本：\n{document.text}"
            ),
        }
    ]
    for image_path in document.images:
        content.append({"type": "image", "image": str(image_path)})
    content.append({"type": "text", "text": f"问题：{question}"})
    return [{"role": "user", "content": content}]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use local Qwen3-VL-8B-Instruct to summarize innovations in a paper."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Input paper file. Supports markdown, txt, and pdf.",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Backward-compatible alias for --input.",
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=None,
        help="Optionally truncate the parsed document before inference.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=8,
        help="Maximum number of local images to include from markdown input. Use 0 to disable.",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help='Passed to from_pretrained; use "cpu" to force CPU inference.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.resolve()
    input_path = (args.markdown or args.input).resolve()
    max_images = None if args.max_images < 0 else args.max_images

    document = parse_document(input_path, args.max_input_chars, max_images)
    messages = build_messages(document, args.question)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLModel.from_pretrained(
        model_path,
        dtype="auto",
        device_map=args.device_map,
        trust_remote_code=True,
    )
    model.eval()

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )

    generated_ids_trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text[0].strip())


if __name__ == "__main__":
    main()
