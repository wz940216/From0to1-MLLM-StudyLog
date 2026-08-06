from pathlib import Path
import os

cache_home = os.environ.get("PADDLE_PDX_CACHE_HOME") or os.environ.get("PADDLEX_HOME") or str(Path("models/.paddlex"))
os.environ["PADDLE_PDX_CACHE_HOME"] = cache_home
os.environ["PADDLEX_HOME"] = cache_home
Path(cache_home, "official_models").mkdir(parents=True, exist_ok=True)
print(f"PADDLE_PDX_CACHE_HOME={cache_home}")

from paddleocr import PPStructureV3

input_file = "docs/notes/BLIP.pdf"
output_path = Path("week21_project_design/output")

pipeline = PPStructureV3(
    engine="transformers",
    lang="en",
    use_table_recognition=False,
    use_formula_recognition=False,
)
# pipeline = PPStructureV3(lang="en") # Set the lang parameter to use the English text recognition model. For other supported languages, see Section 5: Appendix. By default, both Chinese and English text recognition models are enabled.
# pipeline = PPStructureV3(use_doc_orientation_classify=True) # Use use_doc_orientation_classify to enable/disable document orientation classification model
# pipeline = PPStructureV3(use_doc_unwarping=True) # Use use_doc_unwarping to enable/disable document unwarping module
# pipeline = PPStructureV3(use_textline_orientation=True) # Use use_textline_orientation to enable/disable textline orientation classification model
# pipeline = PPStructureV3(device="gpu") # Use device to specify GPU for model inference

output = pipeline.predict(input=input_file)

markdown_list = []
markdown_images = []

for res in output:
    md_info = res.markdown
    markdown_list.append(md_info)
    markdown_images.append(md_info.get("markdown_images", {}))

markdown_texts = pipeline.concatenate_markdown_pages(markdown_list)
print(f"markdown result type={type(markdown_texts).__name__}")
if not isinstance(markdown_texts, str):
    markdown_texts = markdown_texts["markdown_texts"]
if not isinstance(markdown_texts, str):
    raise TypeError(f"Expected markdown text to be str, got {type(markdown_texts).__name__}")

mkd_file_path = output_path / f"{Path(input_file).stem}.md"
mkd_file_path.parent.mkdir(parents=True, exist_ok=True)

with open(mkd_file_path, "w", encoding="utf-8") as f:
    f.write(markdown_texts)

for item in markdown_images:
    if item:
        for path, image in item.items():
            file_path = output_path / path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(file_path)