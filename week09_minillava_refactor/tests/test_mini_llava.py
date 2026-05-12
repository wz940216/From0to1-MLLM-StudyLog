import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import yaml


CODE_DIR = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE_DIR))

import mini_llava  # noqa: E402


class DummyVisionEncoder(torch.nn.Module):
    def __init__(self, model_path, freeze=True, device="cpu"):
        super().__init__()
        self.device = torch.device(device)

    def forward(self, images):
        batch_size = len(images)
        return torch.arange(
            batch_size * 2 * 3,
            dtype=torch.float32,
            device=self.device,
        ).reshape(batch_size, 2, 3)


class DummyLLMDecoder(torch.nn.Module):
    def __init__(
        self,
        model_path,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        freeze=False,
        device="cpu",
    ):
        super().__init__()
        self.device = torch.device(device)
        self.embedding = torch.nn.Embedding(32, 5)
        self.model = SimpleNamespace(config=SimpleNamespace(hidden_size=5))
        self.last_call = None

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, labels=None, use_cache=False):
        self.last_call = {
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "labels": labels,
            "use_cache": use_cache,
        }
        return SimpleNamespace(
            loss=inputs_embeds.sum() * 0,
            logits=torch.zeros(inputs_embeds.size(0), inputs_embeds.size(1), 32),
        )


def write_config(tmp_path):
    config = {
        "DEVICE": "cpu",
        "MINILLAVA": {
            "VISION_ENCODER": {
                "MODEL_PATH": "dummy-clip",
                "FREEZE": True,
            },
            "LLM_DECODER": {
                "MODEL_PATH": "dummy-llm",
                "FREEZE": False,
                "LORA_R": 8,
                "LORA_ALPHA": 32,
                "LORA_DROPOUT": 0.1,
            },
            "PROJECTOR": {
                "INPUT_DIM": 3,
                "HIDDEN_DIM": 4,
            },
        },
        "DATA": {
            "PREPROCESS": {
                "MAX_TEXT_LENGTH": 16,
            },
        },
    }
    config_path = Path(tmp_path) / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


class MiniLlavaModelTest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def build_test_model(self):
        config_path = write_config(Path(self.tmpdir.name))
        with mock.patch.object(mini_llava, "VisionEncoder", DummyVisionEncoder), mock.patch.object(
            mini_llava,
            "LLMDecoder",
            DummyLLMDecoder,
        ):
            return mini_llava.MiniLlavaModel(str(config_path))

    def test_build_multimodal_inputs_concatenates_image_and_text(self):
        model = self.build_test_model()
        images = [object(), object()]
        input_ids = torch.tensor([[1, 2, 0], [3, 4, 5]])
        attention_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        inputs_embeds, combined_attention_mask, image_token_num = model._build_multimodal_inputs(
            images=images,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        self.assertEqual(image_token_num, 2)
        self.assertEqual(inputs_embeds.shape, (2, 5, 5))
        self.assertEqual(combined_attention_mask.shape, (2, 5))
        self.assertTrue(torch.equal(combined_attention_mask[:, :2], torch.ones(2, 2, dtype=torch.long)))
        self.assertTrue(torch.equal(combined_attention_mask[:, 2:], attention_mask))

    def test_forward_prepends_ignore_labels_for_image_tokens(self):
        model = self.build_test_model()
        labels = torch.tensor([[10, 11, -100]])

        outputs = model(
            images=[object()],
            input_ids=torch.tensor([[1, 2, 3]]),
            attention_mask=torch.tensor([[1, 1, 1]]),
            labels=labels,
        )

        call = model.language_decoder.last_call
        self.assertEqual(outputs.logits.shape, (1, 5, 32))
        self.assertFalse(call["use_cache"])
        self.assertEqual(call["labels"].shape, (1, 5))
        self.assertTrue(torch.equal(call["labels"][:, :2], torch.full((1, 2), -100)))
        self.assertTrue(torch.equal(call["labels"][:, 2:], labels))


if __name__ == "__main__":
    unittest.main()
