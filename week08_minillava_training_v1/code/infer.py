import argparse

from PIL import Image
import torch

from dataset import build_prompt
from mini_llava import MiniLlavaModel


def main():
    parser = argparse.ArgumentParser(description="MiniLLaVA推理脚本")
    parser.add_argument("--config", default="week08_minillava_training_v1/code/config.yaml")
    parser.add_argument("--checkpoint", default="week08_minillava_training_v1/outputs/checkpoints/step_3000.pt", help="训练得到的 .pt 检查点路径")
    parser.add_argument("--image", default="dataset/coco128/images/train2017/000000000025.jpg", help="输入图片路径")
    parser.add_argument("--question", default="What is in the picture?", help="关于图片的问题")
    args = parser.parse_args()

    model = MiniLlavaModel(args.config)
    if args.checkpoint is not None:
        # 检查点只保存可训练参数和 LoRA adapter；需用同样 LoRA 配置初始化模型后按名称加载。
        state = torch.load(args.checkpoint, map_location=model.device)
        model.load_state_dict(state["model"], strict=False)

    image = Image.open(args.image).convert("RGB")
    prompt = build_prompt(args.question)
    gen_config = model.config["INFERENCE"]["GENERATION"]
    outputs = model.generate(
        images=[image],
        prompts=[prompt],
        max_new_tokens=int(gen_config["MAX_NEW_TOKENS"]),
        temperature=float(gen_config["TEMPERATURE"]),
        do_sample=bool(gen_config["DO_SAMPLE"]),
        top_p=float(gen_config["TOP_P"]),
        top_k=int(gen_config["TOP_K"]),
        repetition_penalty=float(gen_config["REPETITION_PENALTY"])
    )
    print(outputs[0])


if __name__ == "__main__":
    main()
