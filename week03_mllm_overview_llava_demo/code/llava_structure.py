from accelerate import init_empty_weights
from transformers import LlavaConfig, LlavaForConditionalGeneration

hf_model_path = "models/llava-1.5-7b-hf"

def test_llava_structure(hf_model_path):
    
    # 直接加载模型参数
    # model = LlavaForConditionalGeneration.from_pretrained(
    #     hf_model_path,
    #     torch_dtype="auto",
    #     device_map="auto",
    # )
    # print(model)

    
    # 使用 init_empty_weights()，只构建网络结构，不真正分配参数内存
    with init_empty_weights():
        config = LlavaConfig.from_pretrained(hf_model_path)
        model = LlavaForConditionalGeneration(config)

        # 打印模型结构
        print(model)

        # 打印参数总量（虽然是“空权重”，但 numel 信息是有的）
        total_params = sum(p.numel() for p in model.parameters())
        print(f"模型总参数量: {total_params / 1e6:.2f} M")

        # 打印部分参数名称和形状
        for name, param in list(model.named_parameters())[:50]:  # 想看更多改这个数字
            print(name, param.shape)

if __name__ == "__main__":
    test_llava_structure(hf_model_path)
