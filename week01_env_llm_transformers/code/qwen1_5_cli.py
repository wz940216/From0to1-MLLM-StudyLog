import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "models/Qwen1.5-1.8B-Chat"

class QWEN1_5_Chatbot:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.messages = [
            {"role": "system", "content": "你是一个乐于助人的中文助手。"},
        ]

    def init_model(self):
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=True,        # 明确只用本地文件，不去 Hub 上找
            use_fast=True
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            local_files_only=True,        # 明确只用本地文件，不去 Hub 上找
            dtype=torch.bfloat16,
            device_map="auto"           # 自动用两张 3090
        )

    def generate_response(self, user_input):

        self.messages.append({"role": "user", "content": user_input})

        prompt = self.tokenizer.apply_chat_template(
            self.messages,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        self.messages.append({"role": "assistant", "content": response})

        return response, self.messages

    def reset(self):
        self.messages = [
            {"role": "system", "content": "你是一个乐于助人的中文助手。"},
        ]
        
    def free_model(self):
        del self.model
        del self.tokenizer
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # main()
    chatbot = QWEN1_5_Chatbot()
    chatbot.init_model()
    
    while True:
        text = input("请输入：")
        if text == "reset":
            chatbot.reset()
            print("已重置对话历史。")
            continue
        elif text == "exit":
            print("退出程序。")
            chatbot.free_model()
            break
        response, messages = chatbot.generate_response(text)
        print(response)
