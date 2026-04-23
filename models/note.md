## 模型下载
注册hf账号用于下载模型  
信息填写时国家写美国，尽量填写全  

**下载hf**
```shell
curl -LsSf https://hf.co/cli/install.sh | bash
```
**登录**
```shell
hf auth login
```
**下载模型**
```shell
hf download meta-llama/Llama-3.1-8B-Instruct --local-dir models/Llama-3.1-8B-Instruct
hf download Salesforce/blip-image-captioning-base --local-dir models/blip-image-captioning-base
hf download Salesforce/blip2-flan-t5-xl --local-dir models/blip2-flan-t5-xl
hf download openai/clip-vit-base-patch16 --local-dir models/clip-vit-base-patch16
hf download llava-hf/llava-1.5-7b-hf --local-dir models/llava-1.5-7b-hf
hf download Qwen/Qwen1.5-1.8B-Chat --local-dir models/Qwen1.5-1.8B-Chat
hf download Qwen/Qwen1.5-1.8B --local-dir models/Qwen1.5-1.8B
```