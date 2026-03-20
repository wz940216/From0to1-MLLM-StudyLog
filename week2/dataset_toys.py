from datasets import load_dataset
from datasets import load_from_disk


# 加载原始alpaca数据集
data_files = {
    "train": "dataset/alpaca-chinese-dataset/data_v3/alpaca_chinese_part_0.json",
    "validation": "dataset/alpaca-chinese-dataset/data_v3/alpaca_chinese_part_1.json"
}

dataset = load_dataset("json", data_files=data_files)

print(dataset)
print(dataset["train"][0])

train_dataset = dataset["train"]
val_dataset = dataset["validation"]

# 转换成中文数据集
def convert_to_zh_format(example):
    return {
        "instruction": example["zh_instruction"],
        "input": example["zh_input"],
        "output": example["zh_output"]
    }

zh_train_dataset = train_dataset.map(convert_to_zh_format, remove_columns=train_dataset.column_names)
zh_val_dataset = val_dataset.map(convert_to_zh_format, remove_columns=val_dataset.column_names)

print(zh_train_dataset[0])
print(zh_val_dataset[0])

# 保存在本地
zh_train_dataset.save_to_disk("week2/my_alpaca_train_dataset")
zh_val_dataset.save_to_disk("week2/my_alpaca_val_dataset")

# 加载本地hf格式数据集
train_dataset_loaded = load_from_disk("week2/my_alpaca_train_dataset")
val_dataset_loaded = load_from_disk("week2/my_alpaca_val_dataset")

print(train_dataset_loaded[0])
print(val_dataset_loaded[0])











