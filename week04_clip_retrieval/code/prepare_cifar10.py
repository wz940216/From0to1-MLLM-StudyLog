import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

labels_map = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_batch(file_path):
    with open(file_path, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict

def show_image(data, index):
    img = data[b'data'][index]  # (3072,)
    
    # reshape 成 (3, 32, 32)
    img = img.reshape(3, 32, 32)
    
    # 转成 (32, 32, 3)
    img = img.transpose(1, 2, 0)
    
    plt.imshow(img)
    plt.title(f"Label: {data[b'labels'][index]}")
    plt.axis('off')
    plt.show()
    
def save_image(data, index, path):
    img = data[b'data'][index]
    label = data[b'labels'][index]
    img = img.reshape(3, 32, 32).transpose(1, 2, 0)
    
    image = Image.fromarray(img)
    image.save(path)
    print(f"Image saved to {path} with label: {labels_map[label]}")
    
# 使用示例
batch = load_batch('dataset/cifar-10-batches-py/data_batch_1')
# show_image(batch, 0)
save_image(batch, 0, 'example.png')