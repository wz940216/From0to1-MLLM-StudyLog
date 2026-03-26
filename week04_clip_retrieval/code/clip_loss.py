import torch
import torch.nn.functional as F


class CLIPLoss(torch.nn.Module):
    def __init__(self, temperature=0.07):
        super(CLIPLoss, self).__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, image_features, text_features):
        """
        计算 CLIP 的对比损失

        参数：
        - image_features: 形状为 (N, D) 的图像特征
        - text_features: 形状为 (N, D) 的文本特征

        返回：
        - CLIP 损失
        """

        if image_features.dim() != 2 or text_features.dim() != 2:
            raise ValueError("image_features 和 text_features 必须是二维张量 (N, D)")
        if image_features.shape != text_features.shape:
            raise ValueError("image_features 和 text_features 必须具有相同的形状")

        # 归一化特征（L2 归一化到单位球面）
        image_features = F.normalize(image_features, p=2, dim=-1)  # (N, D)
        text_features = F.normalize(text_features, p=2, dim=-1)  # (N, D)

        # 计算相似度矩阵（余弦相似度）
        logits = (image_features @ text_features.T) / self.temperature  # (N, N)

        # 目标标签：对角线上的才是匹配的
        labels = torch.arange(logits.shape[0], device=logits.device)

        # 计算交叉熵损失（分别计算 I->T 和 T->I）
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)
        
        # 这里 logits.shape = [16, 16] labels.shape = [16]
        # 可以这样理解：
        # logits[i]：第i个样本，对16个类别的“打分”
        # labels[i]：第i个样本的正确类别（0~15）
        # 也就是说：这是一个16分类问题，每个batch有16个样本
        # pytorch中的F.cross_entropy会先对logits[i]中的16个值做softmax，得到每个类别的概率分布
        # 再根据labels[i]指定的正确类别，计算交叉熵损失
        # 其实就是计算-log(p(correct_class_probability))，也就是正确类别的概率越大，损失越小

        return (loss_i2t + loss_t2i) / 2


if __name__ == '__main__':
    # 调试代码：使用随机图像/文本向量验证 CLIP 损失计算流程
    torch.manual_seed(123)

    batch_size = 16
    dim = 512
    image_features = torch.randn(batch_size, dim)
    text_features = torch.randn(batch_size, dim)

    clip_loss = CLIPLoss(temperature=0.07)
    loss_value = clip_loss(image_features, text_features)

    print(f"CLIP loss: {loss_value.item():.6f}")
 