scheduler 的总步数应该按 accelerate 分片后的 dataloader 长度来算，否则多卡时学习率会按单卡步数走得偏慢。
把模型、优化器、dataloader、scheduler 交给 Accelerator.prepare()，反向传播、梯度裁剪、日志和保存都走 accelerate 的多进程安全接口。
每个进程拿自己的 dataloader shard，反向传播和梯度裁剪用 accelerate 接管，checkpoint 只由主进程写


用 accelerator.prepare() 包装 model / optimizer / dataloader / scheduler
用 accelerator.backward(loss) 替代 loss.backward()
用 accelerator.clip_grad_norm_() 做多卡兼容的梯度裁剪
多进程下只在主进程打印日志和保存 checkpoint
保存模型时用 accelerator.unwrap_model(model) 和 accelerator.save()
loss 日志用 accelerator.gather_for_metrics() 聚合各卡 loss

运行方式示例：
accelerate launch --num_processes 2 week08_minillava_training_v1/code/train.py
或者先配置：
accelerate config
accelerate launch week08_minillava_training_v1/code/train.py