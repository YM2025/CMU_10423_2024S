"""
简单的训练循环；这个代码是针对任意神经网络的通用模板，
所以这里的内容并没有专门针对 GPT。
"""

import time
from collections import defaultdict

import torch
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import CfgNode as CN



# 定义一个训练器类
class Trainer:

    # 获取默认配置的静态方法
    @staticmethod
    def get_default_config():
        C = CN()
        # 训练设备（默认设置为自动选择）
        C.device = 'auto'
        # 数据加载器的参数
        C.num_workers = 4
        # 优化器参数
        C.max_iters = 800  # 最大训练迭代次数
        C.batch_size = 64   # 每次训练批量大小
        C.learning_rate = 3e-4  # 学习率
        C.betas = (0.9, 0.95)   # Adam 优化器的 beta 参数
        C.weight_decay = 0.1    # 权重衰减，通常只应用于矩阵乘法的权重
        C.grad_norm_clip = 1.0  # 梯度剪裁，防止梯度爆炸
        return C

    # 初始化函数
    def __init__(self, config, model, train_dataset):
        self.config = config  # 传入的配置
        self.model = model    # 传入的模型
        self.optimizer = None  # 优化器变量，稍后会设置
        self.train_dataset = train_dataset  # 训练数据集
        self.callbacks = defaultdict(list)  # 事件回调，用于保存不同的事件回调函数

        # 确定使用的设备（CPU, CUDA 或 MPS）
        if config.device == 'auto':
            # 如果选择 auto，则自动选择可用的设备
            self.device = 'cuda' if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)  # 将模型转移到选择的设备
        print("running on device", self.device)

        # 一些在训练过程中需要的变量初始化
        self.iter_num = 0    # 迭代次数
        self.iter_time = 0.0  # 记录迭代开始的时间
        self.iter_dt = 0.0    # 记录迭代所花费的时间

    # 添加回调函数
    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    # 设置某一事件的回调函数
    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    # 触发回调函数
    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    # 训练流程
    def run(self):
        model, config = self.model, self.config

        # 设置优化器，使用模型中的方法配置优化器
        self.optimizer = model.configure_optimizers(config)

        # 设置数据加载器，使用随机抽样的策略进行数据采样
        train_loader = DataLoader(
            self.train_dataset,
            sampler=torch.utils.data.RandomSampler(self.train_dataset, replacement=True, num_samples=int(1e10)),
            shuffle=False,  # 因为已经使用了随机抽样，所以不需要再 shuffle
            pin_memory=True,  # 提高数据加载效率
            batch_size=config.batch_size,
            num_workers=config.num_workers,  # 使用多个线程加载数据
        )

        model.train()  # 将模型设置为训练模式
        self.iter_num = 0  # 迭代次数清零
        self.iter_time = time.time()  # 记录训练开始时间
        data_iter = iter(train_loader)  # 数据迭代器

        # 开始训练循环
        while True:
                
            # 获取下一个批次数据（x, y），如果数据迭代器结束了，则重新初始化
            try:
                batch = next(data_iter)
            except StopIteration:  # 数据迭代完后需要重新初始化
                data_iter = iter(train_loader)
                batch = next(data_iter)
                
            batch = [t.to(self.device) for t in batch]  # 将批次数据转移到设备
            x, y = batch
            
            if self.iter_num % 100 == 0:
                print(x.shape)

            # 前向传播
            logits, self.loss, self.attn_times, self.memory_consumed = model(x, y)

            # 反向传播并更新模型参数
            model.zero_grad(set_to_none=True)  # 梯度清零
            self.loss.backward()  # 反向传播计算梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)  # 梯度剪裁，防止爆炸
            self.optimizer.step()  # 更新模型参数

            # 触发 batch 结束的回调函数
            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1  # 更新迭代次数
            tnow = time.time()  # 当前时间
            self.iter_dt = tnow - self.iter_time  # 计算迭代时间
            # self.iter_time = tnow  # 更新时间

            # 如果达到最大迭代次数，则退出训练
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break
