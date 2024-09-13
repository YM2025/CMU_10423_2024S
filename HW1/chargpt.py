"""
训练一个字符级别的语言模型。
"""




import os
import sys
import json

# 追加新的路径到 sys.path
os.chdir('E:/project_2024/CMU_10423/CMU_10423_2024S/HW1/')
import wandb
from torchinfo import summary 


import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from mingpt.model import GPT
from mingpt.trainer import Trainer
from mingpt.utils import set_seed, setup_logging, CfgNode as CN
import pickle
# -----------------------------------------------------------------------------

def get_config():

    C = CN()

    # 系统相关配置
    C.system = CN()
    C.system.seed = 3407
    C.system.work_dir = './out/chargpt'

    # 数据相关配置
    C.data = CharDataset.get_default_config()

    # 模型相关配置
    C.model = GPT.get_default_config()
    # gpt-mini配置
    C.model.n_layer = 6
    C.model.n_query_head = 6
    C.model.n_kv_head = 2
    C.model.n_embd = 192
    C.model.rope = False  # 切换 True 或 False 来分别打开或关闭 RoPE
    

    # 训练器相关配置
    C.trainer = Trainer.get_default_config()
    C.trainer.learning_rate = 5e-4  # 我们使用的模型很小，可以使用较大的学习率
    C.trainer.max_iters = 600  # 总训练迭代次数

    return C

# -----------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    生成字符批次
    """

    @staticmethod
    def get_default_config():
        C = CN()
        C.block_size = 128
        C.tokenizer = "default"
        return C

    def __init__(self, config, data):
        self.config = config

        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('数据有 %d 个字符，%d 个唯一字符。' % (data_size, vocab_size))

        # "分词器" 只是将数据集中的字符映射为整数！
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = vocab_size
        self.data = [self.stoi[s] for s in data]  # 数据

    def get_vocab_size(self):
        return self.vocab_size

    def get_block_size(self):
        return self.config.block_size

    def __len__(self):
        return len(self.data) - self.config.block_size #减去是为了避免idx取到序列末尾时，越界self.data

    def __getitem__(self, idx):
        # 从数据中获取 (block_size + 1) 长度的字符块
        dix = self.data[idx:idx + self.config.block_size + 1]
        # 返回为张量
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

# -----------------------------------------------------------------------------

if __name__ == '__main__':

    USE_WANDB = True

    # 获取默认配置，如果有命令行参数则进行覆盖
    config = get_config()
    config.merge_from_args(sys.argv[1:])
    set_seed(config.system.seed)

    if USE_WANDB:
        # 从文件中读取API密钥并登录
        with open("C:/wandb_key.txt", "r") as key_file:
            api_key = key_file.read().strip()
        wandb.login(key=api_key)
        

        run = wandb.init(
            name    = "vanilla minGPT",     ### Wandb creates random run names if you skip this field, we recommend you give useful names
            reinit  = True,         # 允许你在同一个Python进程中多次初始化WandB运行，这在调试或需要多次启动运行的场景中非常有用。
            project = "10423-hw1",  
            config  =  config   
        )



    # 构建训练数据集
    text = open('input.txt', 'r').read()  # 不用担心文件句柄耗尽
    train_dataset = CharDataset(config.data, text)

    # 构建模型
    config.model.vocab_size = train_dataset.get_vocab_size()
    config.model.block_size = train_dataset.get_block_size()
    print(config)
    model = GPT(config.model)
    

    # 假设模型是 'model'
    # 假设输入数据形状为 (batch_size, sequence_length)，例如 (1, 128)
    # 使用 torch.randint 创建整数类型的输入张量，并将其转换为 LongTensor
    model.to('cuda')
    dummy_input = torch.randint(0, config.model.vocab_size, (64, 128)).long().to('cuda')  # 确保是 LongTensor 类型，并且传递到 CUDA 设备

    # 打印模型结构，输入形状为 (1, 128)
    summary(model, 
            input_data=dummy_input,
            col_names = ("input_size", "output_size", "num_params", "params_percent")
            )
    

    
    if config.model.pretrained_folder != None:
        assert os.path.normpath(os.path.abspath(config.model.pretrained_folder)) != os.path.normpath(os.path.abspath(config.system.work_dir)), "预训练模型文件夹不能与当前文件夹相同。通过标志更改预训练模型或当前目录的文件夹名称"
        model.load_pretrained(config.model.pretrained_folder)
    
    setup_logging(config)

    # 一些测试
    assert config.model.n_query_head % config.model.n_kv_head == 0, f"查询头 ({config.model.n_query_head}) 必须可以被键值头 ({config.model.n_kv_head}) 整除"
    assert ((config.model.n_embd % config.model.n_query_head == 0) and (config.model.n_embd % config.model.n_kv_head == 0)), f"嵌入维度 ({config.model.n_embd}) 必须可以被查询头 ({config.model.n_query_head}) 和键值头 ({config.model.n_kv_head}) 整除"

    # 构建训练器对象
    trainer = Trainer(config.trainer, model, train_dataset)

    train_losses = []
    attn_times = []
    attn_mem = []
    
    # 迭代回调（每次训练完一个batch， 就会调用一次这个函数）
    def batch_end_callback(trainer):
        
        if trainer.iter_num % 1 == 0:
            train_losses.append(trainer.loss.item())
            attn_times.append(trainer.attn_times * 1000)
            if trainer.device == "cuda":
                print(f"迭代耗时 {trainer.iter_dt:.2f}s; 迭代 {trainer.iter_num}: 训练损失 {trainer.loss.item():.5f}; 注意力计算耗时 {trainer.attn_times * 1000:.2f}ms; 内存消耗 {trainer.memory_consumed / (1024 * 1024):.2f}MB")
                attn_mem.append(trainer.memory_consumed / (1024 * 1024))
            else:
                print(f"迭代耗时 {trainer.iter_dt:.2f}s; 迭代 {trainer.iter_num}: 训练损失 {trainer.loss.item():.5f}; 注意力计算耗时 {trainer.attn_times * 1000:.2f}ms; 内存消耗 - CPU上不可用")

            if USE_WANDB:
                # 记录loss
                wandb.log({
                    "train_loss": trainer.loss.item()
                })
            

        if (trainer.iter_num + 1) % 200 == 0:
            # 评估训练和测试得分
            model.eval()
            with torch.no_grad():
                # # 从模型中采样...
                # context = "O God, O God!"
                context = "Before we proceed any further, hear me speak." # 训练集开头第一句话

                encoded_context = [train_dataset.stoi[s] for s in context]  # 使用分词器进行编码
                x = torch.tensor(encoded_context, dtype=torch.long)[None, ...].to(trainer.device)
                y, attn_time = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)
                y = y[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y])  # 使用分词器解码
                print(completion)
                print(f"注意力计算耗时 {attn_time * 1000:.2f}ms，序列长度为 {config.data.block_size}")



                context = "Gregory, o'my word, we'll not carry coals." # 《罗密欧与朱丽叶》第一句话
                encoded_context = [train_dataset.stoi[s] for s in context]  
                x = torch.tensor(encoded_context, dtype=torch.long)[None, ...].to(trainer.device)
                y, attn_time = model.generate(x, 500, temperature=1.0, do_sample=True, top_k=10)
                y = y[0]
                completion = ''.join([train_dataset.itos[int(i)] for i in y]) 
                print(completion)
                print(f"注意力计算耗时 {attn_time * 1000:.2f}ms，序列长度为 {config.data.block_size}")


            # 保存最新的模型
            print("保存模型")
            ckpt_path = os.path.join(config.system.work_dir, "model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print("保存损失和注意力日志")
            data = {
                "train_losses": train_losses,
                "attention_computation_time": attn_times,
                "attention_computation_memory": attn_mem
            }
            with open(os.path.join(config.system.work_dir, 'train_logs.json'), 'w') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            # 将模型恢复到训练模式
            model.train()

    trainer.set_callback('on_batch_end', batch_end_callback)

    # 运行优化
    trainer.run()
    
    if USE_WANDB:
        # 结束 wandb 运行
        wandb.finish()