"""
GPT语言模型的完整定义，所有内容都在此单一文件中。

参考文献：
1) OpenAI发布的GPT-2 TensorFlow官方实现：
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers的PyTorch实现：
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from mingpt.utils import CfgNode as CN

import time
import os

from einops import einsum, rearrange
import json

# -----------------------------------------------------------------------------

class NewGELU(nn.Module):
    """
    GELU激活函数的实现，当前在Google BERT仓库中使用（与OpenAI GPT相同）。
    参考：高斯误差线性单元（GELU）论文：https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

################################################     TODO     ################################################

class RotaryPositionalEmbeddings(nn.Module):
    """ 
    TODO: 实现RoPE（旋转位置嵌入），参考论文《RoFormer: Enhanced Transformer with Rotary Position Embedding》。
    参考文献：https://arxiv.org/abs/2104.09864
    你将实现论文中的方程34（旋转矩阵的高效计算形式，详见课堂讨论）。
    如果需要帮助将公式转化为PyTorch代码，请参考讨论中的“Example: Converting Math to PyTorch Code”幻灯片。
    """


    ## CMU某学生版（速度比下面GPT4优化的慢15%）
    # def __init__(self, d: int, base: int = 10_000):
    #     super().__init__()
    #     self.d = d
    #     self.base = base
    #     self.cosine_mat = None
    #     self.sine_mat = None
    #     # raise NotImplementedError("Initialization is not implemented.")

    # def _build_cache(self, x: torch.Tensor):
    #     """
    #     Compute the fixed variables that do not change during training (see recitation for more details).
    #     """
    #     device = x.device
    #     self.N = x.shape[-2]
    #     # self.d = x.shape[-1]

    #     positions = torch.arange(0, self.d/2, device = device)
    #     theta = torch.pow(self.base, -2**(positions)/self.d)
    #     c_matrix = torch.zeros((self.N,self.d), device = device)
    #     c_matrix[:,:self.d//2] = theta
    #     c_matrix[:,self.d//2:] = theta
    #     c_matrix = c_matrix*torch.arange(1,self.N+1, device = device).unsqueeze_(1)
    #     self.cosine_mat = torch.cos(c_matrix)
    #     self.cosine_mat = self.cosine_mat[None, None, :, :]
    #     self.sine_mat = torch.sin(c_matrix)
    #     self.sine_mat = self.sine_mat[None, None, :, :]
    #     # raise NotImplementedError("Rotary embeddings cache not implemented.")

    # def forward(self, x: torch.Tensor):
    #     """
    #     Perform the forward pass with the input x, following equation 34 in the paper.
    #     """
    #     # print('The shape of x is: ', x.shape)
    #     # print('The shape of the cosine matrix is', self.cosine_mat.shape)
    #     if (self.cosine_mat is None
    #          or x.shape[-2] != self.cosine_mat.shape[-2]  
    #            or x.shape[-1] != self.cosine_mat.shape[-1]):
    #         self._build_cache(x)
    #     return (x*self.cosine_mat) + (torch.cat((-x[:, :, :, self.d//2:], x[:, :, :, :self.d//2]), dim = 3)*self.sine_mat)


    # GPT o1-mini版
    # def __init__(self, dim: int, base: int = 10000):
    #     """
    #     初始化RoPE模块。

    #     参数：
    #         dim (int): 嵌入维度。
    #         base (int): 频率计算的基数，默认为10000。
    #     """
    #     super().__init__()
    #     self.dim = dim
    #     # 计算逆频率
    #     inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    #     self.register_buffer("inv_freq", inv_freq)

    # def _build_cache(self, seq_len: int, device: torch.device):
    #     """
    #     预计算并缓存正弦和余弦位置嵌入。

    #     参数：
    #         seq_len (int): 序列的最大长度。
    #         device (torch.device): 张量所在的设备。

    #     返回：
    #         sin (torch.Tensor): 正弦嵌入，形状为 (1, 1, seq_len, dim)。
    #         cos (torch.Tensor): 余弦嵌入，形状为 (1, 1, seq_len, dim)。
    #     """
    #     position = torch.arange(seq_len, device=device).type_as(self.inv_freq)
    #     freqs = torch.outer(position, self.inv_freq)  # 形状: (seq_len, dim/2)
    #     emb = torch.cat((freqs, freqs), dim=-1)  # 形状: (seq_len, dim)
    #     sin, cos = emb.sin(), emb.cos()
    #     return sin[None, None, :, :], cos[None, None, :, :]  # 形状: (1, 1, seq_len, dim)

    # def forward(self, x: torch.Tensor):
    #     """
    #     对输入张量x应用RoPE。

    #     参数：
    #         x (torch.Tensor): 输入张量，形状为 (batch, heads, seq_len, dim)。

    #     返回：
    #         torch.Tensor: 旋转后的张量，形状与输入相同。
    #     """
    #     seq_len = x.size(-2)
    #     device = x.device
    #     # 检查缓存是否已存在，否则构建缓存
    #     if not hasattr(self, 'sin') or self.sin.size(-2) < seq_len:
    #         self.sin, self.cos = self._build_cache(seq_len, device)
    #     # 裁剪缓存以适应当前序列长度
    #     sin = self.sin[:, :, :seq_len, :].to(x.dtype)
    #     cos = self.cos[:, :, :seq_len, :].to(x.dtype)
    #     return (x * cos) + (self.rotate_half(x) * sin)

    # @staticmethod
    # def rotate_half(x: torch.Tensor) -> torch.Tensor:
    #     """
    #     对最后一个维度进行90度旋转。

    #     参数：
    #         x (torch.Tensor): 输入张量。

    #     返回：
    #         torch.Tensor: 旋转后的张量。
    #     """
    #     x1 = x[..., ::2]
    #     x2 = x[..., 1::2]
    #     return torch.cat([-x2, x1], dim=-1)


    def __init__(self, d: int, base: int = 10_000): # 10_000类似于10,000这种英文写法
        """
        初始化RoPE模块。
        
        参数：
        d (int): 输入的嵌入维度。
        base (int): 用于生成旋转角度的基数，默认为10000。
        """
        super().__init__()
        self.d = d  # 嵌入维度（例如，d_model 或 d_query）
        self.base = base  # 基数，控制旋转角度的幅度
        self.cosine_mat = None  # 用于缓存 cos 矩阵
        self.sine_mat = None    # 用于缓存 sin 矩阵

    def _build_cache(self, x: torch.Tensor):
        """
        计算和缓存 cos 和 sin 矩阵，这些矩阵用于对输入进行旋转嵌入。
        此步骤根据输入的维度（序列长度和嵌入维度）动态生成。

        参数：
        x (torch.Tensor): 输入张量，包含批次大小、头数、序列长度和嵌入维度。
        """
        device = x.device  # 获取输入的设备（例如 CPU 或 GPU）
        self.N = x.shape[-2]  # 获取序列长度 N

        # 生成 theta 作为旋转角度，它基于嵌入维度的前一半
        # positions 是嵌入维度的一半（即 d // 2）个位置，从0到 (d//2 - 1)
        positions = torch.arange(0, self.d // 2, device=device)
        # theta 是旋转频率的参数，随着位置增大而衰减
        theta = torch.pow(self.base, -2 ** (positions) / self.d)

        # 生成矩阵 c_matrix，其尺寸为 (N, d//2)，用于储存每个序列位置的旋转频率
        # arange_N 是一个从 1 到 N 的向量，它表示每个位置（从1开始）
        arange_N = torch.arange(1, self.N + 1, device=device).unsqueeze(1)
        # c_matrix 是每个位置乘以 theta，用于生成 cos 和 sin 值
        c_matrix = arange_N * theta.unsqueeze(0)

        # 将 c_matrix 复制两份，形成 (N, d) 维度的矩阵，其中前半部分和后半部分相同
        # 生成对应的 cos 和 sin 矩阵，并扩展维度为 (1, 1, N, d)，以便广播操作
        self.cosine_mat = torch.cos(torch.cat([c_matrix, c_matrix], dim=-1)).unsqueeze(0).unsqueeze(0)
        self.sine_mat = torch.sin(torch.cat([c_matrix, c_matrix], dim=-1)).unsqueeze(0).unsqueeze(0)

    def forward(self, x: torch.Tensor):
        """
        前向传播：对输入张量 x 进行旋转位置嵌入。

        参数：
        x (torch.Tensor): 输入张量，形状为 (batch_size, n_heads, seq_len, d)。
        
        返回：
        torch.Tensor: 应用旋转位置嵌入后的张量。
        """
        # 检查是否需要重新生成 cos 和 sin 矩阵（基于输入的序列长度和嵌入维度）
        if (self.cosine_mat is None or x.shape[-2] != self.cosine_mat.shape[-2] or x.shape[-1] != self.cosine_mat.shape[-1]):
            self._build_cache(x)  # 如果缓存的矩阵尺寸不匹配，则重新构建

        # 获取嵌入维度的一半，便于将张量分成两部分
        half_d = self.d // 2
        
        # 计算 x 的 cos 部分：每个位置的值乘以相应的 cos 值
        x_cos = x * self.cosine_mat
        
        # 计算 x 的 sin 部分：
        # 将 x 的后半部分移到前面，前半部分移到后面，并乘以相应的 sin 值
        x_sin = torch.cat([-x[:, :, :, half_d:], x[:, :, :, :half_d]], dim=3) * self.sine_mat
        
        # 返回两个部分的和，作为旋转嵌入后的输出
        return x_cos + x_sin

#############################################################################################################

class CausalSelfAttention(nn.Module):
    """
    简单的多头自注意力机制。查询头=键头=值头
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_query_head == 0
        # 所有头的键、查询、值投影，但在一个批量内完成
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # 输出投影
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # 正则化
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # 因果掩码，确保注意力只应用于输入序列中的左边
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_query_head
        self.n_embd = config.n_embd
        
        ################################################     TODO     ################################################
        self.rope = config.rope
        if self.rope:
            # raise NotImplementedError("使用RoPE初始化注意力尚未实现。")
            self.custom_rope = RotaryPositionalEmbeddings(self.n_embd // self.n_head)
        #############################################################################################################

    def forward(self, x):
        B, T, C = x.size() # 批量大小，序列长度，嵌入维度（n_embd）

        # 为批量中所有头计算查询、键、值，并将头移到批量维度
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, d)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, d)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, d)

        ################################################     TODO     ################################################
        if self.rope:
            # raise NotImplementedError("使用RoPE的前向传播尚未实现。")
            #q, k = self.custom_rope(q, k)
            q = self.custom_rope(q)
            k = self.custom_rope(k)
        #############################################################################################################

        # 因果自注意力；自注意力：（B, nh, T, hs） x （B, nh, hs, T） -> （B, nh, T, T）
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, d) -> (B, nh, T, d)
        end_memory = torch.cuda.memory_allocated()
        y = y.transpose(1, 2).contiguous().view(B, T, C) # 将所有头输出重新排列到一起

        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        return y, end_memory-start_memory

################################################     TODO     ################################################
class GroupedQueryAttention(nn.Module):
    """
    实现分组查询注意力机制。
    """

    def __init__(self, config):
        super().__init__()

        # 确保嵌入维度能够整除查询头和键/值头的数量
        assert config.n_embd % config.n_query_head == 0
        assert config.n_query_head % config.n_kv_head == 0

        # 分组数量g（每个键/值头对应多少个查询头）
        self.g = config.n_query_head // config.n_kv_head

        # 键和值的线性投影层，生成键和值的向量，大小为 2 * n_embd
        self.vk_attn = nn.Linear(config.n_embd, 2 * config.n_embd)

        # 查询的线性投影层，生成查询的向量，维度为 g * n_embd
        self.q_attn = nn.Linear(config.n_embd, self.g * config.n_embd)

        # 输出的线性投影层，用于将注意力输出映射回嵌入维度
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)

        # 正则化，防止过拟合
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # 因果遮罩，确保注意力仅关注到当前位置及之前的序列
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

        # 存储键/值头数量和查询头数量
        self.n_head = config.n_kv_head
        self.n_query = config.n_query_head
        self.n_embd = config.n_embd
        
        # 是否使用RoPE（旋转位置嵌入）
        self.rope = config.rope
        if self.rope:
            # 如果使用RoPE，则初始化RoPE模块，维度为每个头的嵌入维度
            self.custom_rope = RotaryPositionalEmbeddings(self.n_embd // self.n_head)

    def forward(self, x):
        """
        前向传播，计算分组查询注意力输出。

        参数：
            x (torch.Tensor): 输入张量，形状为 (batch, seq_len, n_embd)。

        返回：
            torch.Tensor: 注意力输出，形状为 (batch, seq_len, n_embd)。
            int: 显存消耗。
        """
        # 获取输入的批次大小（B）、序列长度（T）和嵌入维度（C）
        B, T, C = x.size()

        # 通过线性层生成键和值，将嵌入维度分为两部分
        v, k = self.vk_attn(x).split(self.n_embd, dim=2)

        # 通过查询的线性层生成查询
        q = self.q_attn(x)

        # 将键和值 reshaped 为多头形式，维度为 (B, nh, T, d)，其中 nh 表示头的数量，d 表示每个头的维度
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, d)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, d)

        # 将查询 reshaped 为多头和分组形式，维度为 (B, nh, g, T, d)，g 表示查询的分组
        q = q.view(B, T, self.n_head, self.g, C // self.n_head).transpose(1, 2).transpose(2, 3)  # (B, nh, g, T, d)

        # 如果启用了 RoPE，则将旋转位置嵌入应用于查询和键
        if self.rope:
            q = self.custom_rope(q)
            k = self.custom_rope(k)

        # 清除缓存，确保准确计算显存消耗
        torch.cuda.empty_cache()
        start_memory = torch.cuda.memory_allocated()  # 获取开始时的显存消耗

        # 使用 einsum 计算注意力分数，维度为 (B, nh, g, T, T)
        att = torch.einsum('ijklm,ijmn->ijln', q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # 应用因果遮罩，确保注意力只关注到当前位置及之前的序列
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))

        # 使用 softmax 计算注意力权重，并应用 dropout 进行正则化
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # 计算注意力输出，将注意力权重与值进行点积，得到 (B, nh, T, d)
        y = att @ v

        # 获取结束时的显存消耗
        end_memory = torch.cuda.memory_allocated()

        # 重塑注意力输出，将多头输出拼接，返回 (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 输出线性变换，并应用残差 dropout 进行正则化
        y = self.resid_dropout(self.c_proj(y))

        # 返回注意力输出和显存消耗差值
        return y, end_memory - start_memory

    
#############################################################################################################

class Block(nn.Module):
    """ 一个简单的Transformer块 """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        if config.n_query_head != config.n_kv_head:
            self.attn = GroupedQueryAttention(config)
        else:
            self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP前向传播

    def forward(self, x):
        start_time = time.time()
        attn_comp, mem_consumed = self.attn(self.ln_1(x))
        end_time = time.time()
        x = x + attn_comp
        x = x + self.mlpf(self.ln_2(x))
        return x, end_time-start_time, mem_consumed



class GPT(nn.Module):
    """ GPT语言模型 """

    @staticmethod
    def get_default_config():
        C = CN()
        # 在配置中必须提供 model_type 或 (n_layer, n_head, n_embd) 之一
        C.model_type = 'gpt'
        C.n_layer = None
        C.n_query_head = None
        C.n_embd =  None
        # 这些选项必须在外部填写
        C.vocab_size = None
        C.block_size = None
        # dropout超参数
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        C.pretrained_folder = None
        C.n_kv_head = C.n_query_head
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size
        self.rope = config.rope

        modules = dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        )
        if self.rope == False:
            modules['wpe'] = nn.Embedding(config.block_size, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 初始化所有权重，并根据GPT-2论文对残差投影应用特殊的缩放初始化
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # 报告参数数量（注意，我们不会计入lm_head中的解码器参数）
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("参数数量: %.2fM" % (n_params/1e6,))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def load_pretrained(self, model_path):
        pretrained_state_dict = torch.load(os.path.join(model_path, "model.pt"))
        old_block_size = 64
        with open(os.path.join(model_path, 'config.json'), 'r') as file:
            old_config = json.load(file)
            old_block_size = old_config['data']['block_size']
        # 初始化当前模型的状态字典
        self_state_dict = self.state_dict()

        # 遍历预训练的状态字典并更新相应的权重
        for name, param in pretrained_state_dict.items():
            if name in self_state_dict:
                # 如果是wpe层且大小不同，则单独处理
                if name == 'transformer.wpe.weight' and param.size(0) != self_state_dict[name].size(0):
                    # 复制前64个神经元的权重
                    self_state_dict[name][:old_block_size, :] = param[:old_block_size, :]
                elif name.startswith("transformer.h.") and name.endswith(".attn.bias") and param.size()[2] != self_state_dict[name].size()[2]:
                    self_state_dict[name][:, :, :old_block_size, :old_block_size] = param
                    # 其余权重已经随机初始化
                else:
                    # 复制wpe以外层的权重
                    self_state_dict[name].copy_(param)

        # 加载更新后的状态字典到模型中
        self.load_state_dict(self_state_dict)

    def configure_optimizers(self, train_config):
        """
        这个长函数虽然冗长，但实际上做了非常简单的事情，并且非常防御性：
        我们将模型的所有参数分成两类：那些需要正则化权重衰减的参数，以及那些不需要的（偏置、层归一化/嵌入层权重）。
        然后我们返回PyTorch优化器对象。
        """

        # 将所有参数分开，分别标记为需要正则化权重衰减的和不需要的
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # 完整参数名
                if pn.endswith('bias'):
                    # 所有偏置项不进行权重衰减
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # 白名单模块的权重将进行权重衰减
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # 黑名单模块的权重将不进行权重衰减
                    no_decay.add(fpn)

        # 验证我们是否考虑了每个参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "参数%s同时出现在decay和no_decay集合中！" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "参数%s未被分到decay或no_decay集合中！" % (str(param_dict.keys() - union_params), )

        # 创建PyTorch优化器对象
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        attn_times = []
        mem_consumed = []
        device = idx.device
        b, t = idx.size()  # batchsize, 序列长度（即字符的个数）
        assert t <= self.block_size, f"无法处理长度为{t}的序列，块大小仅为{self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # 形状 (1, 序列长度)

        # 前向传播GPT模型
        tok_emb = self.transformer.wte(idx) # 就是nn.Embedding。输出形状为 (batchsize, 序列长度, n_embd) 
        if self.rope == False: # 直接将字符的位置信息nn.Embedding编码，然后和词嵌入融合。
            pos_emb = self.transformer.wpe(pos) # 形状为 (1, 序列长度, n_embd) 的位置嵌入
            x = self.transformer.drop(tok_emb + pos_emb)  # dropout
        else:
            x = self.transformer.drop(tok_emb)
        for block in self.transformer.h:
            x, attn_time, mem = block(x)
            mem_consumed.append(mem)
            attn_times.append(attn_time)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        # 如果我们提供了目标值，也计算损失
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss, sum(attn_times)/len(attn_times), sum(mem_consumed)/len(mem_consumed)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        """
        接受一段条件序列 idx (形状为(b,t)的LongTensor) 并补全序列 max_new_tokens 次，每次将预测结果反馈给模型。
        你很可能希望在model.eval()模式下运行此操作。
        """
        attn_times = []
        for _ in range(max_new_tokens):
            # 如果序列上下文变得过长，我们必须将其裁剪到块大小
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            # 前向传播模型以获得序列中某一索引的logits
            logits, _, attn_time, mem_consumed = self(idx_cond)
            # 提取最终步骤的logits，并按所需温度缩放
            logits = logits[:, -1, :] / temperature
            # 可选地将logits裁剪为仅前top k的选项
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
            # 应用softmax将logits转换为（归一化的）概率
            probs = F.softmax(logits, dim=-1)
            # 从分布中采样或选择最可能的元素
            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            # 将采样的索引附加到运行中的序列中并继续
            idx = torch.cat((idx, idx_next), dim=1)
            attn_times.append(attn_time)

        return idx, sum(attn_times)/len(attn_times)
