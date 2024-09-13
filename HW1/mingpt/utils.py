import os
import sys
import json
import random
from ast import literal_eval

import numpy as np
import torch

# -----------------------------------------------------------------------------

def set_seed(seed):
    """
    设置随机种子以确保结果的可重复性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_logging(config):
    """ 
    单调的记录工作 
    """
    work_dir = config.system.work_dir
    # 如果工作目录不存在，则创建
    os.makedirs(work_dir, exist_ok=True)
    # 记录传递的参数（如果有）
    with open(os.path.join(work_dir, 'args.txt'), 'w') as f:
        f.write(' '.join(sys.argv))
    # 记录配置本身
    with open(os.path.join(work_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(config.to_dict(), indent=4))

class CfgNode:
    """ 
    一个轻量级的配置类，灵感来自于yacs 
    """
    # TODO: 转换为继承自dict（类似于yacs）？
    # TODO: 实现冻结功能，以防止误操作
    # TODO: 读取/写入参数时，增加存在性/覆盖检查？

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ 
        需要一个辅助函数，以支持嵌套缩进进行漂亮的打印 
        """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ 
        返回配置的字典表示 
        """
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items()}

    def merge_from_dict(self, d):
        """ 
        从字典中更新配置 
        """
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        从预期来自命令行的字符串列表（例如sys.argv[1:]）中更新配置。

        参数应该是`--arg=value`的形式，并且arg可以使用`.`来表示嵌套的子属性。示例：

        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:
            keyval = arg.split('=')
            assert len(keyval) == 2, "每个覆盖参数应该是`--arg=value`的形式，收到的是 %s" % arg
            key, val = keyval  # 解包

            # 首先将val转换为Python对象
            try:
                val = literal_eval(val)
                """
                需要解释如下：
                - 如果val只是一个字符串，literal_eval将抛出ValueError
                - 如果val代表一个实体（如3、3.14、[1,2,3]、False、None等），则会创建该对象
                """
            except ValueError:
                pass

            # 查找适当的对象以插入属性
            assert key[:2] == '--'
            key = key[2:]  # 去掉'--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # 确保此属性存在
            assert hasattr(obj, leaf_key), f"{key} 不是配置中的有效属性"

            # 覆盖该属性
            print("命令行覆盖配置属性 %s 为 %s" % (key, val))
            setattr(obj, leaf_key, val)
