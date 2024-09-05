import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
import pandas as pd
import argparse


import wandb
from datetime import datetime
from torchsummary import summary
import torchvision.models as models


img_size = (256,256)
num_labels = 3


SAVA_IMAGE_WANDB = False # 是否上传一个batch图像到wandb
USE_GRAY = False         # 是否使用灰度转换
USE_SMALL_SIZE = False   # 是否使用28*28尺寸图像
USE_ADAMW = False        # 是否使用AdamW优化器
USE_CNN = True          # 是否使用CNN架构模型



# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

class CsvImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if idx >= self.__len__(): raise IndexError()
        img_name = self.data_frame.loc[idx, "image"]
        image = Image.open(img_name).convert("RGB")  # Assuming RGB images
        label = self.data_frame.loc[idx, "label_idx"]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_data(batch_size):
    if USE_GRAY:
        transform_img = T.Compose([
            T.ToTensor(), 
            T.Resize(min(img_size[0], img_size[1]), antialias=True),  # Resize the smallest side to 256 pixels
            T.CenterCrop(img_size),  # Center crop to 256x256
            T.Grayscale(num_output_channels=1),  # Convert to grayscale
            T.Normalize(mean=[0.485], std=[0.229]),  # Normalize the single grayscale channel
            ])
    elif USE_SMALL_SIZE:
        transform_img = T.Compose([
            T.ToTensor(), 
            T.Resize((28,28)), 
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize each color dimension
            ])
    else:
        transform_img = T.Compose([
            T.ToTensor(), 
            T.Resize(min(img_size[0], img_size[1]), antialias=True),  # Resize the smallest side to 256 pixels
            T.CenterCrop(img_size),  # Center crop to 256x256
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalize each color dimension
            ])
        
        
    train_data = CsvImageDataset(
        csv_file='./data/img_train.csv',
        transform=transform_img,
    )
    
    val_data = CsvImageDataset(
        csv_file='./data/img_val.csv',
        transform=transform_img,
    )
    test_data = CsvImageDataset(
        csv_file='./data/img_test.csv',
        transform=transform_img,
    )

    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    val_dataloader = DataLoader(val_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    for X, y in train_dataloader:
        print(f"Shape of X [B, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break
    
    return train_dataloader, val_dataloader, test_dataloader

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        
        
        if USE_GRAY:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(img_size[0] * img_size[1], 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, num_labels)
            )
        elif USE_SMALL_SIZE:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28 * 28 * 3, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, num_labels)
            )
        else:
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(img_size[0] * img_size[1] * 3, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, num_labels)
            )
            

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits




import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        
        # 第一个卷积层，kernel size = 4, stride = 4, padding = 0
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=4, padding=0)
        self.norm1 = nn.BatchNorm2d(128)  # 使用 BatchNorm2d 代替 LayerNorm
        
        # 第二个卷积层，kernel size = 7, stride = 1, padding = 3
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.norm2 = nn.BatchNorm2d(128)  # 使用 BatchNorm2d 代替 LayerNorm
        
        # 第一个卷积替代Linear层，kernel size = 3, stride = 1, padding = 1
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.gelu = nn.GELU()
        
        # 第二个卷积替代Linear层，kernel size = 3, stride = 1, padding = 1
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # 平均池化层，kernel size = 2, stride = 2
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 最后一个全连接层
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(in_features=128 * 32 * 32, out_features=3)
        
    def forward(self, x):
        # 前向传播过程
        x = self.conv1(x)
        x = self.norm1(x)  # Batch Normalization
        
        x = self.conv2(x)
        x = self.norm2(x)  # Batch Normalization
        
        x = self.conv3(x)
        x = self.gelu(x)
        
        x = self.conv4(x)
        
        x = self.avg_pool(x)
        
        x = self.flatten(x)  # Flatten
        
        x = self.fc3(x)
        
        return x

import torch
import torch.nn as nn









def log_images_to_wandb(dataloader, model, dataname):
    model.eval()  # 设置模型为评估模式
    images, labels = next(iter(dataloader))  # 获取第一个 batch
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        preds = model(images)
        pred_labels = preds.argmax(dim=1)
        
    # 创建wandb图像列表
    wandb_images = []
    
    for img, pred, true in zip(images, pred_labels, labels):
        caption = f"Pred: {pred.item()} / True: {true.item()}"
        wandb_images.append(wandb.Image(img, caption=caption))
    
    # 上传图像到wandb
    wandb.log({f"{dataname} Images": wandb_images})








def train_one_epoch(dataloader, model, loss_fn, optimizer, t):
    size = len(dataloader.dataset) #样本总数
    batch_size = dataloader.batch_size
    model.train()
    avg_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        # 记录当前batch的平均损失和已处理样本数
        loss_value = loss.item()
        current = (batch + 1) * batch_size
        
        if batch % 10 == 0:
            print(f"Train loss = {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
            
        # 记录当前batch的损失和已处理样本数
        wandb.log({"batch_loss": loss_value, "examples_seen": current})
        
        avg_loss += loss_value
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    avg_loss /= len(dataloader) # num_batches
    correct /= size
    
    #print(f"train_acc = {(100*correct):>0.1f}%, train_avg_loss = {avg_loss:>8f}")
    return 100*correct, avg_loss

        
def evaluate(dataloader, dataname, model, loss_fn):
    size = len(dataloader.dataset) #样本总数
    num_batches = len(dataloader)
    model.eval()
    avg_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            avg_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    avg_loss /= num_batches  
    correct /= size
    print(f"{dataname} accuracy = {(100*correct):>0.1f}%, {dataname} avg loss = {avg_loss:>8f}")
    return 100*correct, avg_loss
    
    
    
def main(n_epochs, batch_size, learning_rate):
    print(f"Using {device} device")
    train_dataloader,val_dataloader, test_dataloader = get_data(batch_size)
    
    if USE_CNN:
        model = CustomCNN().to(device)
    else:
        model = NeuralNetwork().to(device)

    # model = models.resnet18(pretrained=False)
    # num_classes = 3
    # model.fc = nn.Linear(model.fc.in_features, num_classes)
    # model = model.to(device)
    
    print(model)
    
    # # 方法1：打印模型参数量，假设输入图像的大小为 (3, 256, 256)
    # summary(model, input_size=(3, 256, 256))
    
    # 方法2：计算并打印模型的总参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    loss_fn = nn.CrossEntropyLoss()
    
    if USE_ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    
    for t in range(n_epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        
        # 在最后一个 epoch 的第一个batch，记录图像及其预测结果到wandb
        if SAVA_IMAGE_WANDB and t == n_epochs - 1:
            log_images_to_wandb(train_dataloader, model, "Train")
            log_images_to_wandb(val_dataloader, model, "Val")
            log_images_to_wandb(test_dataloader, model, "Test")
        
        # 训练1个epoch
        train_one_epoch(train_dataloader, model, loss_fn, optimizer, t)
        
        # 对3个集进行精度计算
        train_accuracy, train_loss = evaluate(train_dataloader, "Train", model, loss_fn)
        val_accuracy, val_loss = evaluate(val_dataloader, "Val", model, loss_fn)
        test_accuracy, test_loss = evaluate(test_dataloader, "Test", model, loss_fn)
        
        # 记录当前epoch的训练和测试指标
        wandb.log({
            "epoch": t,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy
        })
        
            
    print("Done!")

    # # Save the model
    # torch.save(model.state_dict(), "model.pth")
    # print("Saved PyTorch Model State to model.pth")

    # # Load the model (just for the sake of example)
    # model = NeuralNetwork().to(device)
    # model.load_state_dict(torch.load("model.pth"))
    
    # 结束 wandb 运行
    wandb.finish()




if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description = 'Image Classifier')
    parser.add_argument('--n_epochs', default=5, help='The number of training epochs', type=int)
    parser.add_argument('--batch_size', default=8, help='The batch size', type=int)
    parser.add_argument('--learning_rate', default=1e-3, help='The learning rate for the optimizer', type=float)

    args = parser.parse_args()
    
    # 获取当前时间戳
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    # 从文件中读取API密钥并登录
    with open("C:/wandb_key.txt", "r") as key_file:
        api_key = key_file.read().strip()
    wandb.login(key=api_key)
    
    # 构建运行名称
    run_name = "RUN_" \
           f"GRAY_{int(USE_GRAY)}_" \
           f"SMALL_{int(USE_SMALL_SIZE)}_" \
           f"ADAMW_{int(USE_ADAMW)}_" \
           f"CNN_{int(USE_CNN)}_" \
           f"SaveIMG_{int(SAVA_IMAGE_WANDB)}"
    

    run = wandb.init(
        name    = run_name,     ### Wandb creates random run names if you skip this field, we recommend you give useful names
        reinit  = True,         # 允许你在同一个Python进程中多次初始化WandB运行，这在调试或需要多次启动运行的场景中非常有用。
        project = "10423-hw0-img",  
        config  =  vars(args)   # 将argparse.Namespace对象转换为字典格式，然后传递给wandb.init()中的config参数。
    )
    
    
        
    main(args.n_epochs, args.batch_size, args.learning_rate)