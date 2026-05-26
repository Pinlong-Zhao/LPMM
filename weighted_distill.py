import os
import time  # 导入时间模块
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import copy
from collections import Counter
from torch.utils.tensorboard import SummaryWriter  

#=====================#
# 超参数设置
#=====================#
IMB_FACTOR = 10             
NUM_CLASSES = 10            
BATCH_SIZE = 64             
LATENT_DIM = 1028           
M = 5                       
EPOCHS_ORIG = 20            
EPOCHS_GENERATED = 20       
DISTILL_EPOCHS = 20         
LEARNING_RATE = 0.001       
MOMENTUM = 0.9              
TRAINING_EPOCHS = 50        

#=====================#
# 数据集相关代码
#=====================#

def get_img_num_per_cls(num_classes, max_num, min_num):
    img_num_per_cls = []
    alpha = (min_num / max_num) ** (1 / (num_classes - 1))
    for i in range(num_classes):
        num = int(max_num * (alpha ** i))
        img_num_per_cls.append(num)
    return img_num_per_cls

def make_longtailed_dataset(dataset, num_classes=10, imb_factor=10):
    max_num = 5000
    min_num = max_num // imb_factor
    img_num_per_cls = get_img_num_per_cls(num_classes, max_num, min_num)
    
    targets_np = np.array(dataset.targets)
    classes = np.unique(targets_np)
    
    cls_indices = []
    for c in classes:
        indices = np.where(targets_np == c)[0]
        cls_indices.append(indices)
    
    imbalanced_indices = []
    for c, idxs in enumerate(cls_indices):
        np.random.shuffle(idxs)
        select_count = img_num_per_cls[c]
        imbalanced_indices.extend(idxs[:select_count])
    
    np.random.shuffle(imbalanced_indices)
    return imbalanced_indices

class ImbalanceCIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, imb_factor=10):
        super(ImbalanceCIFAR10, self).__init__(
            root=root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download
        )
        if self.train:
            self.imbalanced_indices = make_longtailed_dataset(self, num_classes=10, imb_factor=imb_factor)
            self.data = self.data[self.imbalanced_indices]
            self.targets = list(np.array(self.targets)[self.imbalanced_indices])
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = torchvision.transforms.functional.to_pil_image(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.targets)

#=====================#
# Encoder和Decoder网络
#=====================#

class EncoderNetwork(nn.Module):
    def __init__(self, input_shape=(3, 32, 32), latent_dim=LATENT_DIM, m=M):
        super(EncoderNetwork, self).__init__()
        self.latent_dim = latent_dim
        self.m = m
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 4x4
        )
        
        self.fc_layer = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, self.m * latent_dim)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv_layer(x)  
        x = x.view(batch_size, -1)  
        
        z = self.fc_layer(x)
        z = z.view(batch_size, self.m, self.latent_dim)  
        
        return z

class DecoderNetwork(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, m=M, output_shape=(3, 32, 32)):
        super(DecoderNetwork, self).__init__()
        self.m = m
        self.output_shape = output_shape
        
        self.fc_layer = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * 4 * 4)  
        )
        
        self.deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),   
            nn.ReLU(),
            nn.ConvTranspose2d(64, output_shape[0], kernel_size=4, stride=2, padding=1)  
        )

    def forward(self, z):
        batch_size = z.size(0)
        z_flat = z.view(-1, z.size(2))  
        
        z_decoded = self.fc_layer(z_flat)  
        z_decoded = z_decoded.view(batch_size, self.m, 256, 4, 4)  

        output_images = []
        for i in range(self.m):
            output_img = self.deconv_layer(z_decoded[:, i, :, :, :].view(batch_size, 256, 4, 4))  
            output_images.append(output_img.unsqueeze(1))  
        
        return torch.cat(output_images, dim=1)  

#=====================#
# 蒸馏流程所需的辅助函数
#=====================#

def compute_uncertainty(logits):
    probs = F.softmax(logits, dim=1)
    log_probs = torch.log(probs + 1e-7)
    shannon_entropy = -torch.sum(probs * log_probs, dim=1)
    return shannon_entropy

def extract_sample_metrics(model, criterion, data_loader, device):
    model.eval()
    metrics = []
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(data_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            uncertainty = compute_uncertainty(outputs)

            torch.set_grad_enabled(True)
            model.zero_grad()
            outputs_for_grad = model(inputs)
            loss_for_grad = criterion(outputs_for_grad, labels)
            loss_for_grad.backward()
            
            grad_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item()
            torch.set_grad_enabled(False)
            
            metrics.append({
                "loss": loss.item(),
                "grad_norm": grad_norm,
                "uncertainty": uncertainty.mean().item()
            })
    return metrics

def average_metrics(metrics):
    avg_loss = np.mean([m["loss"] for m in metrics])
    avg_grad_norm = np.mean([m["grad_norm"] for m in metrics])
    avg_uncertainty = np.mean([m["uncertainty"] for m in metrics])
    return {
        "loss": avg_loss,
        "grad_norm": avg_grad_norm,
        "uncertainty": avg_uncertainty
    }

def print_metrics_average(metrics):
    print(" - 平均损失:         {:.4f}".format(metrics['loss']))
    print(" - 平均梯度范数:     {:.4f}".format(metrics['grad_norm']))
    print(" - 平均不确定性:     {:.4f}".format(metrics['uncertainty']))

def optimize_generated_samples(original_metrics, generated_metrics, optimizer):
    optimizer.zero_grad()  # 清空梯度
    
    loss_mse = nn.MSELoss()  # 使用均方误差损失
    
    # 使用原始和生成指标计算加权损失
    original_loss_tensor = torch.tensor(original_metrics['loss'], device='cuda', requires_grad=True)
    generated_loss_tensor = torch.tensor(generated_metrics['loss'], device='cuda', requires_grad=True)
    
    original_grad_norm_tensor = torch.tensor(original_metrics['grad_norm'], device='cuda', requires_grad=True)
    generated_grad_norm_tensor = torch.tensor(generated_metrics['grad_norm'], device='cuda', requires_grad=True)
    
    original_uncertainty_tensor = torch.tensor(original_metrics['uncertainty'], device='cuda', requires_grad=True)
    generated_uncertainty_tensor = torch.tensor(generated_metrics['uncertainty'], device='cuda', requires_grad=True)

    # 计算加权损失
    loss_avg = (
        loss_mse(original_loss_tensor, generated_loss_tensor) +
        loss_mse(original_grad_norm_tensor, generated_grad_norm_tensor) +
        loss_mse(original_uncertainty_tensor, generated_uncertainty_tensor)
    )

    loss_avg.backward()  # 反向传播以计算梯度
    optimizer.step()  # 更新参数

    return loss_avg.item()  # 返回损失值

def train_model(model, criterion, optimizer, dataloader, device, num_epochs=20):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss/len(dataloader):.4f}")
    return model

def evaluate_model(model, test_loader, device):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

#=======================================#
# 主函数，执行数据蒸馏流程
#=======================================#

def main():
    start_time = time.time()  # 开始计时
    device = torch.device("cuda:0")  # 硬编码为 cuda:0
    writer = SummaryWriter(log_dir='runs/my_experiment')  # 创建 TensorBoard 的 SummaryWriter 实例
    
    # 加载不平衡的 CIFAR10 数据集 (CIFAR-LT, 10:1)
    print("==> 准备不平衡 CIFAR10 数据集 (imb_factor=10) ...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2470, 0.2435, 0.2616))
    ])

    root = './data'
    train_dataset = ImbalanceCIFAR10(
        root=root,
        train=True,
        transform=transform_train,
        download=True,
        imb_factor=IMB_FACTOR  
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=False,
        transform=transform_test,
        download=True
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    train_targets_count = Counter(train_dataset.targets)
    print("-- CIFAR-LT 训练集各类别样本数量 --")
    for k in sorted(train_targets_count.keys()):
        print(f"  类别 {k}: {train_targets_count[k]}")
    print("训练集中总样本量:", len(train_dataset))

    # 用预训练ResNet-18训练20个epoch, 提取信息
    print("\n==> [步骤2] 使用预训练ResNet-18在 CIFAR-LT 上训练 20 epoch...")
    pretrained_resnet = models.resnet18(pretrained=True)
    pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, NUM_CLASSES)

    criterion = nn.CrossEntropyLoss()
    optimizer_resnet = optim.SGD(pretrained_resnet.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    
    pretrained_resnet = train_model(
        pretrained_resnet,
        criterion,
        optimizer_resnet,
        train_loader,
        device,
        num_epochs=EPOCHS_ORIG
    )
    
    # 对所有类别执行步骤3~12的数据蒸馏过程
    # 创建 Encoder 和 Decoder
    encoder = EncoderNetwork(m=M).to(device)
    decoder = DecoderNetwork(latent_dim=LATENT_DIM, m=M).to(device)

    # 最终存储每个类别的distilled data
    final_distilled_data = []
    final_distilled_labels = []

    for target_class in range(NUM_CLASSES):
        print(f"\n==> 现在开始对类别 {target_class} 执行数据蒸馏...")

        # 在筛选 target_samples 之前，对原数据集中样本进行随机权重
        all_samples = [(x, y) for x, y in train_dataset if y == target_class]  # 获取原样本
        weights = [random.choice([0, 1]) for _ in range(len(all_samples))]  # 随机生成权重

        # 根据权重筛选样本
        filtered_samples = [sample for sample, weight in zip(all_samples, weights) if weight == 1]

        # 确保筛选出的样本不为空
        if not filtered_samples:
            print(f"[警告] 类别 {target_class} 的筛选样本为空，跳过该类别的蒸馏。")
            continue
        
        target_samples_loader = DataLoader(filtered_samples, batch_size=BATCH_SIZE, shuffle=True)  # 使用经过筛选的样本
        
        for e_idx in range(DISTILL_EPOCHS):
            selected_batch = None
            for batch_samples, batch_labels in target_samples_loader:
                selected_batch = (batch_samples, batch_labels)
                break
            if selected_batch is None:
                print(f"未找到类别 {target_class} 对应样本，可能该类别在LT分布里样本过少...")
                break

            batch_samples_for_distill, batch_labels_for_distill = selected_batch
            batch_samples_for_distill = batch_samples_for_distill.to(device)

            # Step 6: 用 encoder 提取 m 个表征 z
            z = encoder(batch_samples_for_distill)

            # Step 7: decoder 将这 m 个 z 生成图像
            generated_images = decoder(z)  # 输出应为 (batch_size, m, 3, 32, 32)

            # 将当前生成的图像赋值给 distilled_data
            distilled_data = generated_images.view(-1, 3, 32, 32)  # 确保输出形状是 (num_samples * m, 3, 32, 32)

            # Step 8: 保存该类别生成的 m 张图像
            print(f"\n[类别{target_class} - 蒸馏迭代{e_idx+1}/{DISTILL_EPOCHS}] 生成图像维度: {generated_images.shape}")

            gen_labels = torch.full((distilled_data.size(0),), target_class, dtype=torch.long).to(device)  # 根据生成图像的数量初始化标签

            # 这一步确保生成的数据和标签的大小是匹配的
            assert distilled_data.size(0) == gen_labels.size(0), "生成数据和标签大小不匹配"

            gen_dataset = torch.utils.data.TensorDataset(distilled_data, gen_labels)
            gen_loader = DataLoader(gen_dataset, batch_size=1, shuffle=False)

            distilled_resnet = copy.deepcopy(pretrained_resnet).to(device)
            distilled_optimizer = optim.SGD(distilled_resnet.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

            distilled_resnet.train()
            original_metrics_list = []
            generated_metrics_list = []

            for epoch in range(EPOCHS_GENERATED):
                print(f"[类别{target_class}] 第{e_idx + 1}次蒸馏迭代中，进行生成模型训练，第{epoch + 1}/{EPOCHS_GENERATED}轮...")
                
                # 提取原样本的指标，逐一提取每个样本
                original_metrics = []
                for input_img, label in DataLoader(filtered_samples, batch_size=1, shuffle=True):
                    original_metrics.extend(extract_sample_metrics(distilled_resnet, criterion, [(input_img.to(device), label.to(device))], device))
                
                # 提取生成样本的指标
                generated_metrics = extract_sample_metrics(
                    distilled_resnet,
                    criterion,
                    gen_loader,
                    device
                )
                
                # 记录 TensorBoard 信息
                for i in range(len(generated_metrics)):
                    writer.add_scalar(f'Loss/class_{target_class}/epoch_{epoch}', generated_metrics[i]["loss"], i)

                original_metrics_list.extend(original_metrics)
                generated_metrics_list.extend(generated_metrics)

                print(f"[类别{target_class}] 第{epoch + 1}轮损失: {average_metrics(generated_metrics)['loss']:.4f}")

            averaged_original_metrics = average_metrics(original_metrics_list)
            averaged_generated_metrics = average_metrics(generated_metrics_list)

            print("-- 生成数据（步骤9）平均指标 --")
            print_metrics_average(averaged_generated_metrics)

            optimize_generated_samples(averaged_original_metrics, averaged_generated_metrics, distilled_optimizer)

        # 整合当前类的生成样本数据
        final_distilled_data.append(distilled_data)
        final_distilled_labels.append(gen_labels)

    # 整合所有类别的生成样本成训练集
    final_distilled_data = torch.cat(final_distilled_data)
    final_distilled_labels = torch.cat(final_distilled_labels)

    # 训练模型使用整合后的蒸馏数据集
    print("\n==> 使用整合的蒸馏数据集训练 ResNet-18 ...")
    distilled_train_loader = DataLoader(
        torch.utils.data.TensorDataset(final_distilled_data, final_distilled_labels),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    distilled_resnet = models.resnet18(pretrained=True)
    distilled_resnet.fc = nn.Linear(distilled_resnet.fc.in_features, NUM_CLASSES)
    
    distilled_optimizer = optim.SGD(distilled_resnet.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    trained_distilled_resnet = train_model(
        distilled_resnet,
        criterion,
        distilled_optimizer,
        distilled_train_loader,
        device,
        num_epochs=TRAINING_EPOCHS
    )

    # 使用原训练集进行训练
    print("\n==> 使用原训练集训练 ResNet-18 ...")
    original_train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    original_resnet = models.resnet18(pretrained=True)
    original_resnet.fc = nn.Linear(original_resnet.fc.in_features, NUM_CLASSES)
    
    original_optimizer = optim.SGD(original_resnet.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    trained_original_resnet = train_model(
        original_resnet,
        criterion,
        original_optimizer,
        original_train_loader,
        device,
        num_epochs=TRAINING_EPOCHS
    )

    # 测试集评估
    print("\n==> 测试整合生成的模型 ...")
    test_accuracy_distilled = evaluate_model(trained_distilled_resnet, test_loader, device)
    print(f"整合生成模型测试集准确率: {test_accuracy_distilled:.2f}%")
    
    print("\n==> 测试原始训练模型 ...")
    test_accuracy_original = evaluate_model(trained_original_resnet, test_loader, device)
    print(f"原始训练模型测试集准确率: {test_accuracy_original:.2f}%")

    print("\n全部流程结束。")
    end_time = time.time()  # 结束计时
    print(f"程序总共运行时间: {end_time - start_time:.2f} 秒")
    
    writer.close()  # 关闭 TensorBoard Writer

# 程序入口
if __name__ == "__main__":
    main()