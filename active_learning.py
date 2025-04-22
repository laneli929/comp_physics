#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd



# 定义网络结构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # 输入层到隐藏层
        self.fc2 = nn.Linear(64, 64)  # 隐藏层
        self.fc3 = nn.Linear(64, 1)   # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 创建模型
def create_model():
    model = Net()
    optimizer = optim.Adam(model.parameters())
    return model, optimizer

# 创建五个模型
models = []
optimizers = []
for _ in range(5):
    model, optimizer = create_model()
    models.append(model)
    optimizers.append(optimizer)

# 定义损失函数
criterion = nn.MSELoss()

# 训练模型的函数
def train_model(model, optimizer, X_torch, Y_torch, epochs=10):
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_torch)
        loss = criterion(outputs, Y_torch)
        loss.backward()
        optimizer.step()
        
aa=pd.read_csv('人工智能数据1.csv',skiprows=1)        
aa=np.array(aa)
X = aa[:,0:3]  
Y = aa[:,3:4] 
# 将numpy数组转换为torch张量
X_torch = torch.tensor(X, dtype=torch.float32)
Y_torch = torch.tensor(Y, dtype=torch.float32)
# 训练每个模型
for i, model in enumerate(models):
    train_model(model, optimizers[i], X_torch, Y_torch)

nml_sample1=pd.read_csv('人工智能测试1.csv',skiprows=1)
nml_sample1=np.array(nml_sample1)
nml_samples = nml_sample1[:,0:3]

percent_diffs = []  # 用于存储每组nml的百分差
pre11=[]
# 遍历每组nml值
for nml_sample in nml_samples:
    # 将numpy数组转换为torch张量
    nml_sample_torch = torch.tensor(nml_sample, dtype=torch.float32).unsqueeze(0)

    # 使用每个模型进行预测
    predictions = []
    with torch.no_grad():
        for model in models:
            prediction = model(nml_sample_torch).numpy()[0][0]  # 获取预测值
            predictions.append(prediction)

    # 计算预测值的标准差
    std_dev = np.std(predictions)

    # 计算百分差（标准差/均值）* 100
    mean_val = np.mean(predictions)
    if mean_val != 0:  # 避免除以零
        percent_diff = (std_dev / mean_val) * 100
    else:
        percent_diff = 0
    percent_diffs.append(percent_diff)
    pre11.append(predictions)
# 找到百分差最大的索引
max_diff_idx = np.argmax(percent_diffs)
avp=0
for i in percent_diffs:
    avp+=i
avp=avp/len(percent_diffs)
print('第一次训练平均百分差为：',avp)
# 打印出百分差最大的nml组合
print("百分差最大的一组nml组合:", nml_samples[max_diff_idx])

#%%主动学习第二次

#重新训练模型
new_nml = nml_samples[max_diff_idx]  
new_w = np.array([[103]])  # 用实际测量的w值替换

#转换为torch张量
new_nml_torch = torch.tensor(new_nml, dtype=torch.float32)
new_w_torch = torch.tensor(new_w, dtype=torch.float32)

# 更新每个模型
for i, model in enumerate(models):
    optimizer = optimizers[i]
    optimizer.zero_grad()   # 清空过往梯度
    outputs = model(new_nml_torch)  # 前向传播
    loss = criterion(outputs, new_w_torch)  # 计算损失值
    loss.backward()  # 反向传播，计算当前梯度
    optimizer.step()  # 根据梯度更新网络参数

nml_sample1=pd.read_csv('人工智能测试1.csv',skiprows=1)
nml_sample1=np.array(nml_sample1)
nml_samples = nml_sample1[:,0:3]

percent_diffs = []  # 用于存储每组nml的百分差
pre11=[]
# 遍历每组nml值
for nml_sample in nml_samples:
    # 将numpy数组转换为torch张量
    nml_sample_torch = torch.tensor(nml_sample, dtype=torch.float32).unsqueeze(0)

    # 使用每个模型进行预测
    predictions = []
    with torch.no_grad():
        for model in models:
            prediction = model(nml_sample_torch).numpy()[0][0]  # 获取预测值
            predictions.append(prediction)

    # 计算预测值的标准差
    std_dev = np.std(predictions)

    # 计算百分差（标准差/均值）* 100
    mean_val = np.mean(predictions)
    if mean_val != 0:  # 避免除以零
        percent_diff = (std_dev / mean_val) * 100
    else:
        percent_diff = 0
    percent_diffs.append(percent_diff)
    pre11.append(predictions)
# 找到百分差最大的索引
max_diff_idx = np.argmax(percent_diffs)
max_diff_idx = np.argmax(percent_diffs)
avp=0
for i in percent_diffs:
    avp+=i
avp=avp/len(percent_diffs)
print('第二次训练平均百分差为：',avp)
# 打印出百分差最大的nml组合的索引
print("百分差最大的nml组合的:",nml_samples[max_diff_idx])

#进行更多次训练




