import random
import torch
from torch.distributions.multinomial import Multinomial
import matplotlib.pyplot as plt

num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])  # 列表推导式，生成布尔列表
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])

fair_probs = torch.tensor([0.5, 0.5])
Multinomial(100, fair_probs).sample()  # 每次采样结果[1,0]表示正面，[0,1]表示反面

# 大样本下频率趋于概率
counts = Multinomial(10000, fair_probs).sample()
print(counts / 10000)

# 逐步增加试验次数，考察频率趋向过程
counts = Multinomial(1, fair_probs).sample((10000,))  # 进行1w次采样
cum_counts = counts.cumsum(dim=0)  # 按行累加和，相当于到目前为止正、反面出现的总次数
estimates = cum_counts / cum_counts.sum(dim=1, keepdim=True)  # 按行累加得到总次数作为分母，分式表示频率
estimates = estimates.numpy()
# 绘图
plt.figure(figsize=(4.5, 3.5))
plt.plot(estimates[:, 0], label="P(coin=heads)")
plt.plot(estimates[:, 1], label="P(coin=tails)")
plt.axhline(y=0.5, color='black', linestyle='dashed')
plt.gca().set_xlabel('Samples')
plt.gca().set_ylabel('Estimated probability')
plt.legend()
plt.show()