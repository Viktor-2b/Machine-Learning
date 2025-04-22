import torch

# 张量 初始化
x = torch.arange(12)
torch.zeros((2,3,4))# 创建指定形状的全0张量，like表示与输入张量同型
z = torch.zeros_like(x)
torch.ones((2,3,4))# 创建指定形状的全1张量
torch.randn(3,4)# 从标准正态分布中随机采样
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])# 提供嵌套列表初始化张量
x.clone()# 复制副本
x = x.reshape(3,4)# 改变张量的形状

# 张量 属性
x_shape = x.shape# 访问张量的形状，描述张量沿每个轴的长度
x.numel()# 访问张量的元素总数size 12
len(x)# 访问张量的维度

# 张量 类型转换
A=x.numpy()
B=torch.tensor(A)
type(A),type(B)
x=torch.tensor([1.1])
x,x.item(),float(x),int(x)

x=torch.tensor([1,2,3,4,5],dtype=torch.float32)
y=torch.tensor([5,4,3,2,1],dtype=torch.float32)
A=torch.arange(15,dtype=torch.float32).reshape(3,5)
B=torch.ones(5,2,dtype=torch.float32)
# 张量 按元素运算
x+y# 加减乘除幂
x-y
x*y
x/y
x**y
is_equal = x==y#逻辑运算
torch.exp(x)#求e指数
torch.dot(x,y)# 求点积
torch.mv(A,x)#矩阵-向量积
torch.mm(A,B)#矩阵乘法
torch.norm(x)#L2范数
torch.abs(x).sum()#L1范数

A_T= A.T# 矩阵转置
torch.norm(A)#矩阵Fro范数

x=torch.arange(12,dtype=torch.float32).reshape((3,4))
y=torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
# 张量 降维
x.sum()# 求和，dim指定求和维度
x_sum_axis0=x.sum(dim=0,keepdim=True)
x.mean(),x.sum()/x.numel()# 求均值，dim指定求和维度
x.mean(dim=0),x.sum(dim=0)/x.shape[0]
x_cum_sum_axis0=x.cumsum(dim=0)# 求累加值

# 张量 连接
torch.cat((x,y),dim=0)
torch.cat((x,y),dim=1)

# 张量 广播
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a+b

# 张量 索引和切片
last = x[-1]
x_range = x[1:3]#冒号选取范围
x[1,2]=9#逗号或多个中括号指定索引
x[0:2,0:1]=10
