import torch
import matplotlib.pyplot as plt

x=torch.arange(4.0)
x.requires_grad_(True)
y=2*torch.dot(x,x)  # y=2x^Tx
y.backward()
print(x.grad==4*x)

x.grad.zero_()
y = x.sum()
y.backward()
print(x.grad)

x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # 给出全1梯度，目标函数为y.sum()，Faster: y.sum().backward()
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x
z.sum().backward()
print(x.grad == u)


# 复杂函数，但仍然是线性的
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
print(a.grad == d / a)

x = torch.linspace(-2 * torch.pi, 2 * torch.pi, 200, requires_grad=True)
y = torch.sin(x)
y.backward(torch.ones_like(x))  # y'=cos(x)
dy_dx = x.grad
# 转为 numpy 方便绘图
x_np = x.detach().numpy()
y_np = y.detach().numpy()
dy_dx_np = dy_dx.detach().numpy()
# 绘图
plt.plot(x_np, y_np, label='f(x) = sin(x)')
plt.plot(x_np, dy_dx_np, label="f'(x) (autograd)", linestyle='--')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('f(x) and its derivative')
plt.grid(True)
plt.show()