import numpy as np
import matplotlib.pyplot as plt

# 定义函数
def f(x):
    return 3 * x ** 2 - 4 * x

# 数值微分
for h in 10.0 ** np.arange(-1, -6, -1):
    approx = (f(1 + h) - f(1)) / h
    print(f'h={h:.5f}, numerical limit={approx:.5f}')


# 绘图部分
x = np.arange(0, 3, 0.1)
y = f(x)
tangent_line = 2 * x - 3  # f'(1) = 2, f(1) = -1, so y = 2(x - 1) - 1 = 2x - 3

plt.figure(figsize=(5, 3))  # 设置图像尺寸
plt.plot(x, y, label='f(x)')
plt.plot(x, tangent_line, '--', label='Tangent line (x=1)', color='red')

plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('Function and Tangent Line at x=1')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()