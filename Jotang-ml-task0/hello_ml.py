# 1.记录姓名与成绩
scores = {
    "A":88,
    "B":92,
    "C":85
}

# 2.计算平均分，找出最高分
average = sum(scores.values()) / len(scores)
print("平均分 = ",average)

highest = max(scores.values())
print("最高分 = ",highest)

for name, score in scores.items():
    if score == highest:
        print("最高分是:",name)

# 3.NumPy矩阵乘法及输入输出结果
import numpy as np

A = np.array([
    [1,2,3],
    [4,5,6]
])
B = np.array([
    [1,2],
    [3,4],
    [5,6]
])
C = A @ B   # 也可以写成 C = np.matmul(A,B)

print("输入矩阵A:")
print(A)
print("输入矩阵A的形状:",A.shape)
print("输入矩阵B:")
print(B)
print("输入矩阵B的形状:",B.shape)
print("矩阵乘法结果:")
print(C)
print("输出矩阵C的形状:",C.shape)
