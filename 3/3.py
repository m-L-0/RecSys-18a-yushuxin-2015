import numpy as np
import random
import math as math
def load_data(path):
    f = open(path)
    data = []
    for line in f.readlines():
        arr = []
        lines = line.strip().split("\t")
        for x in lines[2]:
            if x != "-":
                arr.append(float(x))
            else:
                arr.append(float(0))
        data.append(arr)
    return data


def gradAscent(data, k):
    dataMat = np.mat(data)
    m, n = np.shape(dataMat)
    p = np.mat(np.random.random((m, k)))
    q = np.mat(np.random.random((k, n)))
    
    alpha = 0.0002
    beta = 0.02
    maxCycles = 1000 # 梯度下降的次数
    
    for step in range(maxCycles):
        for i in range(m):
            for j in range(n):
                if dataMat[i, j] > 0:
                    error = dataMat[i, j]
                    for lk in range(k):
                        error = error - p[i, lk] * q[lk, j]
                    for lk in range(k):
                        p[i, lk] = p[i, lk] + alpha * (2 * error * q[lk, j] - beta * p[i, lk])
                        q[lk, j] = q[lk, j] + alpha * (2 * error * p[i, lk] - beta * q[lk, j])
        loss = 0.0
        for i in range(m):
            for j in range(n):
                if dataMat[i, j] > 0:
                    error = 0.0
                    for lk in range(k):
                        error = error + p[i, lk] * q[lk, j]
                    loss = (dataMat[i, j] - error) * (dataMat[i, j] - error)
                    for lk in range(k):
                        loss = loss + beta * (p[i, lk] * p[i, lk] + q[lk, j] * q[lk, j]) / 2
        if loss < 0.001:
            break
        if step == 999:
            print("loss is %f" %loss)
    return p, q


# 数据集是自己写的，在movielens的u.data的基础上对几项用户的评分改成了'-'
data = load_data("./movielens/ml-100k/yzb.test")
p, q = gradAscent(data, 5)
result = p * q
total_num = 0
correct_num = 0
for i in range(len(data)):
    if data[i][0] == 0.0:
        continue
    else:
        total_num += 1
        if result[i][0] - math.floor(result[i][0]) < 0.5:
            if math.floor(result[i][0]) == data[i][0]:
                correct_num += 1
        else:
            if math.ceil(result[i][0]) == data[i][0]:
                correct_num += 1
print("正确率：%d"%(correct_num / total_num * 100) + '%')
for i in range(len(data)):
    print("正确得分: %f" %(data[i][0]))
    print("预计得分: %f" %(result[i][0])