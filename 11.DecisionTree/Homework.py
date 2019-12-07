import numpy as np
def membership(data, value):#计算每个value在少年、青年、中年的隶属度
    ship = np.zeros(shape=(4, 3)) #初始化隶属度数组为0
    for i in range(value.shape[0]): #所求的数据个数
        for j in range(data.shape[0]): #取原始数据的行数
            for k in range(data.shape[1]): #取原始数据的列数,统计value在少年、青年、中年中的次数
                if(k % 2 == 0):#取偶数列
                    if(value[i] >= data[j][k] and value[i] <= data[j][k + 1]):
                        ship[i][int(k / 2)] += 1
    print(ship / 10)
#原始数据
data = np.array([[6, 15, 16, 25, 26, 55],[7, 18, 19, 30, 31, 60],[5, 16, 17, 24, 25, 55],
                 [8, 19, 20, 30, 31, 65],[8, 16, 17, 29, 30, 60],[5, 15, 16, 27, 28, 59],
                 [5, 15, 16, 29, 30, 60],[8, 18, 19, 35, 36, 66],[10, 20, 21, 30, 31, 60],
                 [8, 16, 17, 30, 31, 58]])
#待查询数据
value = np.array([15, 25, 35, 45])
membership(data, value)



