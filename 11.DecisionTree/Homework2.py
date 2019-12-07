import numpy as np

#返回e表中符合要求的点的坐标和值
def found_e(e, x):
    e_x = []
    e_value = []
    for i in range(e.shape[0]):
        if(e[i][int(x+4)] > 0):
            e_x.append(i)
            e_value.append(e[i][int(x+4)])
        if(i == 4):
            print(e_x)
            print(e_value)
    return e_x, e_value

#返回delta_e表中符合要求的点的坐标和值
def found_delta_e(delta_e, y):
    delta_e_x, delta_e_value = 0, 0.
    for i in range(delta_e.shape[0]):
        if(delta_e[i][int(y+2)] > 0 ):
            delta_e_x = i
            delta_e_value = delta_e[i][int(y+2)]
        print(delta_e_x)
        print(delta_e_value)
    return delta_e_x, delta_e_value

#由e表和Δe表提供的坐标，查找规则表
def found_rule_table(e_x, e_value, delta_e_x, delta_e_value, rule_table, u):
    for i in range(len(e_x)):
        total, total1, total2 = [], [], []
        print(str(delta_e_x)+ ' '+str(e_x[i]))
        value = rule_table[delta_e_x][e_x[i]]
        print(value)
        if(len(e_value) == 1):
            for i in range(u.shape[1]):
                total.append(min(e_value[0], delta_e_value, u[value][i]))
        else:
            for j in range(u.shape[1]):
                total1.append(min(e_value[0], delta_e_value, u[value][j]))
                total2.append(min(e_value[1], delta_e_value, u[value][j]))
            for k in range(len(total1)):
                total.append(min(total1[k], total2[k]))
        print(total)
        sum = 0.
        for i in range(len(total)):
            sum += ( total[i] * (i - 3) )
        sum /= np.sum(total)
        print(round(sum))
        return round(sum)

#由规则表提供的坐标，查找u表并计算最终结果
# def found_u(value, u):
#     sum = 0.0
#     for i in range(u.shape[1]):
#         sum += u[value][i] * (i-3)
#     sum /= np.sum(u[value])
#     return int(sum)

#e中x轴为：-4,...,4；y轴为：PB、PS、Z、NS、NB
e = np.array([
    [0.,0.,0.,0.,0.,0.,0.,0.5,1.],
    [0.,0.,0.,0.,0.,0.5,1.,0.5,0.],
    [0.,0.,0.,0.5,1.,0.5,0.,0.,0.],
    [0.,0.5,1.,0.5,0.,0.,0.,0.,0.],
    [1.,0.5,0.5,0.,0.,0.,0.,0.,0.]
])
#Δe中x轴为：-2,...,2；y轴为：PB、PS、Z、NS、NB
delta_e = np.array([
    [0.,0.,0.,0.,1.],
    [0.,0.,0.,1.,0.],
    [0.,0.,1.,0.,0.],
    [0.,1.,0.,0.,0.],
    [1.,0.,0.,0.,0.]
])
#u中x轴为：-3,...,3；y轴为：PB、PS、Z、NS、NB
u = np.array([
    [0.,0.,0.,0.,0.,0.5,1.],
    [0.,0.,0.,0.,1.,0.5,0.],
    [0.,0.,0.5,1.,0.5,0.,0.],
    [0.,1.,0.5,0.,0.,0.,0.],
    [1.,0.5,0.,0.,0.,0.,0.]
])
#规则表rule_table中x、y轴均为:PB、PS、Z、NS、NB（分别用0、1、2、3、4表示）
rule_table = np.array([
    [4,4,3,2,0],
    [4,4,3,1,0],
    [4,3,2,1,0],
    [4,3,1,0,0],
    [4,2,1,0,0]
])
#e和Δe的论域
x = np.array([-4,-3,-2,-1,0,1,2,3,4], dtype='int')
y = np.array([-2,-1,0,1,2], dtype='int')
#最后生成的控制表，x轴为e的论域：-4,...,4，y轴为Δe的论域：-2,...,2，初始化为0
control_table = np.zeros(shape=(y.shape[0], x.shape[0]))

for i in x:
    for j in y:
        print(i)
        print(j)
        e_x, e_value = found_e(e, i)
        delta_e_x, delta_e_value = found_delta_e(delta_e, j)
        result = found_rule_table(e_x, e_value, delta_e_x, delta_e_value, rule_table, u)
        control_table[j+2][i+4] = result

print(control_table)



