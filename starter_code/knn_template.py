from numpy import*
import matplotlib.pyplot as plt
import operator

def createDataSet():
    group=array([[1.0,1,1],[1.0,1.0],[0,0],[0,0.1]])
    labels=["A","A","B","B"]
    return group,labels

def file2matrix(filename):
    fr=open(filename)   #打开文件
    array_olines=fr.readlines() #从文件中读取一行
    number_lines=len(array_olines) 
    return_mat=zeros((number_lines,3))
    class_label_vector=[]
    index=0
    for line in array_olines:
        line=line.strip()
        list_from_line=line.split('\t')
        return_mat[index,:]=list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index+=1 
    return return_mat,class_label_vector

#使用系统不一样这里就需要更改
# file_path = r"C:\Users\Administrator\Desktop\lesson1\datingTestSet2.txt"
file_path = "datingTestSet2.txt"
dating_data_mat, dating_labels = file2matrix(file_path)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

fig=plt.figure()
#图形中添加一个子图
#第一个数字 "1"：将图形划分为 1 行
#第二个数字 "1"：将图形划分为 1 列
#第三个数字 "1"：选择第 1 个（也是唯一的一个）子图位置
ax=fig.add_subplot(111)  

#scatter 绘制散点图
#选择第二列作为图的x轴，选择第三列作为y轴
ax.scatter(dating_data_mat[:,1],dating_data_mat[:,2])
#第三个参数调整点的大小，第四个参数控制点的颜色
ax.scatter(dating_data_mat[:,1],dating_data_mat[:,2],15.0*array(dating_labels),15.0*array(dating_labels))
ax.set_xlabel('第二特征')
ax.set_ylabel('第三特征')
ax.set_title('特征关系散点图')
# plt.show()


def autoNorm(dataSet):
    #数据进行归一化

    minVals = dataSet.min(0) # 计算每列的最小值
    maxVals = dataSet.max(0)  # 计算每列的最大值
    ranges = maxVals - minVals  # 计算每列数据的范围（最大值-最小值）
    normDataSet = zeros(shape(dataSet))    # 创建一个与原始数据集形状相同的零矩阵，用于存储归一化后的数据
    m = dataSet.shape[0]     # 获取数据集的行数
    normDataSet = dataSet - tile(minVals, (m, 1))   # 将原始数据减去最小值（使用tile函数将minVals复制m行，使其形状与dataSet匹配）
    normDataSet = normDataSet/tile(ranges, (m, 1))     # 将减去最小值后的数据除以范围，实现归一化到[0,1]范围
    return normDataSet, ranges, minVals   # 返回归一化后的数据集、每列的范围和每列的最小值
    
def datingClassTest(hoRatio, k):
    """使用留出法对约会网站数据进行KNN分类测试

    Args:
        hoRatio (float): 测试集占总数据集的比例（0-1之间）
        k (int): K近邻算法中的邻居数量
    """
    # 使用留出法：设置测试集比例（hold-out比例），这里使用50%的数据作为测试集
    
    datingDataMat, datingLabels = file2matrix(file_path)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], k)
        print("分类结果：%d\t真实类别：%d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount += 1.0
    print("错误率：{:.2f}%".format(errorCount/float(numTestVecs)*100))

def classify0(inX, dataSet, labels, k):
    """使用k邻算法对输入向量进行分类

    Args:
        inX (array): 待分类的输入向量（数据）
        dataSet (array): 训练样本数据集
        labels (list): 标签
        k (int): 用于选择最近邻居的数目

    Returns:
        str/int: 预测的类别标签
    """
    # 计算输入向量与所有训练样本之间的  欧氏距离
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet      #将inX复制dataSetSize行，使其形状与dataSet匹配
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    
    # 对距离进行排序，获取最近的k个邻居
    sortedDistIndicies = distances.argsort()    #返回排序后的索引位置
    classCount = {}
    
    # 统计k个最近邻居中各类别的投票数
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    
    # 按照投票数降序排列，返回得票最多的类别作为预测结果
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
    
def classify_person():
    """
    #交互式输入三项特征，使用约会数据集做 KNN 分类，并输出印象结果。
    """
    resultList = ['didntLike', 'smallDoses', 'largeDoses']
    ffMiles = float(input("每年获得的飞行常客里程数："))
    percentTats = float(input("玩视频游戏所耗时间百分比："))
    iceCream = float(input("每周消费的冰淇淋公升数："))
    dating_data_mat ,dating_labels= file2matrix(file_path)
    normMat,ragnes,minValues = autoNorm(dating_data_mat) 
    inArr=array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minValues)/ragnes, normMat, dating_labels, 3)
    print(" classified as: ",resultList[classifierResult-1])
    

# —— 死循环调用 ——
if __name__ == "__main__":
    # print(autoNorm(dating_data_mat))
    # autoNorm(dating_data_mat)

    # datingClassTest(0.2,5)
    while True:
        user_input = input("按任意键继续，按 'q' 退出: ")
        if user_input.lower() == 'q':
            break
        classify_person()
