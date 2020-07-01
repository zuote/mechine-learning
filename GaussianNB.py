import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn.metrics import classification_report

from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import math
def gnb_classify(train,test):
    labels = train.iloc[:,-1].value_counts().index #提取训练集的标签种类
    #print(labels)
    mean =[] #存放每个类别的均值
    std =[] #存放每个类别的方差
    result = [] #存放测试集的预测结果
    for i in labels:  # labels：['male', 'female']
        item = train.loc[train.iloc[:,-1]==i,:] #分别提取出每一种类别
        m = item.iloc[:,:-1].mean() #当前类别的平均值
        s = np.sum((item.iloc[:,:-1]-m)**2)/(item.shape[0]) #当前类别的方差
        mean.append(m) #将当前类别的平均值追加至列表
        std.append(s) #将当前类别的方差追加至列表
    #print(mean)
    #print(std)
    means = pd.DataFrame(mean,index=labels) #变成DF格式，索引为类标签
    stds = pd.DataFrame(std,index=labels) #变成DF格式，索引为类标签
    #print(mean)
    #print(stds)
    for j in range(test.shape[0]):
        iset = test.iloc[j,:-1].tolist() #当前测试实例
        iprob = np.exp(-((iset-means)**2)/(2 * stds**2))/(stds * np.sqrt(2 * np.pi)) #正态分布公式4

        #print(iprob)
        #print(iprob['Q25'])
        prob = 1 #初始化当前实例总概率
        label = list(iprob)
        #print(label)
        #print(iprob[label[2]])
        for k in range(test.shape[1]-1): #遍历每个特征
            prob *= iprob[label[k]] #特征概率之积即为当前实例概率
            #print(iprob[label[k]])
            #print(prob)
        cla = prob.index[np.argmax(prob.values)] #返回最大概率的类别
        result.append(cla)
    test['predict']=result
    test_male = test.loc[test['label'] == 'male']
    acc_male = (test_male.iloc[:,-1]==test_male.iloc[:,-2]).mean()
    test_female = test.loc[test['label'] == 'female']
    acc_female = (test_female.iloc[:,-1]==test_female.iloc[:,-2]).mean()
    acc = (test.iloc[:,-1]==test.iloc[:,-2]).mean() #计算预测准确率
    print(f'模型预测准确率为{acc}')
    print(f'模型男声预测准确率为{acc_male}')
    print(f'模型女声预测准确率为{acc_female}')
    return acc_male,acc_female,acc


path = "D:\\voice\\voice.csv"
dataset = pd.read_csv(path)

label_tag = list(dataset)
label_n = dataset.iloc[:,-1]
dataset = dataset.iloc[:,:-1]

#print(dataset.head())

label_n = np.array(label_n)
sim = SimpleImputer(missing_values=0,strategy ='mean')
dataset = sim.fit_transform(dataset)
dataset = preprocessing.scale(dataset)

#print(type(dataset))
#print(type(label_n))

dataset = np.column_stack((dataset,label_n))

data = pd.DataFrame(dataset,columns=label_tag)
#print(data)

acc_mean = []
acc_male_mean = []
acc_female_mean = []
for i in range(10):
    train_data,test_data = train_test_split(data,test_size=0.3)
    acc_male,acc_female,acc = gnb_classify(train_data,test_data)
    acc_male_mean.append(acc_male)
    acc_female_mean.append(acc_female)
    acc_mean.append(acc)



'''
plt.ylim(0.8, 1.0)  # 限定纵轴的范围
plt.plot(range(1,11), acc_mean, marker='o', mec='g', mfc='w',label=u'total accuracy')
plt.plot(range(1,11), acc_female_mean, marker='o', mec='b', mfc='w',label=u'female accuracy')
plt.plot(range(1,11), acc_male_mean,marker='o', mec='r', mfc='w',label=u'male accuracy')
plt.legend()  # 让图例生效
#plt.xticks(x, names, rotation=45)
plt.margins(0)
plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"Number") #X轴标签
plt.ylabel("Accuracy") #Y轴标签
plt.title("The accuracy of the GaussianNB") #标题

plt.show()
'''
