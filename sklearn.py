
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
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import math
import time
path = "D:\\voice\\voice.csv"
dataset = pd.read_csv(path)

#print(dataset.head())

data = dataset.iloc[:,:-1]
#data_log = math.log(data)
data_delete = data.drop(['kurt','mindom','meanfreq','skew','maxfun','median','sp.ent','meandom','maxdom', 'sd', 'modindx', 'centroid', 'Q25', 'Q75'],axis = 1)  #删除掉与其他变量有耦合关系的特征

'''
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

ax.scatter(data['meanfreq'],data['median']) 
plt.xlabel('meanfreq')
plt.ylabel('median')
plt.title('The relationship between feature "meanfreq" and "median"')

'''
#ax.scatter(data['meanfun'],data['minfun'], color='r')  # 红色
#ax.scatter(data['meanfun'],data['maxfun'], color='g')  # 绿色

'''
ax.scatter(data['Q25'],data['Q75'], color='r') 
plt.xlabel('Q25')
plt.ylabel('Q75')
plt.title('The relationship between feature "Q25" and "Q75"')


plt.show()
'''

label = dataset.iloc[:,-1]
#print(data)
#print(label.head())
label = label.map({'male':0,'female':1})
label = preprocessing.scale(label)

#print(label)

#label = LabelEncoder().fit_transform(label)

#print(label)
features_list = list(data)

sim = SimpleImputer(missing_values=0,strategy ='mean')
data = sim.fit_transform(data)
data = preprocessing.scale(data)

data_delete = sim.fit_transform(data_delete)
data_delete = preprocessing.scale(data_delete)
#print(data)


'''
# 使用sklearn的DecisionTreeClassifier判断变量的重要性
model_tree = DecisionTreeClassifier(random_state=0)
model_tree.fit(data, label)
# 获取所有变量的重要性得分
feature_importance = model_tree.feature_importances_
print(feature_importance)


sorted_idx = np.argsort(feature_importance)

plt.figure(figsize=(12,10))
plt.grid(True,)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], 0.5)
plt.yticks(range(len(sorted_idx)), np.array(features_list)[sorted_idx])

plt.rcParams['font.size'] =16
plt.rcParams['font.sans-serif'] = ['Times New Roman']  #用于刻度font
plt.xlim(0,0.1)
plt.title('Feature importance')
plt.draw()
plt.show()
'''

normal_acc = 0
male_normal_sum = 0
female_normal_sum = 0
for i in range (1000):
    train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.3) #对数据集进行分类
    gnb = GaussianNB()
    gnb.fit(train_data,train_label)
    test_predict = gnb.predict(test_data)
    score = gnb.score(test_data,test_label)
    #print(gnb.predict_log_proba(test_data))
    print("高斯准确率：%.2f%%"%(round(score*100,2)))
    #print(classification_report(test_label,test_predict))
    normal_acc += score.sum()
    male_normal_sum += ((test_label==-1) | (test_label!= test_predict)).sum()
    female_normal_sum += ((test_label==1) | (test_label!= test_predict)).sum()
normal_x = male_normal_sum/(male_normal_sum + female_normal_sum)
normal_y = female_normal_sum/(male_normal_sum + female_normal_sum)
print(normal_x,normal_y)
print(normal_acc)


del_acc = 0
male_del_sum = 0
female_del_sum = 0
for i in range (1000):
    train_data,test_data,train_label,test_label = train_test_split(data_delete,label,test_size=0.3) #对数据集进行分类
    gnb = GaussianNB()
    gnb.fit(train_data,train_label)
    test_predict = gnb.predict(test_data)
    #print(gnb.predict_log_proba(test_data))
    print("高斯准确率：%.2f%%"%(round(gnb.score(test_data,test_label)*100.0,2)))
    #print(classification_report(test_label,test_predict))
    score = gnb.score(test_data,test_label)

    del_acc += score.sum()
    male_del_sum += ((test_label==-1) | (test_label!= test_predict)).sum()
    female_del_sum += ((test_label==1) | (test_label!= test_predict)).sum()
del_x = male_del_sum/(male_del_sum + female_del_sum)
del_y = female_del_sum/(male_del_sum + female_del_sum)
print(del_x,del_y)
print(del_acc)

d = [round(normal_acc/1000,4), round(del_acc/1000,4)] #grouped sum of sales at Gender level
rects = plt.bar( (0.2,1), d, color=('r','g'), width = 0.25)
plt.ylim(0.85 , 0.98)
plt.title('The accuracy changes aftet removing "Q25" feature')
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.001*height, '%s' % float(height))
autolabel(rects)
plt.ylabel('accuracy')
plt.xticks((0.2,1),('normal_acc','del_acc'))
plt.show()

'''
gnb_acc = 0
knn_acc = 0
svc_acc = 0

start = time.time()
for i in range (1000):
    train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.3) #对数据集进行分类
    gnb = GaussianNB()


    gnb.fit(train_data,train_label)
    
    test_predict_gnb = gnb.predict(test_data)
    
    score_gnb = gnb.score(test_data,test_label)
    

    #print("%.2f%%"%(round(gnb.score(test_data,test_label)*100.0,2)))
    
    #print(classification_report(test_label,test_predict))
    gnb_acc += score_gnb.sum()
end = time.time()
time_gnb = end - start
start = time.time()
for i in range(1000):
    train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.3) #对数据集进行分类
    knn = KNeighborsClassifier()    
    knn.fit(train_data,train_label)    
    test_predict_knn = knn.predict(test_data)   
    score_knn = knn.score(test_data,test_label)    
    #print("%.2f%%"%(round(knn.score(test_data,test_label)*100.0,2)))    
    knn_acc += score_knn.sum()
end = time.time()
time_knn = end - start
start = time.time()   

for i in range(1000):
    train_data,test_data,train_label,test_label = train_test_split(data,label,test_size=0.3) #对数据集进行分类    
    svc = SVC(C=1, kernel='rbf', probability=True) 
    svc.fit(train_data,train_label)
    test_predict_svc = svc.predict(test_data)
    score_svc = svc.score(test_data,test_label)
    #print("%.2f%%"%(round(svc.score(test_data,test_label)*100.0,2)))
    svc_acc += score_svc.sum()
end = time.time()
time_svc = end - start


d = [round(gnb_acc/1000,4), round(knn_acc/1000,4), round(svc_acc/1000,4)] #grouped sum of sales at Gender level
rects = plt.bar( (0.2,1,1.8), d, color=('r','g','b'), width = 0.25)
plt.ylim(0.85 , 1)
plt.title('The compare of different classifier about accuracy')
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.001*height, '%s' % float(height))
autolabel(rects)
plt.ylabel('accuracy')
plt.xticks((0.2,1,1.8),('gnb_acc','knn_acc','svc_acc'))
plt.show()


d = [round(time_gnb,4), round(time_knn,4), round(time_svc,4)] #grouped sum of sales at Gender level
rects = plt.bar( (0.2,1,1.8), d, color=('r','g','b'), width = 0.25)
#plt.ylim(0.85 , 1)
plt.title('The compare of different classifier about performance')
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2., 1.001*height, '%s' % float(height))
autolabel(rects)
plt.ylabel('time')
plt.xticks((0.2,1,1.8),('gnb_time','knn_time','svc_time'))
plt.show()
'''




