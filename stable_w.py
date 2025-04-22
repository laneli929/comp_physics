import numpy as np
import random
import time
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
#所有需要的库

time_start = time.time()  # 记录开始时间
filename = 'star_classification.csv'
df = pd.read_csv(filename)
data=df.pop('class')
data=np.array(data)
df=np.array(df)
df=df[:,2:7]
X_train=df[:7000,:]
X_test=df[7000:7500,:]
Y_train=data[:7000]
Y_test=data[7000:7500]
#创建用于计算AUC的Ytest矩阵


class hashedknn:
    def __init__(self,k,w):
        self.k=k
        self.w=w
        
    def counter(self,Y_train):
        self.mc=Counter(Y_train).most_common(1)[0][0]
        
    def getHash(self,dataSet):
        '''生成一个存储哈希值的列表'''
        k = dataSet.shape[1]
        b = random.uniform(0, self.w)
        x = np.random.random(k)
        buket=[]
        for data in dataSet:
            h=((data.dot(x)+b)//self.w)
            buket.append(h)
        return buket
    
    def findbuket(self,X_train,hash,X_train_hash):
        '''寻找和哈希值相同哈希值的索引并储存在一个列表中'''
        #按照小桶归类
        buket=[]
        for i in range(X_train.shape[0]):
            if X_train_hash[i]==hash:
                buket.append(i)
        return buket
    
    def predict(self,X_train,Y_train,X_test):
        '''返回对Y_test的预测'''
        X_train_hash=self.getHash(X_train)
        X_test_hash=self.getHash(X_test)
        bkt1=[]
        for i in range(X_test.shape[0]):
            b=self.findbuket(X_train , X_test_hash[i] , X_train_hash)#把X_train中哈希值相同元素的下标记录到b中
            if b:
                distances=[np.sqrt(np.sum((X_train[j]-X_test[i])**2)) for j in b]#计算距离数组
                #distance：[1,2,3]
                ylabel=[Y_train[j] for j in b]#记录其label
                sort_index=np.argsort(distances)
                #记录下距离最长的几个的索引
                first_k=[ylabel[l] for l in sort_index[:self.k]]
                #返回该索引下的label
                pre=Counter(first_k).most_common(1)[0][0]
            else:
                pre='GALAXY'
            bkt1.append(pre)
        return bkt1

    def auccount(self,a,X_train,Y_train,X_test,Y_test):
        '''计算AUC'''
        #给出test矩阵
        num_samples = len(Y_test)
        num_classes = 3
        Y_test_matrix = np.zeros((num_samples, num_classes))
        tag=['GALAXY','STAR','QSO']
        for i in range(num_samples):
            aaaa=tag.index(Y_test[i])
            Y_test_matrix[i][aaaa]=1
        #给出predict矩阵 
        Y_test_matrix =np.array(Y_test_matrix)
        
        bb=[]
        for _ in range(a):
            Y_pre = self.predict(X_train,Y_train,X_test)
            bb.append(Y_pre)
        # Convert predictions to a NumPy array
        Y_predictions_matrix = np.array(bb) 
        Y_probabilities_matrix = np.zeros((num_samples, num_classes))
        tag=['GALAXY','STAR','QSO']
        for i in range(num_samples):
            class_counts = Counter(Y_predictions_matrix[:, i])
            for j in range(0, num_classes):
                Y_probabilities_matrix[i, j] = class_counts[tag[j]] / a
        Y_probabilities_matrix=np.array(Y_probabilities_matrix)
        auc = roc_auc_score(Y_test_matrix,Y_probabilities_matrix,multi_class='ovr',labels=[1, 2, 3])
        # Print the probability matrix
        return auc

time_end = time.time()  # 记录结束时间
time_sum = time_end - time_start  # 计算的时间差为程序的执行时间，单位为秒/s
print(time_sum)
xiteamauc1=hashedknn(10, 1)
auc1=xiteamauc1.auccount(1, X_train, Y_train, X_test, Y_test)
print(auc1)
w=[]
auc=[]
for i in range(20,70):
    xiteam=hashedknn(10,i)
    aaa=xiteam.predict(X_train, Y_train, X_test)
    bbb=xiteam.auccount(3, X_train, Y_train, X_test, Y_test)
    w.append(i)
    auc.append(bbb)

plt.title('auc-w')
plt.plot(w,auc)
plt.xlabel('w')
plt.ylabel('auc')