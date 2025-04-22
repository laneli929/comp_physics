import numpy as np
import random
import time
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

#所有需要的库

filename = 'star_classification.csv'
df = pd.read_csv(filename)
data=df.pop('class')
data=np.array(data)
df=np.array(df)
df=df[:,3:7]
X_train=df[:1000,:]
X_test=df[1000:1250,:]
Y_train=data[:1000]
Y_test=data[1000:1250]
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
    def plot_scatter_pca(self, X_test, Y_pre):
        # 使用PCA进行降维
        pca = PCA(n_components=2)
        X_test_pca = pca.fit_transform(X_test)

        # 绘制分类后的示意图
        colors = {'GALAXY': 'red', 'STAR': 'green', 'QSO': 'blue'}
        labels = list(set(Y_pre))

        for label in labels:
            indices = [i for i, x in enumerate(Y_pre) if x == label]
            plt.scatter(X_test_pca[indices, 0], X_test_pca[indices, 1], c=colors[label], label=label)

        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('Scatter Plot of Classified Stars (PCA)')
        plt.legend()
        plt.show()



xiteam=hashedknn(10,20)
aaa=xiteam.predict(X_train, Y_train, X_test)
xiteam.plot_scatter_pca(X_test, aaa)

