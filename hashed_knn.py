import numpy as np
import random
import time
from collections import Counter
import pandas as pd
from sklearn.metrics import roc_auc_score

# Load and prepare data
filename = 'star_classification.csv'
df = pd.read_csv(filename)
data = df.pop('class')
data = np.array(data)
df = np.array(df)
df = df[:, 2:7]
X_train = df[:5000, :]
X_test = df[5000:5100, :]
Y_train = data[:5000]
Y_test = data[5000:5100]


class hashedknn:
    def __init__(self, k, w):
        self.k = k
        self.w = w

    def counter(self, Y_train):
        self.mc = Counter(Y_train).most_common(1)[0][0]

    def getHash(self, dataSet):
        '''Generate a list of hash values'''
        k = dataSet.shape[1]
        b = random.uniform(0, self.w)
        x = np.random.random(k)
        buket = []
        for data in dataSet:
            h = ((data.dot(x) + b)) // self.w
            buket.append(h)
        return buket

    def findbuket(self, X_train, hash, X_train_hash):
        '''Find indices with the same hash value and store in a list'''
        # Group by bucket
        buket = []
        for i in range(X_train.shape[0]):
            if X_train_hash[i] == hash:
                buket.append(i)
        return buket

    def predict(self, X_train, Y_train, X_test):
        '''Return predictions for Y_test'''
        X_train_hash = self.getHash(X_train)
        X_test_hash = self.getHash(X_test)
        bkt1 = []
        for i in range(X_test.shape[0]):
            b = self.findbuket(X_train, X_test_hash[i], X_train_hash)  # Store indices of X_train with same hash
            if b:
                distances = [np.sqrt(np.sum((X_train[j] - X_test[i]) ** 2)) for j in b]  # Calculate distances
                ylabel = [Y_train[j] for j in b]  # Record labels
                sort_index = np.argsort(distances)
                first_k = [ylabel[l] for l in sort_index[:self.k]]  # Get labels of k nearest neighbors
                pre = Counter(first_k).most_common(1)[0][0]
            else:
                pre = 'GALAXY'
            bkt1.append(pre)
        return bkt1

    def auccount(self, a, X_train, Y_train, X_test, Y_test):
        '''Calculate AUC'''
        # Create test matrix
        num_samples = len(Y_test)
        num_classes = 3
        Y_test_matrix = np.zeros((num_samples, num_classes))
        tag = ['GALAXY', 'STAR', 'QSO']
        for i in range(num_samples):
            aaaa = tag.index(Y_test[i])
            Y_test_matrix[i][aaaa] = 1

        bb = []
        for _ in range(a):
            Y_pre = self.predict(X_train, Y_train, X_test)
            bb.append(Y_pre)

        Y_predictions_matrix = np.array(bb)
        Y_probabilities_matrix = np.zeros((num_samples, num_classes))
        tag = ['GALAXY', 'STAR', 'QSO']
        for i in range(num_samples):
            class_counts = Counter(Y_predictions_matrix[:, i])
            for j in range(0, num_classes):
                Y_probabilities_matrix[i, j] = class_counts[tag[j]] / a
        Y_probabilities_matrix = np.array(Y_probabilities_matrix)
        auc = roc_auc_score(Y_test_matrix, Y_probabilities_matrix, multi_class='ovr', labels=[1, 2, 3])
        return auc


# Time LSH-kNN
time_start = time.time()
xiteam = hashedknn(10, 5)
aaa = xiteam.predict(X_train, Y_train, X_test)
time_end = time.time()
time_sum = time_end - time_start
print('time1: LSH-kNN', time_sum)


# Regular kNN without LSH
class knn_no_lsh:
    def __init__(self, k, fixed_hash_value):
        self.k = k
        self.fixed_hash_value = fixed_hash_value

    def getHash(self, dataSet):
        '''Generate a list with all hash values set to fixed value'''
        buket = [self.fixed_hash_value] * len(dataSet)
        return buket

    def findbuket(self, X_train, hash, X_train_hash):
        '''Find indices with the same hash value'''
        buket = []
        for i in range(X_train.shape[0]):
            if X_train_hash[i] == hash:
                buket.append(i)
        return buket

    def predict(self, X_train, Y_train, X_test):
        '''Return predictions for Y_test'''
        X_train_hash = self.getHash(X_train)
        X_test_hash = self.getHash(X_test)
        bkt1 = []
        for i in range(X_test.shape[0]):
            b = self.findbuket(X_train, X_test_hash[i], X_train_hash)
            if b:
                distances = [np.sqrt(np.sum((X_train[j] - X_test[i]) ** 2)) for j in b]
                ylabel = [Y_train[j] for j in b]
                sort_index = np.argsort(distances)
                first_k = [ylabel[l] for l in sort_index[:self.k]]
                pre = Counter(first_k).most_common(1)[0][0]
            else:
                pre = 'GALAXY'
            bkt1.append(pre)
        return bkt1

    def auccount(self, a, X_train, Y_train, X_test, Y_test):
        '''Calculate AUC'''
        num_samples = len(Y_test)
        num_classes = 3
        Y_test_matrix = np.zeros((num_samples, num_classes))
        tag = ['GALAXY', 'STAR', 'QSO']
        for i in range(num_samples):
            aaaa = tag.index(Y_test[i])
            Y_test_matrix[i][aaaa] = 1

        bb = []
        for _ in range(a):
            Y_pre = self.predict(X_train, Y_train, X_test)
            bb.append(Y_pre)

        Y_predictions_matrix = np.array(bb)
        Y_probabilities_matrix = np.zeros((num_samples, num_classes))
        tag = ['GALAXY', 'STAR', 'QSO']
        for i in range(num_samples):
            class_counts = Counter(Y_predictions_matrix[:, i])
            for j in range(0, num_classes):
                Y_probabilities_matrix[i, j] = class_counts[tag[j]] / a

        Y_probabilities_matrix = np.array(Y_probabilities_matrix)
        auc = roc_auc_score(Y_test_matrix, Y_probabilities_matrix, multi_class='ovr', labels=[1, 2, 3])
        return auc


# Time regular kNN
time_start = time.time()
no_lsh_knn = knn_no_lsh(10, 1)  # Set hash value to 1
auc_no_lsh_knn = no_lsh_knn.auccount(3, X_train, Y_train, X_test, Y_test)
time_end = time.time()
time_sum = time_end - time_start
print('time2: kNN without LSH', time_sum)