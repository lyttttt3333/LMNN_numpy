import numpy as np
from scipy.io import loadmat
from sklearn.datasets import load_wine, load_digits
from sklearn.decomposition import PCA
import random
from lmnn_impl import LMNN
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.svm import SVC
import csv

def transform(train, test):
    train_X = train[0]
    test_X = test[0]
    pca = PCA(n_components=200)
    train_X = pca.fit_transform(train_X)
    test_X = pca.transform(test_X)
    train[0] = train_X
    test[0] = test_X
    return train, test

def split_data(X,y):
    batch_num=X.shape[0]
    test_size = 0.3
    test_set_size = int(batch_num * test_size)   
    indices = list(range(batch_num))
    random.shuffle(indices)  
    train_indices=indices[test_set_size:]
    test_indices=indices[:test_set_size]
    train_X = X[train_indices,:]  
    test_X = X[test_indices,:]  
    train_y = y[train_indices]+1 
    test_y = y[test_indices]+1 
    return [train_X,train_y], [test_X, test_y]



def load_olivettifaces():
    mfile = loadmat("/home/stu10/PRMLTA_24/project_1/olivettifaces.mat")
    faces = mfile["faces"].T.copy()
    faces = np.float32(faces)
    faces = faces - faces.min()
    faces /= faces.max()
    faces = faces.reshape((400, 64, 64)).transpose(0, 2, 1)
    # 10 images per class, 400 images total, each class is contiguous.
    target = np.array([i // 10 for i in range(400)])
    faces_vectorized = faces.reshape(len(faces), -1)
    
    return faces_vectorized, target

def lmnn_fit_transform(X, y):
    lmnn = LMNN(k=3,epoch=6, mu=0.5)
    lmnn.fit(X, y)
    X_lmnn = lmnn.transform(X)
    return X_lmnn, lmnn

def test_error(knn:LMNN,train_pair, test_pair):
    result_list=[]
    train_X = train_pair[0]
    train_y = train_pair[1]
    test_X = test_pair[0]
    test_y = test_pair[1]
    for i in [0,1,2,3]:
        if i == 0:
            pred=knn.Euclidean_predict(train_X, test_X, train_y)
        if i == 1:
            pred=knn.predict(train_X, test_X, train_y)
        if i == 2:
            pred=knn.energy_predict(train_X, test_X, train_y)
        if i == 3:
            svm_clf = SVC(kernel='linear', C=1, random_state=42) 
            svm_clf.fit(train_X, train_y)
            pred = svm_clf.predict(test_X)
        result = np.abs(pred - test_y)
        result[result != 0] = 1
        rate = np.sum(result) / result.shape[0]
        if i == 0:
            print(f"Euclidean: ", rate)
            result_list.append(rate)
        if i == 1:
            print(f"LMNN: ", rate)
            result_list.append(rate)
        if i == 2:
            print(f"Energy: ", rate)
            result_list.append(rate)
        if i == 3:
            print(f"SVM: ", rate)
            result_list.append(rate)
    csv_filename = 'result.csv'  

    with open(csv_filename, mode='a', newline='') as file:  
        writer = csv.writer(file)  
        writer.writerow(result_list)  


    
if __name__ == "__main__":
    dataset_list=["faces","digits","wines"]
    dataset_name=dataset_list[1]
    for i in range(10):
        if dataset_name == "faces":
            X, y=load_olivettifaces()
            train_pair, test_pair=split_data(X,y)
            train_pair, test_pair=transform(train_pair, test_pair)
        elif dataset_name == "digits":
            data=load_digits()
            X=data["data"]
            y=data["target"]
            train_pair, test_pair=split_data(X,y)
        elif dataset_name == "wines":
            data=load_wine()
            X=data["data"]
            y=data["target"]
            train_pair, test_pair=split_data(X,y)
        X_lmnn, lmnn=lmnn_fit_transform(train_pair[0], train_pair[1])
        test_error(lmnn, train_pair=train_pair, test_pair=test_pair)



