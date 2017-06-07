# import libraries
import numpy as np
import pandas as pd

np.random.seed(0); 
              
# DATA PREPROCESSING
mem_use = pd.DataFrame(columns=[0, 1, 2])
for i in range(0, 110):
  a = 3 * np.random.randn(110, 1) + 3
  mem_use.loc[i, 0] = np.mean(a)
  mem_use.loc[i, 1] = np.var(a)
  mem_use.loc[i, 2] = np.std(a)             
  
peak_mem_use = pd.DataFrame(columns=[0, 1, 2])
for i in range(0, 110):
  a = 3 * np.random.randn(110, 1) + 4
  peak_mem_use.loc[i, 0] = np.mean(a)
  peak_mem_use.loc[i, 1] = np.var(a)
  peak_mem_use.loc[i, 2] = np.std(a)             

threads = pd.DataFrame(columns=[0, 1, 2])
for i in range(0, 110):
  a = np.random.randint(low = 1, high = 90, size =(110,1)) 
  threads.loc[i, 0] = np.mean(a)
  threads.loc[i, 1] = np.var(a)
  threads.loc[i, 2] = np.std(a)  
           
handles = pd.DataFrame(columns=[0, 1, 2])
for i in range(0, 110):
  a = np.random.randint(low = 1, high = 5, size =(110,1)) 
  handles.loc[i, 0] = np.mean(a)
  handles.loc[i, 1] = np.var(a)
  handles.loc[i, 2] = np.std(a) 
            
packets = pd.DataFrame(columns=[0, 1, 2])
for i in range(0, 110):
  a = np.random.randint(low = 1, high = 50, size =(110,1)) 
  packets.loc[i, 0] = np.mean(a)
  packets.loc[i, 1] = np.var(a)
  packets.loc[i, 2] = np.std(a)             

byte = pd.DataFrame(columns=[0, 1, 2])
for i in range(0, 110):
  a = np.random.randint(low = 1, high = 36, size =(110,1)) 
  byte.loc[i, 0] = np.mean(a)
  byte.loc[i, 1] = np.var(a)
  byte.loc[i, 2] = np.std(a)             

flows = pd.DataFrame(columns=[0, 1, 2])
for i in range(0, 110):
  a = np.random.randint(low = 1, high = 15, size =(110,1)) 
  flows.loc[i, 0] = np.mean(a)
  flows.loc[i, 1] = np.var(a)
  flows.loc[i, 2] = np.std(a)             

dataset = pd.concat([mem_use[0], mem_use[1], mem_use[2], 
                    peak_mem_use[0], peak_mem_use[1], peak_mem_use[2],
                    threads[0], threads[1], threads[2],
                    handles[0], handles[1], handles[2],
                    packets[0], packets[1], packets[2],
                    byte[0], byte[1], byte[2],
                    flows[0], flows[1], flows[2]], axis = 1, ignore_index = True)

# malware 1, normal -1
dataset[21] = np.negative(np.ones((110, 1)))

# create X, y
X = dataset.iloc[:, :21].values
y = dataset.iloc[:, 21].values
                              
# split dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# fit the model
from sklearn.svm import OneClassSVM
clf = OneClassSVM(nu=0.8, kernel="poly", gamma=0.1)
clf.fit(X_train)

# test it on normal training set
y_train_pred = clf.predict(X_train)

# test it on normal test set
y_test_pred = clf.predict(X_test)

# outliers
X_outlier = X_train
y_outlier = np.ones((88, 1))

for i in range(0, 88):
  for col in range(0, 21):
    X_outlier[i, col] += 50
  
# test it on normal test set
y_outlier_pred = clf.predict(X_outlier)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_test_pred)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_outlier, y_outlier_pred)
















