import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix

#path = "/home/agnieszka/dataset_beehive/dataframe_mfccs.csv"
#dataset = pd.read_csv(path)
#for hive, data in dataset.groupby('hive_name'):
#    data.to_csv("{}.csv".format(hive))
    
#---------------------------dividing data into folds--------------------------#
cf001 = pd.read_csv("/home/agnieszka/dataset_beehive/cf001 .csv")
cf003 = pd.read_csv("/home/agnieszka/dataset_beehive/cf003 .csv")
cj001 = pd.read_csv("/home/agnieszka/dataset_beehive/cj001 .csv")
gh001 = pd.read_csv("/home/agnieszka/dataset_beehive/gh001 .csv")
hive1 = pd.read_csv("/home/agnieszka/dataset_beehive/hive1.csv")
hive3 = pd.read_csv("/home/agnieszka/dataset_beehive/hive3.csv")

frames1 = [cf001, cf003]
cc = pd.concat(frames1)
test_fold1 = cc.values

frames2 = [cj001, gh001]
cg = pd.concat(frames2)
test_fold2 = cg.values

test_fold3 = hive1.values
test_fold4 = hive3.values

train_fold1 = np.concatenate((test_fold2, test_fold3, test_fold4))
train_fold2 = np.concatenate((test_fold1, test_fold3, test_fold4))
train_fold3 = np.concatenate((test_fold2, test_fold1, test_fold4))
train_fold4 = np.concatenate((test_fold2, test_fold3, test_fold1))

#------------------------SVM classification------------------------#
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
#------------------------------------------------------------------#
print('Training folds...')
print('Fold 1:')

X1_train = train_fold1[:,3:882]
Y1_train = train_fold1[:,2]
X1_train=X1_train.astype('float')
Y1_train=Y1_train.astype('int')
X1_test = test_fold1[:,3:882]
Y1_test = test_fold1[:,2]
X1_test=X1_test.astype('float')
Y1_test=Y1_test.astype('int')
clf.fit(X1_train, Y1_train)
#confusion matrix
matrix = plot_confusion_matrix(clf, X1_test, Y1_test, cmap=plt.cm.Blues, normalize=None) 
plt.title('Confusion matrix for MFCCs+SVM using cross validation for Fold1:')
plt.show(matrix)
plt.show()
#fmeasure, precision, recall, accuracy
Y1_pred = clf.predict(X1_test)
plt.show()
target_names=['no queen', 'queen' ]
print('\nClasification report for MFCCs+SVM using cross validation for Fold1:\n', classification_report(Y1_test, Y1_pred, target_names=target_names ))
#-------------------------------------------------------------------#
print('Training folds...')
print('Fold 2:')

X2_train = train_fold2[:,3:882]
Y2_train = train_fold2[:,2]
X2_train=X2_train.astype('float')
Y2_train=Y2_train.astype('int')
X2_test = test_fold2[:,3:882]
Y2_test = test_fold2[:,2]
X2_test=X2_test.astype('float')
Y2_test=Y2_test.astype('int')

clf.fit(X2_train, Y2_train)
#confusion matrix
matrix = plot_confusion_matrix(clf, X2_test, Y2_test, cmap=plt.cm.Blues, normalize=None) 
plt.title('Confusion matrix for MFCCs+SVM using cross validation for Fold2:')
plt.show(matrix)
plt.show()
#fmeasure, precision, recall, accuracy
Y2_pred = clf.predict(X2_test)
plt.show()
target_names=['no queen', 'queen' ]
print('\nClasification report for MFCCs+SVM using cross validation for Fold2:\n', classification_report(Y2_test, Y2_pred, target_names=target_names ))
#-------------------------------------------------------------------#
print('Training folds...')
print('Fold 3:')

X3_train = train_fold3[:,3:882]
Y3_train = train_fold3[:,2]
X3_train=X3_train.astype('float')
Y3_train=Y3_train.astype('int')
X3_test = test_fold3[:,3:882]
Y3_test = test_fold3[:,2]
X3_test=X3_test.astype('float')
Y3_test=Y3_test.astype('int')

clf.fit(X3_train, Y3_train)
#confusion matrix
matrix = plot_confusion_matrix(clf, X3_test, Y3_test, cmap=plt.cm.Blues, normalize=None) 
plt.title('Confusion matrix for MFCCs+SVM using cross validation for Fold3:')
plt.show(matrix)
plt.show()
#fmeasure, precision, recall, accuracy
Y3_pred = clf.predict(X3_test)
plt.show()
target_names=['no queen', 'queen' ]
#print('\nClasification report for MfCCs + SVM:\n', classification_report(Y_test, Y_pred, target_names=target_names ))
print('\nClasification report for MFCCs+SVM using cross validation for Fold3:\n', classification_report(Y3_test, Y3_pred, target_names=target_names ))
#-------------------------------------------------------------------#
print('Training folds...')
print('Fold 4:')

X4_train = train_fold4[:,3:882]
Y4_train = train_fold4[:,2]
X4_train=X4_train.astype('float')
Y4_train=Y4_train.astype('int')
X4_test = test_fold4[:,3:882]
Y4_test = test_fold4[:,2]
X4_test=X4_test.astype('float')
Y4_test=Y4_test.astype('int')
clf.fit(X4_train, Y4_train)
#confusion matrix
matrix = plot_confusion_matrix(clf, X4_test, Y4_test, cmap=plt.cm.Blues, normalize=None) 
plt.title('Confusion matrix for MFCCs+SVM using cross validation for Fold4:')
plt.show(matrix)
plt.show()
#fmeasure, precision, recall, accuracy
Y4_pred = clf.predict(X4_test)
plt.show()
target_names=['no queen', 'queen' ]
#print('\nClasification report for MfCCs + SVM:\n', classification_report(Y_test, Y_pred, target_names=target_names ))
print('\nClasification report for MFCCs+SVM using cross validation for Fold4:\n', classification_report(Y4_test, Y4_pred, target_names=target_names ))
print("Training folds completed!")
#----------------------------------------------------------------------------#

X_test_mean = np.concatenate((X1_test, X2_test, X3_test, X4_test))
Y_test_mean = np.concatenate((Y1_test, Y2_test, Y3_test, Y4_test))
matrix = plot_confusion_matrix(clf, X_test_mean, Y_test_mean, cmap=plt.cm.Blues, normalize=None) 
plt.title('Confusion matrix for MFCCs+SVM using cross validation for all the folds:')
plt.show(matrix)
plt.show()



Y_test_mean = np.concatenate((Y1_test, Y2_test, Y3_test, Y4_test))
Y_pred_mean = np.concatenate((Y1_pred, Y2_pred, Y3_pred, Y4_pred))
plt.show()
target_names=['no queen', 'queen' ]
#print('\nClasification report for MfCCs + SVM:\n', classification_report(Y_test, Y_pred, target_names=target_names ))
print('\nClasification report for MFCCs+SVM using cross validation for all the folds:\n', classification_report(Y_test_mean, Y_pred_mean, target_names=target_names ))









