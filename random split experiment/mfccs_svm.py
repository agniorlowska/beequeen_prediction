#svm classification script
import pandas as pd
import os
import re
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import plot_confusion_matrix

#------------------------------------------------------------------------------#
def hive_info(filepath):
  filename = os.path.basename(filepath)
  filename = filename.lower()
  filename = filename.strip()
  info = re.split(pattern = r"[-_]", string = filename)
  hive = info[0]
  return hive

def queen_info(filepath):
  filename = os.path.basename(filepath)
  filename = filename.lower()
  filename = filename.strip()
  info = re.split(pattern = r"[-_]", string = filename)
  #info = np.asarray(info)
  if info[1] == ' missing queen ':
    queen = 0 
  elif info[1] == ' active ':
    queen = 1
  elif info[4] == 'no':
    queen = 0
  elif info[4] == 'queenbee':
    queen = 1
  return queen

def feature_extraction(filepath):
  x, sr = librosa.load(filepath)
  mfccs = librosa.feature.mfcc(x, n_mfcc=20, sr=sr)
  
  #name of beehive
  hive_df=pd.DataFrame()
  hive = hive_info(filepath)
  #is a queen inside
  queen = queen_info(filepath)

  #beehive info
  hive_df['hive_name']=0
  hive_df['queen']=0
  hive_df.loc[0]=[hive, queen]
  
  #mfccs array to vector/
  mfccs_vec = np.reshape(mfccs,880)
  mfccs_df=pd.DataFrame()
  for i in range(0,880):
      mfccs_df['mfccs_'+str(i)]=mfccs_vec[i]
  mfccs_df.loc[0]=mfccs_vec
  

  final_df=pd.DataFrame()
  final_df=pd.concat((hive_df,mfccs_df),axis=1) 
  final_df.head()
  return final_df 
#------------------------------------------------------------------------------#
a_directory = "/home/agnieszka/dataset_beehive/bees"
df = pd.DataFrame()

for filename in os.listdir(a_directory):
    if filename.endswith(".wav"):
        filepath = os.path.join(a_directory, filename)
        out = feature_extraction(filepath)
        df = df.append(out, True)

print('{}\n'.format(df))
df.to_csv("/home/agnieszka/dataset_beehive/dataframe_mfccs", index = False, header=True)
#------------------------------------------------------------------------------#
df   = df.sort_values("queen")
array = df.values
queen_0 = array[0:8849,:]
queen_1 = array[8850:17294,:]
X0 = queen_0[:,2:882]
Y0 = queen_0[:,1]
X1 = queen_1[:,2:882]
Y1 = queen_1[:,1]
X0=X0.astype('float')
Y0=Y0.astype('int')
X1=X1.astype('float')
Y1=Y1.astype('int')
X0_train, X0_test, Y0_train, Y0_test = train_test_split(X0, Y0, test_size = 0.3)
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size = 0.3)
X_train = np.concatenate((X0_train, X1_train))
X_test = np.concatenate((X0_test, X1_test))
Y_train = np.concatenate((Y0_train, Y1_train))
Y_test = np.concatenate((Y0_test, Y1_test))
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)

#svm classifier
clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
clf.fit(X_train, Y_train)

#confusion matrix
matrix = plot_confusion_matrix(clf, X_test, Y_test, cmap=plt.cm.Blues, normalize=None) 
plt.title('Confusion matrix for MFCCs + SVM classifier:')

plt.show(matrix)
plt.show()

#fmeasure, precision, recall, accuracy
Y_pred = clf.predict(X_test)

plt.show()
target_names=['no queen', 'queen' ]
#print('\nClasification report for MfCCs + SVM:\n', classification_report(Y_test, Y_pred, target_names=target_names ))
print('\nClasification report for MFCCs + SVM:\n', classification_report(Y_test, Y_pred, target_names=target_names ))









