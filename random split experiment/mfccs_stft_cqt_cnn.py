# mfccs with cnn classification
import librosa
import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
import re
from sklearn.model_selection import train_test_split
from scipy.stats import iqr
from keras.utils import to_categorical
from keras.models import Sequential, Input, Model 
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D , MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
from datetime import datetime 
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


#---------plotting confusion mqtrix--------------#
class_names= ['no queen', 'queen' ]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=25)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#---------------------------------------------------------------------------#

queen_directory = "/home/agnieszka/dataset_beehive/queen_noqueen/queen"
noqueen_directory = "/home/agnieszka/dataset_beehive/queen_noqueen/noqueen"

#is a queen inside?
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

#mfccs extraction
def mfccs_extraction(filepath):
  x, sr = librosa.load(filepath)
  mfccs = librosa.feature.mfcc(x, n_mfcc=20, sr=sr)
  return mfccs
n_chunks
#stft extraction
def stft_extraction(filepath):
  x, sr = librosa.load(filepath)
  s = np.abs(librosa.stft(x, n_fft=1025, hop_length=512, win_length=1025, window='hann', center=True, dtype=np.complex64, pad_mode='reflect'))
  stft_mean = []
  chunks = 27       
  split = np.split(s, chunks, axis = 0)
  for i in range(0, chunks):
      stft_mean.append(split[i].mean(axis=0))
      stft_mean = np.asarray(stft_mean)
  return stft_mean

#cqt extraction
def cqt_extraction(filepath):
    x, sr = librosa.load(filepath)
    cqt = np.abs(librosa.cqt(x, sr=sr, n_bins=1080, bins_per_octave=216))
    cqt_mean = []
    chunks = 40
    split = np.split(cqt, chunks, axis = 0)
    for i in range(0, chunks):
        cqt_mean.append(split[i].mean(axis=0))
    cqt_mean = np.asarray(cqt_mean)
    print(cqt_mean.shape)
    return cqt_mean
n_chunks

features_q = []
for filename in os.listdir(queen_directory):
    if filename.endswith(".wav"):
        filepath = os.path.join(queen_directory, filename)
        #out = mfccs_extraction(filepath)
        #out = stft_extraction(filepath)
        out = cqt_extraction(filepath)
        features_q.append(out)
        q = queen_info(filepath)
#preparing the data
queen_q = []  
queen_q.append(q)

features_q = np.asarray(features_q)    
queen_q = np.asarray(queen_q)    

queen_nq = []  
features_nq = []
for filename in os.listdir(noqueen_directory):
    if filename.endswith(".wav"):
        filepath = os.path.join(noqueen_directory, filename)
        #out = mfccs_extraction(filepath)
        #out = stft_extraction(filepath)
        out = cqt_extraction(filepath)
        features_nq.append(out)
        q = queen_info(filepath)
        queen_nq.append(q)

features_nq = np.asarray(features_nq)    
queen_nq = np.asarray(queen_nq)   

X_q_train, X_q_test, Y_q_train, Y_q_test = train_test_split(features_q, queen_q, test_size=0.3)
X_nq_train, X_nq_test, Y_nq_train, Y_nq_test = train_test_split(features_nq, queen_nq, test_size=0.3)

X_train = np.concatenate((X_q_train, X_nq_train))
X_test = np.concatenate((X_q_test, X_nq_test))
Y_train = np.concatenate((Y_q_train, Y_nq_train))
Y_test = np.concatenate((Y_q_test, Y_nq_test))

X_train = X_train.reshape(-1, 40, 44, 1)
X_test = X_test.reshape(-1, 40, 44, 1)
Y_train = Y_train.reshape(-1, 1)
Y_test = Y_test.reshape(-1, 1)

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


le = LabelEncoder()
Y_train = to_categorical(le.fit_transform(Y_train)) 
Y_test = to_categorical(le.fit_transform(Y_test)) 


#----------------------Convolutional Neural Network----------------------------------------------#
# Neural Network Architecture 
model=Sequential()
model.add(Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=(40,44, 1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(16, kernel_size=(3,3), activation='relu', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(16, kernel_siqrize=(3,1), activation='relu', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(16, kernel_size=(3,1), activation='relu', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Conv2D(16, kernel_size=(3,1), activation='relu', padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2), padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.25))
model.add(Dense(32 , activation='relu'))
model.add(LeakyReLU(alpha=0.1))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
print(model.summary())
#------------------------------------------------------------------------------------------------#
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])


# Calculate pre-training accuracy 
score = model.evaluate(X_test, Y_test, verbose=1)
accuracy = 100*score[1]
print("Predicted accuracy: ", accuracy)

#Training the network
num_epochs = 50
num_batch_size = 145
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
start = datetime.now()
adam= model.fit(X_train, Y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, Y_test), verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluating the model on the training and testing set
score1 = model.evaluate(X_train, Y_train, verbose=1)
print("Training Accuracy: ", score1[1])
score = model.evaluate(X_test, Y_test, verbose=1)
print("Testing Accuracy: ", score[1])

#predicting
y_pred = model.predict(X_test)
y_pred = np.argmax(np.round(y_pred), axis=1)


rounded_predictions = model.predict_classes(X_test, batch_size=128, verbose=0)
print(rounded_predictions[1])
rounded_labels=np.argmax(Y_test, axis=1)
fig = plt.figure()

cnf_matrix = confusion_matrix(rounded_labels, rounded_predictions)
np.set_printoptions(precision=2)

plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix for CQT+CNN:')
plt.show()
target_names=['no queen', 'queen' ]
print ('\nClasification report for CQT + CNN:\n', classification_report(rounded_labels, rounded_predictions, target_names=target_names ))











