#dividing samples into folds
import librosa
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D , MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import RMSprop
from sklearn.preprocessing import LabelEncoder
from datetime import datetime 
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
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

n_chunks = 27

#stft extraction - original samples
def stft_extraction(filepath):
  x, sr = librosa.load(filepath)
  s = np.abs(librosa.stft(x, n_fft=1025, hop_length=512, win_length=1025, window='hann', center=True, dtype=np.complex64, pad_mode='reflect'))
  stft_mean = []
  chunks = n_chunks
  split = np.split(s, chunks, axis = 0)
  for i in range(0, chunks):
      stft_mean.append(split[i].mean(axis=0))
  stft_mean = np.asarray(stft_mean)
  return stft_mean

#data augmentation with white noise

def whitenoise_stft(filepath):
        x, sr = librosa.load(filepath)
        SNR = 20
        RMS_s=math.sqrt(np.mean(x**2))
        RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
        STD_n = RMS_n
        noise = np.random.normal(0, STD_n, x.shape)
        a = x + noise
        s = np.abs(librosa.stft(a, n_fft=1025, hop_length=512, win_length=1025, window='hann', center=True, dtype=np.complex64, pad_mode='reflect'))
        stft_mean = []
        chunks = n_chunks
        split = np.split(s, chunks, axis = 0)
        for i in range(0, chunks):
            stft_mean.append(split[i].mean(axis=0))
        stft_mean = np.asarray(stft_mean)
        return stft_mean
 
#--------------------------CNN model--------------------------#
# Neural Network Architecture 
model=Sequential()
model.add(Conv2D(16, kernel_size=(3,3), activation='relu', input_shape=(n_chunks,44, 1), padding='same'))
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
model.add(Conv2D(16, kernel_size=(3,1), activation='relu', padding='same'))
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
#---------------------------------------------------------------#

model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

num_epochs = 50
num_batch_size = 145

#---------------preparing training and testing data--------------#
fold1_directory = "/home/agnieszka/dataset_beehive/foldscnn/fold1"
fold2_directory = "/home/agnieszka/dataset_beehive/foldscnn/fold2"
fold3_directory = "/home/agnieszka/dataset_beehive/foldscnn/fold3"
fold4_directory = "/home/agnieszka/dataset_beehive/foldscnn/fold4"
bees_directory = "/home/agnieszka/dataset_beehive/bees"

queen_1 = []  
features_1 = []
queen_1_test = []
features_1_test = []
for filename in os.listdir(fold1_directory):
    if filename.endswith(".wav"):
        filepath = os.path.join(fold1_directory, filename)
        out = stft_extraction(filepath)
        features_1.append(out)
        features_1_test.append(out)
        q = queen_info(filepath)
        queen_1.append(q)
        queen_1_test.append(q)


features_1 = np.asarray(features_1)    
queen_1 = np.asarray(queen_1)    
features_1_test = np.asarray(features_1_test)    
queen_1_test = np.asarray(queen_1_test)   

queen_2 = []  
features_2 = []
queen_2_test = []
features_2_test = []
for filename in os.listdir(fold2_directory):
    if filename.endswith(".wav"):
        filepath = os.path.join(fold2_directory, filename)
        out =stft_extraction(filepath)
        features_2.append(out)
        features_2_test.append(out)
        q = queen_info(filepath)
        queen_2.append(q)
        queen_2_test.append(q)
        

features_2 = np.asarray(features_2)    
queen_2 = np.asarray(queen_2)    
features_2_test = np.asarray(features_2_test)    
queen_2_test = np.asarray(queen_2_test)   

queen_3 = []  
features_3 = []
queen_3_test = []
features_3_test = []
for filename in os.listdir(fold3_directory):
    if filename.endswith(".wav"):
        filepath = os.path.join(fold3_directory, filename)
        out = stft_extraction(filepath)
        features_3.append(out)
        features_3_test.append(out)
        q = queen_info(filepath)
        queen_3.append(q)
        queen_3_test.append(q)

features_3 = np.asarray(features_3)    
queen_3 = np.asarray(queen_3)  
features_3_test = np.asarray(features_3_test)    
queen_3_test = np.asarray(queen_3_test)   


queen_4 = []  
features_4 = []
queen_4_test = []
features_4_test = []
for filename in os.listdir(fold4_directory):
    if filename.endswith(".wav"):
        filepath = os.path.join(fold4_directory, filename)
        out = stft_extraction(filepath)
        features_4.append(out)
        features_4_test.append(out)
        q = queen_info(filepath)
        queen_4.append(q)
        queen_4_test.append(q)


features_4 = np.asarray(features_4)    
queen_4 = np.asarray(queen_4)    
features_4_test = np.asarray(features_4_test)    
queen_4_test = np.asarray(queen_4_test)  

#augmentation
queen_aug = []
features_aug = []
for filename in os.listdir(bees_directory):
    if filename.endswith(".wav"):
        filepath = os.path.join(bees_directory, filename)
        #out = whitenoise_mfccs
        #out=whitenoise_stft(filepath)
        #out = resampling_stft
        out=whitenoise_stft(filepath)
        print(out.shape)
        features_aug.append(out)        
        queen = queen_info(filepath)
        queen_aug.append(queen)
        
features_aug, queen_aug = shuffle(features_aug, queen_aug, random_state=0)
features_aug = np.asarray(features_aug)    
queen_aug = np.asarray(queen_aug)   
features_aug = features_aug[0:8648, :]
queen_aug = queen_aug[0:8648]
print(features_aug.shape, queen_aug.shape)
        

X1_train = np.concatenate((features_2, features_3, features_4, features_aug))
X2_train = np.concatenate((features_1, features_3, features_4, features_aug))
X3_train = np.concatenate((features_1, features_2, features_4, features_aug))
X4_train = np.concatenate((features_2, features_3, features_1, features_aug))

Y1_train = np.concatenate((queen_2, queen_3, queen_4, queen_aug))
Y2_train = np.concatenate((queen_1, queen_3, queen_4, queen_aug))
Y3_train = np.concatenate((queen_2, queen_1, queen_4, queen_aug))
Y4_train = np.concatenate((queen_2, queen_3, queen_1, queen_aug))
print
#---------------------Training Folds----------------------------#
print("Training folds...")
print("Fold 1: ")

X1_train = X1_train.reshape(-1, n_chunks, 44, 1)
X1_test = features_1_test.reshape(-1, n_chunks, 44, 1)
Y1_train = Y1_train.reshape(-1, 1)
Y1_test = queen_1_test.reshape(-1, 1)
print(X1_train.shape, X1_test.shape, Y1_train.shape, Y1_test.shape)

le = LabelEncoder()
Y1_train = to_categorical(le.fit_transform(Y1_train)) 
Y1_test = to_categorical(le.fit_transform(Y1_test)) 

#Calculate pre-training accuracy 
score = model.evaluate(X1_test, Y1_test, verbose=1)
accuracy = 100*score[1]
print("Predicted accuracy: ", accuracy)

#Training the network
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
start = datetime.now()
adam= model.fit(X1_train, Y1_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X1_test, Y1_test), verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluating the model on the training and testing set
score1 = model.evaluate(X1_train, Y1_train, verbose=1)
print("Training Accuracy: ", score1[1])
score = model.evaluate(X1_test, Y1_test, verbose=1)
print("Testing Accuracy: ", score[1])

# Visualise and list all data in history
# ---------------------------------------------------    
print(adam.history.keys())

accuracy= adam.history['accuracy']
val_accuracy=adam.history['val_accuracy']
loss=adam.history['loss']

val_loss=adam.history['val_loss']

epochs=range(len(accuracy))
fig = plt.figure()
plt.plot(epochs, accuracy,'ro',  label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
fig.savefig('accuracy.png')
plt.legend()
fig = plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
fig.savefig('loss.png')
#plt.legend()
plt.show()


#predicting
Y1_pred = model.predict(X1_test)
Y1_pred = np.argmax(np.round(Y1_pred), axis=1)
rounded_predictions_1 = model.predict_classes(X1_test, batch_size=128, verbose=0)
print(rounded_predictions_1[1])
rounded_labels_1=np.argmax(Y1_test, axis=1)
print(rounded_labels_1[1])
fig = plt.figure()

#predicting
Y1_pred = model.predict(X1_test)
Y1_pred = np.argmax(np.round(Y1_pred), axis=1)
rounded_predictions_1 = model.predict_classes(X1_test, batch_size=128, verbose=0)
print(rounded_predictions_1[1])
rounded_labels_1=np.argmax(Y1_test, axis=1)
print(rounded_labels_1[1])
fig = plt.figure()

#Confusion matrix
cnf_matrix = confusion_matrix(rounded_labels_1, rounded_predictions_1)
np.set_printoptions(precision=2)

plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix for STFT+CNN for fold1:')
plt.show()
target_names=['no queen', 'queen' ]
print ('\nClasification report for STFT+ CNN for fold1:\n', classification_report(rounded_labels_1, rounded_predictions_1, target_names=target_names ))
#-----------------------------------------------------------------------------#
print("Training folds...")
print("Fold 2: ")

X2_train = X2_train.reshape(-1, n_chunks, 44, 1)
X2_test = features_2_test.reshape(-1, n_chunks, 44, 1)
Y2_train = Y2_train.reshape(-1, 1)
Y2_test = queen_2_test.reshape(-1, 1)
print(X2_train.shape, X2_test.shape, Y2_train.shape, Y2_test.shape)

le = LabelEncoder()
Y2_train = to_categorical(le.fit_transform(Y2_train)) 
Y2_test = to_categorical(le.fit_transform(Y2_test)) 

#Calculate pre-training accuracy 
score = model.evaluate(X2_test, Y2_test, verbose=1)
accuracy = 100*score[1]
print("Predicted accuracy: ", accuracy)

#Training the network
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
start = datetime.now()
adam= model.fit(X2_train, Y2_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X2_test, Y2_test), verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluating the model on the training and testing set
score1 = model.evaluate(X2_train, Y2_train, verbose=1)
print("Training Accuracy: ", score1[1])
score = model.evaluate(X2_test, Y2_test, verbose=1)
print("Testing Accuracy: ", score[1])

# Visualise and list all data in history
# ---------------------------------------------------    
print(adam.history.keys())

accuracy= adam.history['accuracy']
val_accuracy=adam.history['val_accuracy']
loss=adam.history['loss']

val_loss=adam.history['val_loss']

epochs=range(len(accuracy))
fig = plt.figure()
plt.plot(epochs, accuracy,'ro',  label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
fig.savefig('accuracy.png')
plt.legend()
fig = plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
fig.savefig('loss.png')
#plt.legend()
plt.show()


#predicting
Y1_pred = model.predict(X1_test)
Y1_pred = np.argmax(np.round(Y1_pred), axis=1)
rounded_predictions_1 = model.predict_classes(X1_test, batch_size=128, verbose=0)
print(rounded_predictions_1[1])
rounded_labels_1=np.argmax(Y1_test, axis=1)
print(rounded_labels_1[1])
fig = plt.figure()

#predicting
Y2_pred = model.predict(X2_test)
Y2_pred = np.argmax(np.round(Y2_pred), axis=1)
rounded_predictions_2 = model.predict_classes(X2_test, batch_size=128, verbose=0)
print(rounded_predictions_2[1])
rounded_labels_2=np.argmax(Y2_test, axis=1)
print(rounded_labels_2[1])
fig = plt.figure()

#Confusion matrix
cnf_matrix = confusion_matrix(rounded_labels_2, rounded_predictions_2)
np.set_printoptions(precision=2)

plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix for STFT+CNN for fold2:')
plt.show()
target_names=['no queen', 'queen' ]
print ('\nClasification report for STFT+ CNN for fold2:\n', classification_report(rounded_labels_2, rounded_predictions_2, target_names=target_names ))
#-----------------------------------------------------------------------------#
print("Training folds...")
print("Fold 3: ")

X3_train = X3_train.reshape(-1, n_chunks, 44, 1)
X3_test = features_3_test.reshape(-1, n_chunks, 44, 1)
Y3_train = Y3_train.reshape(-1, 1)
Y3_test = queen_3_test.reshape(-1, 1)
print(X3_train.shape, Y3_train.shape)

le = LabelEncoder()
Y3_train = to_categorical(le.fit_transform(Y3_train)) 
Y3_test = to_categorical(le.fit_transform(Y3_test)) 

#Calculate pre-training accuracy 
score = model.evaluate(X3_test, Y3_test, verbose=1)
accuracy = 100*score[1]
print("Predicted accuracy: ", accuracy)

#Training the network
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
start = datetime.now()
adam= model.fit(X3_train, Y3_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X3_test, Y3_test), verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluating the model on the training and testing set
score1 = model.evaluate(X3_train, Y3_train, verbose=1)
print("Training Accuracy: ", score1[1])
score = model.evaluate(X3_test, Y3_test, verbose=1)
print("Testing Accuracy: ", score[1])

# Visualise and list all data in history
# ---------------------------------------------------    
print(adam.history.keys())

accuracy= adam.history['accuracy']
val_accuracy=adam.history['val_accuracy']
loss=adam.history['loss']

val_loss=adam.history['val_loss']

epochs=range(len(accuracy))
fig = plt.figure()
plt.plot(epochs, accuracy,'ro',  label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
fig.savefig('accuracy.png')
plt.legend()
fig = plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
fig.savefig('loss.png')
#plt.legend()
plt.show()


#predicting
Y1_pred = model.predict(X1_test)
Y1_pred = np.argmax(np.round(Y1_pred), axis=1)
rounded_predictions_1 = model.predict_classes(X1_test, batch_size=128, verbose=0)
print(rounded_predictions_1[1])
rounded_labels_1=np.argmax(Y1_test, axis=1)
print(rounded_labels_1[1])
fig = plt.figure()


#predicting
Y3_pred = model.predict(X3_test)
Y3_pred = np.argmax(np.round(Y3_pred), axis=1)
rounded_predictions_3 = model.predict_classes(X3_test, batch_size=128, verbose=0)
print(rounded_predictions_3[1])
rounded_labels_3=np.argmax(Y3_test, axis=1)
print(rounded_labels_3[1])
fig = plt.figure()

#Confusion matrix
cnf_matrix = confusion_matrix(rounded_labels_3, rounded_predictions_3)
np.set_printoptions(precision=2)

plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix for STFT+CNN for fold3:')
plt.show()
target_names=['no queen', 'queen' ]
print ('\nClasification report for STFT + CNN for fold3:\n', classification_report(rounded_labels_3, rounded_predictions_3, target_names=target_names ))
#-----------------------------------------------------------------------------#
print("Training folds...")
print("Fold 4: ")

X4_train = X4_train.reshape(-1, n_chunks, 44, 1)
X4_test = features_4_test.reshape(-1, n_chunks, 44, 1)
Y4_train = Y4_train.reshape(-1, 1)
Y4_test = queen_4_test.reshape(-1, 1)
print(X4_train.shape, Y4_train.shape)

le = LabelEncoder()
Y4_train = to_categorical(le.fit_transform(Y4_train)) 
Y4_test = to_categorical(le.fit_transform(Y4_test)) 

#Calculate pre-training accuracy 
score = model.evaluate(X4_test, Y4_test, verbose=1)
accuracy = 100*score[1]
print("Predicted accuracy: ", accuracy)

#Training the network
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)
start = datetime.now()
adam= model.fit(X4_train, Y4_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X4_test, Y4_test), verbose=1)
duration = datetime.now() - start
print("Training completed in time: ", duration)

# Evaluating the model on the training and testing set
score1 = model.evaluate(X4_train, Y4_train, verbose=1)
print("Training Accuracy: ", score1[1])
score = model.evaluate(X4_test, Y4_test, verbose=1)
print("Testing Accuracy: ", score[1])

# Visualise and list all data in history
# ---------------------------------------------------    
print(adam.history.keys())

accuracy= adam.history['accuracy']
val_accuracy=adam.history['val_accuracy']
loss=adam.history['loss']

val_loss=adam.history['val_loss']

epochs=range(len(accuracy))
fig = plt.figure()
plt.plot(epochs, accuracy,'ro',  label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
fig.savefig('accuracy.png')
plt.legend()
fig = plt.figure()
plt.plot(epochs, loss, 'ro', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
fig.savefig('loss.png')
#plt.legend()
plt.show()


#predicting
Y1_pred = model.predict(X1_test)
Y1_pred = np.argmax(np.round(Y1_pred), axis=1)
rounded_predictions_1 = model.predict_classes(X1_test, batch_size=128, verbose=0)
print(rounded_predictions_1[1])
rounded_labels_1=np.argmax(Y1_test, axis=1)
print(rounded_labels_1[1])
fig = plt.figure()

#predicting
Y4_pred = model.predict(X4_test)
Y4_pred = np.argmax(np.round(Y4_pred), axis=1)
rounded_predictions_4 = model.predict_classes(X4_test, batch_size=128, verbose=0)
print(rounded_predictions_4[1])
rounded_labels_4=np.argmax(Y4_test, axis=1)
print(rounded_labels_4[1])
fig = plt.figure()

#Confusion matrix
cnf_matrix = confusion_matrix(rounded_labels_4, rounded_predictions_4)
np.set_printoptions(precision=2)

plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix for STFT+CNN for fold4:')
plt.show()
target_names=['no queen', 'queen' ]
print ('\nClasification report for STFT + CNN for fold4:\n', classification_report(rounded_labels_4, rounded_predictions_4, target_names=target_names ))
#-----------------------------------------------------------------------------#
#Overall statistics:
rounded_predictions = np.concatenate((rounded_predictions_1, rounded_predictions_2, rounded_predictions_3, rounded_predictions_4))
rounded_labels = np.concatenate((rounded_labels_1, rounded_labels_2, rounded_labels_3, rounded_labels_4))

fig = plt.figure()
cnf_matrix = confusion_matrix(rounded_labels, rounded_predictions)
np.set_printoptions(precision=2)
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix for STFT+CNN:')
target_names=['no queen', 'queen' ]
print ('\nClasification report for STFT + CNN:\n', classification_report(rounded_labels, rounded_predictions, target_names=target_names ))










