import os
from sklearn.model_selection import train_test_split
import feature_extraction as ft
import data_augmentation as da
import numpy as np

#-----------------4-fold-cross validation----------------------#
def make_folds(fold1_directory, fold2_directory, fold3_directory, fold4_directory, n_chunks, mode):
    queen_1 = []  
    features_1 = []
    for filename in os.listdir(fold1_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(fold1_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_1.append(out)
            q =ft.queen_info(filepath)
            queen_1.append(q)
                 
    features_1 = np.asarray(features_1)    
    queen_1 = np.asarray(queen_1)    
    
    queen_2 = []  
    features_2 = []
    for filename in os.listdir(fold2_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(fold2_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_2.append(out)
            q = ft.queen_info(filepath)
            queen_2.append(q)
            
    features_2 = np.asarray(features_2)    
    queen_2 = np.asarray(queen_2)    
    
    queen_3 = []  
    features_3 = []
    for filename in os.listdir(fold3_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(fold3_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_3.append(out)
            q = ft.queen_info(filepath)
            queen_3.append(q)
    features_3 = np.asarray(features_3)    
    queen_3 = np.asarray(queen_3)    
    
    queen_4 = []  
    features_4 = []
    for filename in os.listdir(fold4_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(fold4_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_4.append(out)
            q = ft.queen_info(filepath)
            queen_4.append(q)
    
    features_4 = np.asarray(features_4)    
    queen_4 = np.asarray(queen_4)      
    
    #data augmentation - for conducting experiment without data augmentation, 
    # comment lines below and uncomment block "no data augmentation" (line 98)
    queen_aug = []  
    features_aug = []
    for filename in os.listdir(fold1_directory):
            filepath = os.path.join(fold1_directory, filename)
            out = da.data_augmentation(filepath, n_chunks)
            features_aug.append(out)
            q = ft.queen_info(filepath)
            queen_aug.append(q)
            
    for filename in os.listdir(fold2_directory):
            filepath = os.path.join(fold2_directory, filename)
            out = da.data_augmentation(filepath, n_chunks)
            features_aug.append(out)
            q = ft.queen_info(filepath)
            queen_aug.append(q)  
            
    for filename in os.listdir(fold3_directory):
            filepath = os.path.join(fold3_directory, filename)
            out = da.data_augmentation(filepath, n_chunks)
            features_aug.append(out)
            q = ft.queen_info(filepath)
            queen_aug.append(q)        

    features_aug = np.asarray(features_aug)    
    queen_aug = np.asarray(queen_aug)    
    
    X1_train = np.concatenate((features_2, features_3, features_4, features_aug))
    X2_train = np.concatenate((features_1, features_3, features_4, features_aug))
    X3_train = np.concatenate((features_1, features_2, features_4, features_aug))
    X4_train = np.concatenate((features_2, features_3, features_1, features_aug))
    Y1_train = np.concatenate((queen_2, queen_3, queen_4, queen_aug))
    Y2_train = np.concatenate((queen_1, queen_3, queen_4, queen_aug))
    Y3_train = np.concatenate((queen_1, queen_2, queen_4, queen_aug))
    Y4_train = np.concatenate((queen_2, queen_3, queen_1, queen_aug))
    #--------------------------------------------------------------#
    
    #no data augmentation
#    X1_train = np.concatenate((features_2, features_3, features_4))
#    X2_train = np.concatenate((features_1, features_3, features_4))
#    X3_train = np.concatenate((features_1, features_2, features_4))
#    X4_train = np.concatenate((features_2, features_3, features_1))
#    
#    Y1_train = np.concatenate((queen_2, queen_3, queen_4))
#    Y2_train = np.concatenate((queen_1, queen_3, queen_4))
#    Y3_train = np.concatenate((queen_1, queen_2, queen_4))
#    Y4_train = np.concatenate((queen_2, queen_3, queen_1))
    #--------------------------------------------------------------#
    
    X1_test = features_1
    X2_test = features_2
    X3_test = features_3
    X4_test = features_4
    
    Y1_test = queen_1
    Y2_test = queen_2
    Y3_test = queen_3
    Y4_test = queen_4
    
    return X1_train, X2_train, X3_train, X4_train, X1_test, X2_test, X3_test, X4_test, Y1_train, Y2_train, Y3_train, Y4_train, Y1_test, Y2_test, Y3_test, Y4_test
  
#-------------70-30 split experiment--------------------#
          
def make_random(queen_directory, noqueen_directory, n_chunks, mode):
    features_q = []
    queen_q = []  
    for filename in os.listdir(queen_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(queen_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_q.append(out)
            q = ft.queen_info(filepath)
            queen_q.append(q)
    
    features_q = np.asarray(features_q)    
    queen_q = np.asarray(queen_q)    
    
    queen_nq = []  
    features_nq = []
    for filename in os.listdir(noqueen_directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(noqueen_directory, filename)
            out = ft.feature_extraction(filepath, n_chunks, mode)
            features_nq.append(out)
            q = ft.queen_info(filepath)
            queen_nq.append(q)
    
    features_nq = np.asarray(features_nq)    
    queen_nq = np.asarray(queen_nq)   
    
    X_q_train, X_q_test, Y_q_train, Y_q_test = train_test_split(features_q, queen_q, test_size=0.3)
    X_nq_train, X_nq_test, Y_nq_train, Y_nq_test = train_test_split(features_nq, queen_nq, test_size=0.3)
    
    X_train = np.concatenate((X_q_train, X_nq_train))
    X_test = np.concatenate((X_q_test, X_nq_test))
    Y_train = np.concatenate((Y_q_train, Y_nq_train))
    Y_test = np.concatenate((Y_q_test, Y_nq_test))
    return X_train, X_test, Y_train, Y_test


    
    
    