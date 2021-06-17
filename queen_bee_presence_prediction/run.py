import numpy as np
import prepare_data as prep
import classification as clf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import my_tools as mt


#-------4fold-cross-validation--------#
def four_folds(fold1_directory, fold2_directory, fold3_directory, fold4_directory, n_chunks, mode, n_outputs, num_batch_size, num_epochs, class_names, target_names):
    X1_train, X2_train, X3_train, X4_train, X1_test, X2_test, X3_test, X4_test, Y1_train, Y2_train, Y3_train, Y4_train, Y1_test, Y2_test, Y3_test, Y4_test = prep.make_folds(fold1_directory, fold2_directory, fold3_directory, fold4_directory, n_chunks, mode) 
    rounded_predictions_1, rounded_labels_1 = clf.train_evaluate(X1_train, Y1_train, X1_test, Y1_test, n_outputs, num_batch_size, num_epochs, class_names, target_names)
    rounded_predictions_2, rounded_labels_2 = clf.train_evaluate(X2_train, Y2_train, X2_test, Y2_test, n_outputs, num_batch_size, num_epochs, class_names, target_names)
    rounded_predictions_3, rounded_labels_3 = clf.train_evaluate(X3_train, Y3_train, X3_test, Y3_test, n_outputs, num_batch_size, num_epochs, class_names, target_names)
    rounded_predictions_4, rounded_labels_4 = clf.train_evaluate(X4_train, Y4_train, X4_test, Y4_test, n_outputs, num_batch_size, num_epochs, class_names, target_names)
    rounded_predictions = np.concatenate((rounded_predictions_1, rounded_predictions_2, rounded_predictions_3, rounded_predictions_4))
    rounded_labels = np.concatenate((rounded_labels_1, rounded_labels_2, rounded_labels_3, rounded_labels_4))
    
    cnf_matrix = confusion_matrix(rounded_labels, rounded_predictions)
    np.set_printoptions(precision=2)
    mt.plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix:')
    print ('\nClasification report:\n', classification_report(rounded_labels, rounded_predictions, target_names=target_names))
    print('Accuracy: ', accuracy_score(rounded_labels, rounded_predictions))
    df = mt.get_classification_report(rounded_labels, rounded_predictions)
    print(df)
    return df

#-------random 70-30 split--------#
def random_split(queen_directory, noqueen_directory, n_chunks, mode, n_outputs, num_batch_size, num_epochs, class_names, target_names):
    X_train, X_test, Y_train, Y_test = prep.make_random(queen_directory, noqueen_directory, n_chunks, mode)
    rounded_predictions, rounded_labels = clf.train_evaluate(X_train, Y_train, X_test, Y_test, n_outputs, num_batch_size, num_epochs, class_names, target_names)
    
    cnf_matrix = confusion_matrix(rounded_labels, rounded_predictions)
    np.set_printoptions(precision=2)
    mt.plot_confusion_matrix(cnf_matrix, classes=class_names,
                          title='Confusion matrix:')
    print ('\nClasification report:\n', classification_report(rounded_labels, rounded_predictions, target_names=target_names))
    print('Accuracy: ', accuracy_score(rounded_labels, rounded_predictions))
    df = mt.get_classification_report(rounded_labels, rounded_predictions)
    print(df)
    return df   


    