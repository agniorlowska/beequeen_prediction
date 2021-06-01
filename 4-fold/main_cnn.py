#dividing samples into folds
import my_tools as mt


#some global variables
class_names= ['no queen', 'queen' ]
mode = 2 #here you can choose the approach of feature extraction:
         # 0 - mean-STFT
         # 1 - complex- mean - STFT
         # 2 - mean-CQT
         # 3 - MFCCS
n_chunks = 27 #here you can choose value of B
#some directories
fold1_directory = "/home/agnieszka/dataset_beehive/foldscnn/fold1"
fold2_directory = "/home/agnieszka/dataset_beehive/foldscnn/fold2"
fold3_directory = "/home/agnieszka/dataset_beehive/foldscnn/fold3"
fold4_directory = "/home/agnieszka/dataset_beehive/foldscnn/fold4"
foldaug_directory = "/home/agnieszka/dataset_beehive/foldscnn1/fold_aug"
#some cnn performance properties
num_epochs = 50
num_batch_size = 145

#####################################################################################

mt.run(mode, n_chunks, fold1_directory, fold2_directory, 
      fold3_directory, fold4_directory, foldaug_directory,
      num_batch_size, num_epochs)











