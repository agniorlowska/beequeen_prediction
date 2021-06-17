#dividing samples into folds
import run 

#here you can choose the classification method:
random = 0 #
           # 0 - 4-fold cross validation
           # 1 - 70-30 random split
           
#here you can choose the approach of feature extraction:
mode = 0 #
         # 0 - mean-STFT
         # 1 - complex- mean - STFT
         # 2 - mean-CQT
         # 3 - MFCCS
         # 4 - STFT without mean spectrogram (input size 513x44)
         # 5 - CQT without mean spectrogram (input size 513x44)

#some global variables
class_names= ['no queen', 'queen' ]
target_names=['no queen', 'queen' ]
#we've got two labels, so:
n_outputs = 2
 #here you can choose value of B:
n_chunks = 16 
#some directories
fold1_directory = "/home/agnieszka/foldscnn/fold1"
fold2_directory = "/home/agnieszka/foldscnn/fold2"
fold3_directory = "/home/agnieszka/foldscnn/fold3"
fold4_directory = "/home/agnieszka/foldscnn/fold4"

queen_directory = "/home/agnieszka/dataset_beehive/queen_noqueen/queen"
noqueen_directory = "/home/agnieszka/dataset_beehive/queen_noqueen/noqueen"

#some cnn performance properties
num_epochs = 50
num_batch_size = 145

#####################################################################################

if random == 0:
    run.four_folds(fold1_directory, fold2_directory, fold3_directory, fold4_directory, n_chunks, mode, n_outputs, num_batch_size, num_epochs, class_names, target_names)
elif random == 1:
    run.random_split(queen_directory, noqueen_directory, n_chunks, mode, n_outputs, num_batch_size, num_epochs, class_names, target_names)








