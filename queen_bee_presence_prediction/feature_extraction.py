#label and feature extraction functions
import re
import math
import os
import numpy as np
import librosa

#--------label extraction tool-------------#
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
#------------------------------------------#
  
#---------mean summarization function------#
def mean(s, n_chunks):
    m, f = s.shape
    mod = m % n_chunks
    #print(mod)
    if m % n_chunks != 0:
        s = np.delete(s, np.s_[0:mod] , 0)
    stft_mean = []
    split = np.split(s, n_chunks, axis = 0)
    for i in range(0, n_chunks):
        stft_mean.append(split[i].mean(axis=0))
    stft_mean = np.asarray(stft_mean)
    return stft_mean
#------------------------------------------#
    
#--------feature extraction tools----------#
#stft 
def stft_extraction(filepath, n_chunks):
  x, sr = librosa.load(filepath)
  s = np.abs(librosa.stft(x, n_fft=1024, hop_length=512, win_length=1024, window='hann', 
                           center=True, dtype=np.complex64, pad_mode='reflect'))
  #m, t, s = signal.stft(x, window='hann', nperseg=1025, noverlap=None, nfft=1025, detrend=False, 
                       #return_onesided=True, boundary='zeros', padded=True, axis=- 1)
  summ_s = mean(s, n_chunks)
  return summ_s

#complex stft - using scipy.stft 
def complex_stft(filepath, n_chunks):
    x, fs = librosa.load(filepath)
    zs = np.abs(librosa.stft(x, n_fft=1024, hop_length=512, win_length=1024, window='hann', 
                center=True, dtype=np.complex64, pad_mode='reflect'))
    #m, t, zs = signal.stft(x, window='hann', nperseg=1025, noverlap=None, nfft=1025, detrend=False, 
                       #return_onesided=True, boundary='zeros', padded=True, axis=- 1)
    real = zs.real
    imag = zs.imag

    summ_real = mean(real, n_chunks)
    summ_imag = mean(imag, n_chunks)
    summ_complex = summ_real**2 + summ_imag**2
    return summ_complex 
     
#cqt
def cqt_extraction(filepath, n_chunks):
    x, sr = librosa.load(filepath)
    cqt = np.abs(librosa.cqt(x, sr=sr, n_bins=513, bins_per_octave=216))
    summ_cqt = mean(cqt, n_chunks)
    return summ_cqt 

#mfccs - as a baseline
def mfccs_extraction(filepath):
  x, sr = librosa.load(filepath)
  mfccs = librosa.feature.mfcc(x, n_mfcc=20, sr=sr)
  return mfccs

#STFT without mean-spectrogram
def stft_classic(filepath):
    x, sr = librosa.load(filepath)
    s = np.abs(librosa.stft(x, n_fft=1024, hop_length=512, win_length=1024, window='hann', 
                           center=True, dtype=np.complex64, pad_mode='reflect'))
    return s

#CQT without mean-spectrogram
def  cqt_classic(filepath):
    x, sr = librosa.load(filepath)
    cqt = np.abs(librosa.cqt(x, sr=sr, n_bins=513, bins_per_octave=216))
    return cqt

#------------------------------------------------#
  
#------------approach selection------------------#
def feature_extraction(filepath, n_chunks, mode): 
  
  if mode == 0:
      s = stft_extraction(filepath, n_chunks)
  elif mode == 1:
      s = complex_stft(filepath, n_chunks)
  elif mode == 2:
      s = cqt_extraction(filepath, n_chunks)
  elif mode == 3:
      s = mfccs_extraction(filepath)
  elif mode == 4:
      s = stft_classic(filepath)
  elif mode == 5:    
      s = cqt_classic(filepath)
  return s
#------------------------------------------------#
  


