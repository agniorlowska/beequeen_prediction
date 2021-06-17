#data augmenatation
import librosa
import math
import numpy as np

#-------data augmentation-----------------#
def data_augmentation(filepath, n_chunks):
        x, sr = librosa.load(filepath)
        SNR = 30
        RMS_s=math.sqrt(np.mean(x**2))
        RMS_n=math.sqrt(RMS_s**2/(pow(10,SNR/10)))
        STD_n = RMS_n
        noise = np.random.normal(0, STD_n, x.shape)
        a = x + noise
        s = np.abs(librosa.stft(a, n_fft=1024, hop_length=512, win_length=1024, window='hann', center=True, dtype=np.complex64, pad_mode='reflect'))
        stft_mean = []
        chunks = n_chunks
        split = np.split(s, chunks, axis = 0)
        for i in range(0, chunks):
            stft_mean.append(split[i].mean(axis=0))
        stft_mean = np.asarray(stft_mean)
        return stft_mean
#------------------------------------------#  