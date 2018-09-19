import numpy as np
import scipy
import h5py
import librosa
import os 

FrameSize = 512# 512 under 16KHz time=32ms(normally 20~30ms )
                # 512 under 25KHz time=20ms
                # 1024 under 25KHz time=40ms
Overlap = FrameSize//2 # half of Framesize(return the integer part)
FFTSize = FrameSize #FFT window size=FRAMESIZE
Rate = 25000
FrameWidth = 2
FrequencyBin = FrameSize//2+1 #stft_matrix:np.ndarray [shape=(1 + n_fft/2, t)]

#----------------------Set a default noisy data size--------------------------------#
noisydata = np.zeros((300000,FrequencyBin,FrameWidth*2+1),dtype=np.float32)
noisy_id = 0 # Each segment audio id number

#------------------------------Get the path-----------------------------------------#
AudioFilename='s1'
NoisyAudiopath= os.getcwd()+'\\Noisy_list\\'+AudioFilename+'\\'
CleanAudiopath= os.getcwd()+'\\Clean_list\\'+AudioFilename+'\\'
NoisyAudioList=['list_noisy_buses.txt','list_noisy_outdoor.txt','list_noisy_street.txt']

with open(NoisyAudiopath+NoisyAudioList[0], 'r') as f:
    for line in f:
        filename = line.split('/')[3].strip('\n')#read file name 
        y,sr=librosa.load(line[:-1],sr=Rate)#if sr=None to read raw sample_rate
        #librosa.stft ( Short-time Fourier transform)
        D=librosa.stft(y,n_fft=FrameSize,hop_length=Overlap,win_length=FFTSize,window=scipy.signal.hamming)#scipy.signal。hanning
        # D.shape [shape=(1 + n_fft/2, t), dtype=dtype] 
        # np.abs(D[f, t]) is the magnitude of frequency bin f at frame t
        # take log10 to improve computing efficiency
        SignaldB=np.log10(np.abs(D)**2) #np.abs() is the magnitude of frequency bin f at frame t
        #take log is easy for Calculation
        
#------------------------Normolize the data----------------------------------------#
        mean = np.mean(SignaldB, axis=1).reshape(FrequencyBin,1)
        # Take the m * n matrix as an example
        # Axis = 0: Compresses the line, averages each column, returns 1* n matrix
        # Axis = 1 : Compresses the column, averages each row, and returns m *1 matrix
        std = np.std(SignaldB, axis=1).reshape(FrequencyBin,1)
        SignaldB = (SignaldB-mean)/std

#----------------------Each 5 frames of noisy speech as a segment------------------#        
        for i in range(FrameWidth, SignaldB.shape[1]-FrameWidth): # 5 Frame
            noisydata[noisy_id,:,:] = SignaldB[:,i-FrameWidth:i+FrameWidth+1] 
            noisy_id = noisy_id + 1 #update the id number

noisydata = noisydata[:noisy_id]# Get the valid data until max id number
#print(noisydata.shape)
#np.reshape(idnumber,-1) -1:automatically calculate the number of columns in the new array.
noisydata = np.reshape(noisydata,(noisy_id,-1))

#print(noisydata.shape)
#noisydata.shape (Idnumber,1285(257*5))
#--------------Delte the last data when running again--------------------------#
if os.path.exists('data.h5'):
    os.remove('data.h5')
#Save Noisy data in data.h5
with h5py.File('data.h5', 'a') as hf:
    hf.create_dataset('trainnoise', data=noisydata)  
noisdydata = []# reset the noise data

#--------------Get the specturm features for clean data------------------------#
cleandata = np.zeros((300000,FrequencyBin),dtype=np.float32)# each frame will get one accurate value.
clean_id = 0

with open(CleanAudiopath+'list_clean.txt', 'r') as f:
    for line in f:
        filename = line.split('/')[2].strip('\n')#read name
        y,sr=librosa.load(line[:-1],sr=Rate)
        D=librosa.stft(y,n_fft=FrameSize,hop_length=Overlap,win_length=FFTSize,window=scipy.signal.hamming)
        ProcessData=np.log10(np.abs(D)**2) #the same process as the noise  
        for i in range(FrameWidth, ProcessData.shape[1]-FrameWidth):     
            cleandata[clean_id,:] = ProcessData[:,i] 
            clean_id = clean_id + 1

cleandata = cleandata[:clean_id]
#print(cleandata.shape)
#clean data shape(idnumber,257)
with h5py.File('data.h5', 'a') as hf:
    hf.create_dataset('trainclean', data=cleandata)
print('The spectrum information save as data.h5 successfully!')