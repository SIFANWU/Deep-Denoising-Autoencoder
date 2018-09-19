from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation,Flatten
from keras.layers.normalization import BatchNormalization
from keras import backend as K 
from keras.engine import Layer
import scipy.io
import scipy.io.wavfile as wav
import numpy as np
import librosa
import os
import math

#-------------------------------Build My activation function--------------------------------#
class My_activation(Layer):
    """
    f(x) = x for x>0
    f(x) = (1+exp(-x))**(-1) for x<=0
  
    """
    def __init__(self,**kwargs):
    	super(My_activation,self).__init__(**kwargs)
        
    def call(self,x,mask=None):
    	fx_0 = K.relu(x)  #for x>0
    	fx_1 = np.power(1+ np.power(math.e,-x),-1)*K.cast(x<=0.0,K.floatx()) #for x<0
    	return fx_0+fx_1
    def get_output_shape_for(self, input_shape):
        #we don't change the input shape
        return input_shape

#-----------------------------Default Paramters----------------------------------------------#
FrameSize = 512
Overlap = 256
FFTSize = 512
Rate = 25000
FrameWidth = 2
FrequencyBin = FrameSize//2+1 #stft_matrix:np.ndarray [shape=(1 + n_fft/2, t)]
Counter = 0

AudioFilename='s10' #using s10 as testing data set
NoisyAudiopath= os.getcwd()+'\\Noisy_list\\'+AudioFilename+'\\'
NoisyAudioList=['list_noisy_buses.txt','list_noisy_outdoor.txt','list_noisy_street.txt']
Enhancementpath=os.getcwd()+'\\Ehancement\\'+AudioFilename+'\\'+"enhanced_"+NoisyAudioList[0].split('_')[2][:-4]
EnhancementFileListPath=os.getcwd()+'\\Enhance_list\\'+AudioFilename+'\\'+"enhanced_"+NoisyAudioList[0].split('_')[2][:-4]+'\\'

#------------------------Create the path for enhancement audios-------------------------------#
def createpath(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
createpath(Enhancementpath)
createpath(EnhancementFileListPath)
EnhancementFileNameList=os.listdir(Enhancementpath)

#-----------------------------------Load the model--------------------------------------------#

model=load_model('model.hdf5',custom_objects={'My_activation': My_activation})

with open(NoisyAudiopath+NoisyAudioList[0], 'r') as f:
    
    for line in f:
        filename = line.split('/')[3].strip('\n')
        y,sr=librosa.load(line[:-1],sr=Rate)
        #print(len(y))
        Counter=Counter+1
        #Setting a default training data size as the same as the beginning
        training_data = np.empty((300000, FrequencyBin, FrameWidth*2+1)) 
        
        #librosa.stft ( Short-time Fourier transform)
        D=librosa.stft(y,n_fft=FrameSize,hop_length=Overlap,win_length=FFTSize,window=scipy.signal.hamming)
        
        #np.abs(D[f, t]) is the magnitude of frequency bin f at frame t = an amplitude spectrogram
        #np.angle(D[f, t]) is the phase of frequency bin f at frame t
        SignaldB = np.log10(abs(D)**2) 
        phase = np.exp(1j * np.angle(D))# for reconstruct the clean aurdio 
        #j means Complex number unit
        
#--- normalise the mangnitude of frequency on the enhancement part (5 ('2-6', '3-7','4-8'...) frames)---#
        mean = np.mean(SignaldB, axis=1).reshape((257,1))
        std = np.std(SignaldB, axis=1).reshape((257,1))
        SignaldB = (SignaldB-mean)/std
              
        enhanced_id = 0     
        for i in range(FrameWidth, SignaldB.shape[1]-FrameWidth): 
            training_data[enhanced_id,:,:] = SignaldB[:,i-FrameWidth:i+FrameWidth+1] 
            enhanced_id = enhanced_id + 1
        Noisy_train = training_data[:enhanced_id]
        Noisy_train = np.reshape(Noisy_train,(enhanced_id,-1))
        
#------------Get the predict value from the DNN model------------------------------------------#        
        predict = model.predict(Noisy_train)
        count=0
        for i in range(FrameWidth, SignaldB.shape[1]-FrameWidth):
            SignaldB[:,i] = predict[count]
            count+=1
            
#---------------The un-enhanced part of spec should be un-normalized ---------------------------#
        SignaldB[:, :FrameWidth] = (SignaldB[:, :FrameWidth] * std) + mean
        SignaldB[:, -FrameWidth:] = (SignaldB[:, -FrameWidth:] * std) + mean    

#-------------Inverse short-time Fourier transform (ISTFT)--------------------------------------#
        # Restore data becuase pre-process step (SignaldB = np.log10(abs(D)**2)) 
        SignaldB = np.sqrt(10**SignaldB)
        StftMatrix = np.multiply(SignaldB , phase)
        # reconstruct the enhancement data 
        result = librosa.istft(StftMatrix, hop_length=Overlap,win_length=FFTSize,window=scipy.signal.hamming)
        # Fix the length an array data to exactly size.
        output = librosa.util.fix_length(result, y.shape[0])
        #print(len(output))
#----------------------Save the enhancede audio file--------------------------------------------#
        wav.write(os.path.join(Enhancementpath,'enhanced_'+filename),Rate,np.int16(output*32767))
        print('Enhancement completed!','Number:'+'%05d'% Counter,'enhanced_'+filename)
