import os
from pydub import AudioSegment
import numpy as np

Environmentpath = os.getcwd()+'\\data\\Environment\\'
CleanSetName='s1_audio'
Cleanpath = os.getcwd()+'\\data\\'+CleanSetName+'\\'
NoisyfileListpath=os.getcwd()+'\\Noisy_list\\'+CleanSetName.split('_')[0]+'\\'
CleanfileListpath=os.getcwd()+'\\Clean_list\\'+CleanSetName.split('_')[0]+'\\'

SNR = 5 #Set the SNR value 

#-----------------Calculate the value of SNR------------------------#
def snr_db(rms_amplitude_A, rms_amplitude_B):
    return 20.0*np.log10(rms_amplitude_A/rms_amplitude_B)

def createpath(path):
    if not os.path.exists(path):
        os.makedirs(path)
#----------------------Make the directory---------------------------#
createpath(NoisyfileListpath)
createpath(CleanfileListpath)
createpath(Cleanpath)

#--------------------Get Clean and Environment audio list-----------#
CleanAudioNameList=os.listdir(Cleanpath)
EnvironmentNameList=os.listdir(Environmentpath)

#------------------Add Environment audio with each clean audio------#
for j in EnvironmentNameList:
    EnvironmentSound = AudioSegment.from_file(Environmentpath+j,format='wav',frame_rate=25000)
    EnvironmentSound = EnvironmentSound.set_channels(1)
    Noisypath = os.getcwd() +'\\Noise\\'+CleanSetName.split('_')[0]+'\\'+j[:-4]+'\\'
    createpath(Noisypath)

    for i in CleanAudioNameList:
        CleanSound = AudioSegment.from_file(Cleanpath+i,format='wav',frame_rate=25000)
        CleanSound = CleanSound.set_channels(1)
        framewidth = CleanSound.frame_width
        condition =  EnvironmentSound.dBFS-CleanSound.dBFS
        
#-------------------SNR(Signal-to-noise ratio) =0 (dB)-----------------------#
        if SNR == 0 :
            EnvironmentSound = EnvironmentSound.apply_gain(-condition)
#----------------------- SNR = -5 (dB)---------------------------------------#        
        if SNR < 0 :
            EnvironmentSound = EnvironmentSound.apply_gain(-condition)
            EnvironmentSound = EnvironmentSound.apply_gain(-SNR)
#----------------------- SNR = 5(dB)-----------------------------------------#
        if SNR > 0:
            EnvironmentSound = EnvironmentSound.apply_gain(-condition)  
            EnvironmentSound = EnvironmentSound.apply_gain(-SNR)
            
        NoisyAudio = CleanSound.overlay(EnvironmentSound, loop=True)
        NoisyAudiohandle = NoisyAudio.export(Noisypath+'Noisy_'+i, format="wav") 
        
#--------------Make a  list of noisy audio--------------------------#
 
    NoisyAudioNameList=os.listdir(Noisypath)
    Noisyfile=open(NoisyfileListpath+'list_noisy_'+j[:-4]+'.txt','w')
    for i in NoisyAudioNameList:
        Noisyfile.write('Noise/'+CleanSetName.split('_')[0]+'/'+j[:-4]+'/'+i+'\n')
    Noisyfile.close()
    print('Environment_'+j[:-4]+' Noisy audio file list has done!')
print('Noise production succeeded !')        

#---------------Make a list of clean audio--------------------------#

Cleanfile=open(CleanfileListpath+'list_clean'+'.txt','w')
for i in CleanAudioNameList:
    Cleanfile.write('data/'+CleanSetName+'/'+i+'\n')
Cleanfile.close()
print('Clean audio file list has done!')