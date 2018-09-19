from scipy.io.wavfile import read
from pystoi.stoi import stoi
import os
import numpy as np
import librosa

AudioFilename='s10'
PathName= os.getcwd().replace('\\','/')
NoisyAudioList=['list_noisy_buses.txt','list_noisy_outdoor.txt','list_noisy_street.txt']

CleanAudiopath= os.getcwd()+'\\Clean_list\\'+AudioFilename+'\\'
EnhancmentAudiopath =os.getcwd()+'\\Enhance_list\\'+AudioFilename+'\\enhanced_buses\\'
Enhancementpath=os.getcwd()+'\\Ehancement\\'+AudioFilename+'\\'+"enhanced_"+NoisyAudioList[0].split('_')[2][:-4]
EnhancementFileListPath=os.getcwd()+'\\Enhance_list\\'+AudioFilename+'\\'+"enhanced_"+NoisyAudioList[0].split('_')[2][:-4]+'\\'
PESQFilePath =os.getcwd()+'\\P862\\Software\\source\\'
EnhancementFileNameList=os.listdir(Enhancementpath)
scorelist=[]


#----------------Save the Enhancement filename List--------------------------------------#
Ehfile=open(EnhancementFileListPath+'Enhance_list'+'.txt','w')
for EnhancementFileName in EnhancementFileNameList :
    Ehfile.write('Ehancement/'+AudioFilename+'/'+'enhanced_'+NoisyAudioList[0].split('_')[2][:-4]+'/'+EnhancementFileName+'\n')
Ehfile.close()
print('The Enhancement file list has been done!')

#-------------(Short-Time Objective Intelligibility)STOI----------------#
with open(CleanAudiopath+'list_clean.txt', 'r') as f1, open(EnhancmentAudiopath+'Enhance_list.txt', 'r') as f2:
    for lineC,lineE in zip(f1,f2):
        
        fs, clean = read(lineC[:-1])
        fs, den = read(lineE[:-1])  
        clean = librosa.util.fix_length(clean,den.shape[0])
        score = stoi(clean, den, fs)
        scorelist.append(score)
        #print('the score is %.3f'%score)
abc=np.array(scorelist)
#get the score of STOI
print('The average score is %.3f' %np.mean(abc))


#------------------------------PESQ-----------------------------------#
'''
produce a .bat file called MyPESQTest.bat need to run 
and sample rate sets as 16000 Hz
'''
PESQFile= open(PESQFilePath+'MyPESQTest.bat','w')
with open(CleanAudiopath+'list_clean.txt', 'r') as f3,open(EnhancmentAudiopath+'Enhance_list.txt', 'r') as f4:
    for clean,enhance in zip(f3,f4):
        PESQFile.write('pesq +16000 '+PathName+'/'+clean.strip('\n')+'\000\000'+PathName+'/'+enhance)
PESQFile.close()    
