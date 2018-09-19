import numpy as np
import os

Path= os.getcwd().replace('\\','/')+'/P862/Software/source/'
scorelist=[]
#--------Calculate the average score-----------------------------#
with open(Path+'_pesq_results.txt','r') as file:
    for line in file.readlines()[1:]:
        line=line.strip()
        if len(line)!=0:
            score=line.split('\t')[1]
            scorelist.append(float(score))
abc=np.array(scorelist)
print('The average score of PESQ is %.3f' %np.mean(abc))
    