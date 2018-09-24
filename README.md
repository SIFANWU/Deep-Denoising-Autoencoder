# Deep-Denoising-Autoencoder
DAE for noise reduction and speech enhancement

Using Keras to construct the model (backend is Tensorflow)

The evaluation methods include PESQ (Perceptual Evaluation of Speech Quality) and STOI (Short Term Objective Intelligibility)


#----------------------------------------------------------------------------------------------------------------------------#
Execution order：pre-process.py -getspectrum.py -trainmodel.py -testmodel.py -Evaluation.py -PESQ_socre.py


#----------------------------------------------------------------------------------------------------------------------------#
The original source data is from The GRID audiovisual sentence corpus(University of Sheffield) for clean speech
The link: http://spandh.dcs.shef.ac.uk/gridcorpus/

The noise is from the freesound: https://freesound.org/ 

#----------------------------------------------------------------------------------------------------------------------------#
Issue:
PESQ is not open source code and Writting with C language (In this case, PESQ has been downloaded and save in file called P862)

When PESQ finish compiled using $ PESQ +16000 Reference.wav Enhancement.wav to get the score (Max is 4.5)

The score of STOI is from 0 to 1.
