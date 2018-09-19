from keras import backend as K # for Custom objective function
from keras.engine import Layer
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Flatten, Activation, SpatialDropout2D, Reshape
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU, PReLU, LeakyReLU
from keras.utils.io_utils import HDF5Matrix
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import math
import numpy as np

FrameSize = 512
FrameWidth = 2
FrequencyBin = FrameSize//2+1
Input_dim = FrequencyBin*(FrameWidth*2+1)

BatchSize = 200 # the number of samples used for each gradient iteration update
Epochs = 30 # Total number of iterations

#----------Build My activation function----------------------------#

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

#----------------Build a model-------------------------------------#

#-----advanced_activations.ELU(Exponential Linear Unit)------------#
# f(x) = alpha * (exp(x) - 1.)  for x < 0, f(x) = x   for x>=0

model = Sequential()
    
# encoder layer 1
model.add(Dense(2048, input_dim = Input_dim))
#model.add(My_activation())
model.add(ELU())
model.add(Dropout(0.05))#avoid the over-fitting
    
# encoder layer 2
model.add(Dense(2048))
#model.add(My_activation())
model.add(ELU())    
model.add(Dropout(0.05))
  
# encoder layer 3
model.add(Dense(2048))
#model.add(My_activation())
model.add(ELU())
model.add(Dropout(0.05))

# encoder layer 4
model.add(Dense(2048))
#model.add(My_activation())
model.add(ELU())
model.add(Dropout(0.05))

# encoder layer 5
model.add(Dense(2048))
#model.add(My_activation())
model.add(ELU())
model.add(Dropout(0.05))

model.add(Dense(257))

#print model information
model.summary()

#--------------Compile model-------------------------------------#

# Adam maybe is the best choice 
# Lr:Lrean rate  
# Epsilon: small floating point number with or equal to 0, preventing the divide by 0 error
adam= optimizers.Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=1e-08,decay=0.00)
model.compile(loss='mse', optimizer=adam, metrics=["accuracy"])
#loss='mae' mean_absolute_error
#loss-'mse' mean_squared_error 

#--------------Loading data--------------------------------------#  

noisy_train = HDF5Matrix('data.h5',"trainnoise")
clean_train = HDF5Matrix('data.h5',"trainclean") 
print('Noisy data shape: ', noisy_train.shape)
print('Clean data shape: ', clean_train.shape)

#-------------Save the model as model.hdf5-----------------------#

# Using callbacks function 
# Parameter:
#         Filename: string, the path to save the model
#         Monitor: the value to be monitored
#         Verbose: information display mode, 0 or 1  
#         Save_best_only: When set to True, will only save the best performing model on the validation set

Savemodel =ModelCheckpoint( filepath="model.hdf5", monitor="loss",verbose=0, save_best_only=True)

#----------------training model----------------------------------#  

# verbose: 0 means no log output, 1 means output log with progress bar, 2 means print one line of log for each epoch
# Shuffle: boolean, will each epoch mess up the data
hist=model.fit(noisy_train, clean_train, epochs=Epochs, batch_size=BatchSize, verbose=1,shuffle="batch", callbacks=[Savemodel])
