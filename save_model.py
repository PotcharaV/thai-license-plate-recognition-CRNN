from keras import backend as K
from Model import get_Model
from parameter import *
K.set_learning_phase(0)

# # Model description and training

model = get_Model(training=False)

try:
    model.load_weights('LSTM+BN5--thai-v3.hdf5')       # previous trained model to continue train further
    print("...Load weight data...")
    model.summary()
    model.save('Model_LSTM+BN5--thai-v3.h5', save_format='tf')

except:
    print("...Weight data not found...")
    pass