"""
file: sentiment_module.py
author Petri Lamminaho
Simple text sentiment classifier trained and ready to use
Neuron network based model
Uses tfLearn library
Dataset is tfLear's imdb dataset
"""
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb
# 1.
#---------------------
#loading imdb-dataset
#------------------------

train, test, _ = imdb.load_data(path='imdb.pkl', n_words=10000, valid_portion=0.1)
trainX, trainY = train
testX, testY = test

# 2.
#-----------------------
# pre processing data
#----------------------
# padding
#------------

trainX = pad_sequences(trainX, maxlen=100, value=0.)
testX = pad_sequences(testX, maxlen=100, value=0.)
#-------------------
# labels to vectors
#-------------------
trainY = to_categorical(trainY, nb_classes=2)
testY  = to_categorical(testY, nb_classes=2) # two categories positive or negative

# 3.
#--------------------
# create neural net
#---------------------
net = tflearn.input_data([None, 100]) # maxlen is 100
net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                        loss='categorical_crossentropy')

# 4.
#-----------------------
# train model
#------------------------
n_epoch = 100
model = tflearn.DNN(net, tensorboard_verbose=0)

#-------------------------------------------------------------------------
#first time you use this for training and save the model when your done
# You can decrease n_epoch value to speed up the training now it is 100
#------------------------------------------------------------------------
#model.fit(trainX, trainY,     #training
#          validation_set=(testX, testY),
#          n_epoch=n_epoch,
#          show_metric=True,
#         batch_size=32,  snapshot_epoch=True, # Snapshot (save & evaluate) model every epoch.
#          snapshot_step=500, # Snapshot (save & evalaute) model every 500 steps.
#          run_id='model_and_weights')
#model.save('model.tfl')  # Save the model
#print('Model Saved!')

model.load('model.tfl') # Load saved model
print('Model Loaded!')

print('Classifier\'s Accuracy:',model.evaluate(testX, testY))

# 5.
############################################
# resume the training for 1 epoch
#############################################
#model.fit(trainX, trainY,
#          validation_set=(testX, testY),
#          n_epoch=1,
#          show_metric=True,
#         batch_size=32,  snapshot_epoch=True, # Snapshot (save & evaluate) model every epoch.
#          snapshot_step=500, # Snapshot (save & evalaute) model every 500 steps.
#          run_id='model_and_weights')

#save when training is done

#model.save('model.tfl')
#print('Model Saved!')
