# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:37:21 2022

@author: tanne
"""

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Import Libraries Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn import preprocessing
import datetime
from sklearn.model_selection import train_test_split
import itertools
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Parameters Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
num_nodes_first = 200 #nodes in the first layer
num_nodes_second = 20 #nodes in the second layer
num_classes = 1 #number of outputs 
num_epochs = 20 #number of epochs
size_batch = 32 #batch size
verbose_setting=1
inputs = 21 #number of predictive variables
learn_rate = 0.01 #learning rate
decay = 0.1 #decay rate

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Load Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Batting = pd.read_csv("Batting.csv")
Salary = pd.read_csv("salary.csv")
Player = pd.read_csv("player.csv")

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Pretreat Data Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

Batting= Batting[Batting['yearID'] > 1984]
#Only up to 2012

Salary = Salary[Salary['year'] < 2013] #filtering to match batting dataset

BattingPlayer = pd.merge(Batting, Player,  how='inner', left_on=['playerID'], right_on = ['player_id']) #Merging the batter and player datasets

BattingPlayer.loc[BattingPlayer['birth_month'] >6, 'Effective_birth_year'] = BattingPlayer['birth_year'] -1 #Decreasing effective birth year by 1 year if born after June

BattingPlayer.loc[BattingPlayer['birth_month'] <7, 'Effective_birth_year'] = BattingPlayer['birth_year'] #Creating effective birth year for players born before June.

BattingPlayer['Age']=BattingPlayer['yearID'] - BattingPlayer['Effective_birth_year'] #Creating age by subtracting the year from the player's effective birth year 

data = pd.merge(BattingPlayer, Salary,  how='inner', left_on=['playerID','yearID'], right_on = ['player_id','year']) #creating full dataset by merging Salary data in.

data = data.drop(columns =['playerID', 'stint', 'teamID', 'lgID', 'G_batting', 'G_old', 'player_id_x', 
                           'team_id', 'league_id', 'player_id_y', 'birth_year', 'birth_month', 'Effective_birth_year', 'year'] ) #Dropping unnecessary columns
data = data[data.AB >100] #filtering the data so only players with over 100 ABs are included.

X = data.iloc[:,0:21] #Creating X
y = data.iloc[:, 21:22] #Creating Y

encoder = LabelEncoder()
encoder.fit(X['throws'])
encoded_cat1 = encoder.transform(X['throws'])
# convert integers to dummy variables (i.e. one hot encoded)
X['throws'] = to_categorical(encoded_cat1) #Encoding the throws column to be a dummy variable

encoder.fit(X['bats'])
encoded_cat2 = encoder.transform(X['bats'])
# convert integers to dummy variables (i.e. one hot encoded)
X['bats'] = to_categorical(encoded_cat2) #Encoding the throws column to be a dummy variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) #Creating training and test data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=1) #Doing the same to create validation data

##### Preprocessing data to make values range between 0-1 to increase effectiveness of ANN
X_MinMax = preprocessing.MinMaxScaler() 
Y_MinMax = preprocessing.MinMaxScaler()

X_train = X_MinMax.fit_transform(X_train)
X_test = X_MinMax.fit_transform(X_test)
X_val = X_MinMax.fit_transform(X_val)
y_train = Y_MinMax.fit_transform(y_train)
y_test = Y_MinMax.fit_transform(y_test)
y_val = Y_MinMax.fit_transform(y_val)
X_train.mean(axis=0)
X_test.mean(axis =0)
X_val.mean(axis =0)


"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Define Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model = Sequential()
model.add(Dense(num_nodes_first, input_dim = inputs, activation = 'linear')) #adds the input nodes, which were experimentally tweaked.
model.add(BatchNormalization()) #Uses batch normalization
model.add(Dense(num_nodes_second, input_dim = inputs, activation = 'linear'))#adds another layer of the input nodes, which were experimentally tweaked.
model.add(layers.Dense(num_classes, activation='linear')) #Adds last dense layer
optimizer = SGD(lr = learn_rate, decay = decay) #Creates optimizer with learning rate and decay rates changed from defaults

#compile the model
model.compile(loss = 'mean_squared_error', optimizer = optimizer, metrics=['mean_squared_error', 'mean_absolute_error']) #Compiles model


start_time = datetime.datetime.now() #starts the timer for tracking the run time of the model.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Train Model Section
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#fix random seed for repeatability

seed = 7
np.random.seed(seed) #allows for consistent results

history = model.fit(X_train, y_train, epochs=num_epochs, verbose=verbose_setting, batch_size = size_batch, validation_data=(X_test, y_test)) #Tracks performance of model.

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Show output Section

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


loss_train = history.history['loss']
loss_test = history.history['val_loss']
epochs = range(1,num_epochs+1) #Creates range of graph, utilizing the fact that we set the number of epochs as an object so that it works if we change the # of epochs.
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_test, 'b', label='Test loss')
plt.title('Training and Test loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
stop_time = datetime.datetime.now() #stops the model run time
print ("Time required for training:",stop_time - start_time)

