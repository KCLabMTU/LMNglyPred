import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow.keras
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l1,l2
from tensorflow.keras import backend as K
from tensorflow.keras.backend import expand_dims
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from numpy import sort
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import *
from sklearn.metrics import roc_curve, roc_auc_score, classification_report,auc
import tensorflow.keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from tensorflow.keras.layers import Dense, Bidirectional,Dense, LSTM, Activation, Dropout, Flatten, LeakyReLU
from sklearn.metrics import accuracy_score
from tensorflow.keras.regularizers import l2
from tensorflow.keras import regularizers
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import LSTM

from tensorflow.keras.models import Model
from tensorflow.keras.applications import mobilenet_v2, mobilenet, resnet50, densenet
from tensorflow.keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, \
    BatchNormalization, Activation, GlobalAveragePooling2D, DepthwiseConv2D, \
    Dropout, ReLU, Concatenate, Input, add, Conv1D, MaxPooling1D

from tensorflow.keras.layers import LSTM, GRU, SimpleRNN
import random

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

import matplotlib.pyplot as plt
import math
from tensorflow.keras.callbacks import CSVLogger
from datetime import datetime

import os.path
from scipy.spatial import distance
import scipy.io as sio

import os 

basedir = "/home/t326h379/Prot_T5/Prot_T5_csv"
os.chdir(basedir)

a = random.sample(range(1, 10000), 200)

for i in range(len(a)):
    seed = a[i]
    np.random.seed(seed)

    Header_name = ["Label","PID","POsition","Sequence","Middle_Amino_Acid_ASN(N)"]

    col_of_feature = [i for i in range(1,1025)]

    Header_name = Header_name + col_of_feature

    df = pd.read_csv("August_12_Subash_123_DeepNGlyPred_data.txt", header=None)

    df.columns = Header_name

    df = df.iloc[:,5:]

    train = np.array(df)

    y_train_positive = [1]*9362
    y_train_negative = [0]*(18812-9362)
    y_train = y_train_positive+y_train_negative
    y_train = np.array(y_train)
    print(len(y_train))

    from sklearn.model_selection import train_test_split
    from sklearn.utils import shuffle

    X_train, y_train = shuffle(train, y_train, random_state = seed)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, y_train, test_size = 0.1, random_state = seed)


    print(X_train.shape,Y_train.shape,X_val.shape,Y_val.shape)

    X_train = X_train.reshape(X_train.shape[0],1024,1)
    X_val = X_val.reshape(X_val.shape[0],1024,1)

    print(X_train.shape,Y_train.shape,X_val.shape,Y_val.shape)


    df_test = pd.read_csv("Independent_Test_Set_Prot_T5_feature_Aug_12.txt", header=None)

    df_test.columns = Header_name

    df_test = df_test.iloc[:,5:]
    test = np.array(df_test)
    test.shape

    y_test_indi_positive = [1]*166
    y_test_indi_negative = [0]*(444-166)
    y_independent = y_test_indi_positive+y_test_indi_negative
    y_independent = np.array(y_independent)
    print(len(y_independent))

    X_independent = test

    print(X_independent.shape,y_independent.shape)

    X_independent = X_independent.reshape(X_independent.shape[0],1024,1)

    print(X_train.shape,Y_train.shape,X_val.shape,Y_val.shape)

    Y_train = tf.keras.utils.to_categorical(Y_train,2)
    Y_val = tf.keras.utils.to_categorical(Y_val,2)

    model = Sequential()

    model.add(Input(shape=(1024,1)))

    model.add(Conv1D(filters=64,kernel_size=3,activation='relu',name='Conv_1D_1_add'))
    model.add(MaxPooling1D(pool_size=2,name="MaxPooling1D"))
    model.add(Dropout(0.3))

    model.add(LSTM(64))
    model.add(Dropout(0.3))


    model.add(Flatten())

    model.add(Dense(1024,activation='relu',name="Dense_1"))
    model.add(Dropout(0.3))


    model.add(Dense(2,activation='softmax',name="Dense_4"))


    model.compile(optimizer=tf.keras.optimizers.Adam(),loss="binary_crossentropy",metrics=["accuracy"])

    checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath="ROC_ROC_Premise_Assumption.h5", 
                                    monitor = 'val_accuracy',
                                    verbose=0, 
                                    save_weights_only=False,
                                    save_best_only=True)

    reduce_lr_acc = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.001, patience=3, verbose=1, min_delta=1e-4, mode='max')

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5,mode='max')

    history = model.fit(X_train, Y_train,epochs=400,verbose=1,batch_size=256,
                            callbacks=[checkpointer,reduce_lr_acc, early_stopping],validation_data=(X_val, Y_val))

    pd.DataFrame(history.history).plot(figsize=(8,5))
    plt.grid(True)
    plt.gca().set_ylim(0,1)
    plt.show()
    plt.savefig('accuracy_loss@@@@@@@@@@@@@@@_curve.png', dpi=300, bbox_inches='tight')
    
    model.save("Bigger_data_set_again_extracted_LMNglyPred"+str(seed)+".h5")
    print("**********************************************************")
    print("\n")
    print("Seed:   ",seed)
    print("\n")
    print("Bigger_data_set_again_extracted_LMNglyPred"+str(seed)+".h5")
    print("\n")
    print("**********************************************************")  

    X_independent, y_independent = shuffle(X_independent, y_independent)

    Y_pred = model.predict(X_independent)
    Y_pred = (Y_pred > 0.5)
    y_pred = [np.argmax(y, axis=None, out=None) for y in Y_pred]
    y_pred = np.array(y_pred)
    print("Matthews Correlation : ",matthews_corrcoef(y_independent, y_pred))
    print("Confusion Matrix : \n",confusion_matrix(y_independent, y_pred))
    print("Accuracy on test set:   ",accuracy_score(y_independent, y_pred))

    cm = confusion_matrix(y_independent, y_pred)

    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]

    mcc = matthews_corrcoef(y_independent, y_pred)

    Sensitivity = TP/(TP+FN)

    Specificity = TN/(TN+FP)

    print("Sensitivity:   ",Sensitivity,"\t","Specificity:   ",Specificity)

    print(classification_report(y_independent, y_pred))

    fpr, tpr, _ = roc_curve(y_independent, y_pred)

    roc_auc_test = auc(fpr,tpr)



    print("Area Under Curve:   ",roc_auc_test)

    model.summary()