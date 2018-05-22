import pickle
import statsmodels.api as sm
import os
import re
import tarfile
import math
import random
import sys
import time
import logging
import numpy as np
import math

from tensorflow.python.platform import gfile
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_1d, max_pool_1d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
#from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell,GRUCell
from recurrent import bidirectional_rnn, BasicLSTMCell,GRUCell


def  train_dev_split(train_protein,train_compound,train_IC50,dev_perc,comp_MAX_size,protein_MAX_size,batch_size):
    num_whole= len(train_IC50)
    num_train = math.ceil(num_whole*(1-dev_perc)/batch_size)*batch_size
    num_dev = math.floor((num_whole - num_train)/batch_size)*batch_size

    index_total = range(0,num_whole)
    index_dev = sorted(random.sample(index_total,num_dev))
    remain = list(set(index_total)^set(index_dev))
    index_train = sorted(random.sample(remain,num_train))

    compound_train = [train_compound[i] for i in index_train]
    compound_train = np.reshape(compound_train,[len(compound_train),comp_MAX_size])
    compound_dev = [train_compound[i] for i in index_dev]
    compound_dev = np.reshape(compound_dev,[len(compound_dev),comp_MAX_size])

    IC50_train = [train_IC50[i] for i in index_train]
    IC50_train = np.reshape(IC50_train,[len(IC50_train),1])
    IC50_dev = [train_IC50[i] for i in index_dev]
    IC50_dev = np.reshape(IC50_dev,[len(IC50_dev),1])

    protein_train = [train_protein[i] for i in index_train]
    protein_train = np.reshape(protein_train,[len(protein_train),protein_MAX_size])
    protein_dev = [train_protein[i] for i in index_dev]
    protein_dev = np.reshape(protein_dev,[len(protein_dev),protein_MAX_size])

    return compound_train, compound_dev, IC50_train, IC50_dev, protein_train, protein_dev


dev_perc = 0.1
batch_size = 64
random.seed(1234)
def normalize_labels(label):
   x = []
   minprecision = 0.001
   maxprecision = 30
   for i in label:
      if i < minprecision:
         i = minprecision
      elif i > maxprecision:
         i = maxprecision
      m = -math.log(i/maxprecision)
      x.append(m)

   return x

################# reading labels
data_path = "./data/"
label_path_train=data_path+"train_IC50"
label_train = []
f = open(label_path_train, "r")
length_train=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality in IC50!!!\n")
    else:
            label_train.append(float(line))
            length_train = length_train+1

f.close()
label_train = normalize_labels(label_train)


label_path_test=data_path+"test_IC50"
label_test = []
f = open(label_path_test, "r")
length_test=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality in IC50!!!\n")
    else:
            label_test.append(float(line))
            length_test = length_test+1

f.close()
label_test = normalize_labels(label_test)

label_path_ER=data_path+"ER_IC50"
label_ER = []
f = open(label_path_ER, "r")
length_ER=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality in IC50!!!\n")
    else:
            label_ER.append(float(line))
            length_ER = length_ER+1

f.close()
label_ER = normalize_labels(label_ER)


label_path_GPCR=data_path+"GPCR_IC50"
label_GPCR = []
f = open(label_path_GPCR, "r")
length_GPCR=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality in IC50!!!\n")
    else:
            label_GPCR.append(float(line))
            length_GPCR = length_GPCR+1

f.close()
label_GPCR = normalize_labels(label_GPCR)


label_path_kinase=data_path+"channel_IC50"
label_kinase = []
f = open(label_path_kinase, "r")
length_kinase=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality in IC50!!!\n")
    else:
            label_kinase.append(float(line))
            length_kinase = length_kinase+1

f.close()
label_kinase = normalize_labels(label_kinase)

##################  reading compound features
feature_ER_compound = np.zeros((length_ER,256))
feature_ER_protein = np.zeros((length_ER,512))
textfile1 = open(data_path+"ER_smile_feature")
textfile2 = open(data_path+"ER_fasta_feature")
count=0
while length_ER > count:
    x = textfile1.readline()
    y = textfile2.readline()
    x = x.strip()
    y = y.strip()
    result1 = np.array([list(map(float, x.split()))])
    result2 = np.array([list(map(float, y.split()))])
    feature_ER_compound[count,]=result1
    feature_ER_protein[count,]=result2
    count = count+1


feature_GPCR_compound = np.zeros((length_GPCR,256))
feature_GPCR_protein = np.zeros((length_GPCR,512))
textfile1 = open(data_path+"GPCR_smile_feature")
textfile2 = open(data_path+"GPCR_fasta_feature")
count=0
while length_GPCR > count:
    x = textfile1.readline()
    y = textfile2.readline()
    x = x.strip()
    y = y.strip()
    result1 = np.array([list(map(float, x.split()))])
    result2 = np.array([list(map(float, y.split()))])
    feature_GPCR_compound[count,]=result1
    feature_GPCR_protein[count,]=result2
    count = count+1

feature_kinase_compound = np.zeros((length_kinase,256))
feature_kinase_protein = np.zeros((length_kinase,512))
textfile1 = open(data_path+"channel_smile_feature")
textfile2 = open(data_path+"channel_fasta_feature")
count=0
while length_kinase > count:
    x = textfile1.readline()
    y = textfile2.readline()
    x = x.strip()
    y = y.strip()
    result1 = np.array([list(map(float, x.split()))])
    result2 = np.array([list(map(float, y.split()))])
    feature_kinase_compound[count,]=result1
    feature_kinase_protein[count,]=result2
    count = count+1

feature_train_compound = np.zeros((length_train,256))
feature_train_protein = np.zeros((length_train,512))
textfile1 = open(data_path+"train_smile_feature")
textfile2 = open(data_path+"train_fasta_feature")
count=0
while length_train > count:
    x = textfile1.readline()
    y = textfile2.readline()
    x = x.strip()
    y = y.strip()
    result1 = np.array([list(map(float, x.split()))])
    result2 = np.array([list(map(float, y.split()))])
    feature_train_compound[count,]=result1
    feature_train_protein[count,]=result2
    count = count+1

feature_train_compound,mean_feature_train_compound = tflearn.data_utils.featurewise_zero_center(feature_train_compound)
feature_train_protein,mean_feature_train_protein = tflearn.data_utils.featurewise_zero_center(feature_train_protein)

feature_train_compound,std_feature_train_compound = tflearn.data_utils.featurewise_std_normalization(feature_train_compound)
feature_train_protein,std_feature_train_protein = tflearn.data_utils.featurewise_std_normalization(feature_train_protein)

compound_train, compound_dev, IC50_train, IC50_dev, protein_train, protein_dev  =train_dev_split(feature_train_protein,feature_train_compound,label_train,dev_perc,256,512,batch_size)

feature_test_compound = np.zeros((length_test,256))
feature_test_protein = np.zeros((length_test,512))
textfile1 = open(data_path+"test_smile_feature")
textfile2 = open(data_path+"test_fasta_feature")
count=0
while length_test > count:
    x = textfile1.readline()
    y = textfile2.readline()
    x = x.strip()
    y = y.strip()
    result1 = np.array([list(map(float, x.split()))])
    result2 = np.array([list(map(float, y.split()))])
    feature_test_compound[count,]=result1
    feature_test_protein[count,]=result2
    count = count+1

feature_test_compound = tflearn.data_utils.featurewise_zero_center(feature_test_compound,mean_feature_train_compound)
feature_test_protein = tflearn.data_utils.featurewise_zero_center(feature_test_protein,mean_feature_train_protein)

feature_test_compound = tflearn.data_utils.featurewise_std_normalization(feature_test_compound,std_feature_train_compound)
feature_test_protein = tflearn.data_utils.featurewise_std_normalization(feature_test_protein,std_feature_train_protein)

feature_ER_compound = tflearn.data_utils.featurewise_zero_center(feature_ER_compound,mean_feature_train_compound)
feature_ER_protein = tflearn.data_utils.featurewise_zero_center(feature_ER_protein,mean_feature_train_protein)

feature_ER_compound = tflearn.data_utils.featurewise_std_normalization(feature_ER_compound,std_feature_train_compound)
feature_ER_protein = tflearn.data_utils.featurewise_std_normalization(feature_ER_protein,std_feature_train_protein)


feature_GPCR_compound = tflearn.data_utils.featurewise_zero_center(feature_GPCR_compound,mean_feature_train_compound)
feature_GPCR_protein = tflearn.data_utils.featurewise_zero_center(feature_GPCR_protein,mean_feature_train_protein)

feature_GPCR_compound = tflearn.data_utils.featurewise_std_normalization(feature_GPCR_compound,std_feature_train_compound)
feature_GPCR_protein = tflearn.data_utils.featurewise_std_normalization(feature_GPCR_protein,std_feature_train_protein)


feature_kinase_compound = tflearn.data_utils.featurewise_zero_center(feature_kinase_compound,mean_feature_train_compound)
feature_kinase_protein = tflearn.data_utils.featurewise_zero_center(feature_kinase_protein,mean_feature_train_protein)

feature_kinase_compound = tflearn.data_utils.featurewise_std_normalization(feature_kinase_compound,std_feature_train_compound)
feature_kinase_protein = tflearn.data_utils.featurewise_std_normalization(feature_kinase_protein,std_feature_train_protein)

# Sep model
prot_data = input_data(shape=[None, 512])
prot_reshape = tflearn.reshape(prot_data, [-1, 256,2])
conv_1 = conv_1d(prot_reshape, 64, 4,2, activation='leakyrelu',weights_init="xavier", regularizer="L2",name='conv1')
pool_1 = max_pool_1d(conv_1, 4,name='pool1')
conv_2 = conv_1d(pool_1, 32, 4,2, activation='leakyrelu', weights_init="xavier",regularizer="L2",name='conv2')
pool_2 = max_pool_1d(conv_2, 2,name='pool2')
prot_reshape_4 = tflearn.reshape(pool_2, [-1, 32*8])


drug_data = input_data(shape=[None,256])
drug_reshape = tflearn.reshape(drug_data, [-1, 128,2])
conv_3 = conv_1d(drug_reshape, 64, 4,2, activation='leakyrelu', weights_init="xavier",regularizer="L2",name='conv3')
pool_3 = max_pool_1d(conv_3, 2,name='pool3')
conv_4 = conv_1d(pool_3, 32, 4,2, activation='leakyrelu', weights_init="xavier",regularizer="L2",name='conv4')
pool_4 = max_pool_1d(conv_4, 2,name='pool4')
drug_reshape_4 = tflearn.reshape(pool_4, [-1, 32*8])

merging =  merge([prot_reshape_4,drug_reshape_4],mode='concat',axis=1)
fc_2 = fully_connected(merging, 200, activation='leakyrelu',weights_init="xavier",name='fully2')
drop_3 = dropout(fc_2, 0.8)
fc_3 = fully_connected(drop_3, 50, activation='leakyrelu',weights_init="xavier",name='fully2')
drop_4 = dropout(fc_3, 0.8)
linear = fully_connected(drop_4, 1, activation='linear',name='fully3')
reg = regression(linear, optimizer='adam', learning_rate=0.001,
                     loss='mean_square', name='target')

# Training
model = tflearn.DNN(reg, tensorboard_verbose=0,tensorboard_dir='./mytensor/',checkpoint_path="./checkpoints/")

######## training
model.fit([protein_train,compound_train], {'target': IC50_train}, n_epoch=200,batch_size=batch_size,
           validation_set=([protein_dev,compound_dev], {'target': IC50_dev}),
            show_metric=True, run_id='joint_model')

# saving model
model.save('my_model.tflearn')

# evaluation on ER
print("error on ER")
y_pred= model.predict([feature_ER_protein,feature_ER_compound])
mse = ((y_pred - label_ER) ** 2).mean(axis=None)
print(mse)
print(tf.size(y_pred))
print(len(label_ER))
results = sm.OLS(y_pred,sm.add_constant(label_ER)).fit()
print(results.summary())

# evaluation on GPCR
print("error on GPCR")
y_pred= model.predict([feature_GPCR_protein,feature_GPCR_compound])
mse = ((y_pred - label_GPCR) ** 2).mean(axis=None)
print(mse)

results = sm.OLS(y_pred,sm.add_constant(label_GPCR)).fit()
print(results.summary())

# evaluation on kinase
print("error on kinase")
y_pred= model.predict([feature_kinase_protein,feature_kinase_compound])
mse = ((y_pred - label_kinase) ** 2).mean(axis=None)
print(mse)

results = sm.OLS(y_pred,sm.add_constant(label_kinase)).fit()
print(results.summary())

# evaluation on train
print("error on train")
size = 5000
num_bins = math.ceil(length_train/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([feature_train_protein[0:size,],feature_train_compound[0:size,]])
        elif i < num_bins-1:
          temp = model.predict([feature_train_protein[(i*size):((i+1)*size),],feature_train_compound[(i*size):((i+1)*size),]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([feature_train_protein[(i*size):length_train,],feature_train_compound[(i*size):length_train,]])
          y_pred = np.concatenate((y_pred,temp), axis=0)

er=0
for i in range(length_train):
  er += (y_pred[i]-label_train[i])**2

mse = er/length_train
print(mse)

results = sm.OLS(y_pred,sm.add_constant(label_train)).fit()
print(results.summary())

# evaluation on test
print("error on test")
size = 5000
num_bins = math.ceil(length_test/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([feature_test_protein[0:size,],feature_test_compound[0:size,]])
        elif i < num_bins-1:
          temp = model.predict([feature_test_protein[(i*size):((i+1)*size),],feature_test_compound[(i*size):((i+1)*size),]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([feature_test_protein[(i*size):length_test,],feature_test_compound[(i*size):length_test,]])
          y_pred = np.concatenate((y_pred,temp), axis=0)

er=0
for i in range(length_test):
  er += (y_pred[i]-label_test[i])**2

mse = er/length_test
print(mse)

results = sm.OLS(y_pred,sm.add_constant(label_test)).fit()
print(results.summary())

