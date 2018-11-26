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


def  train_dev_split(train_protein,train_compound,train_kd,dev_perc,comp_MAX_size,protein_MAX_size,batch_size):
    num_whole= len(train_kd)
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

    kd_train = [train_kd[i] for i in index_train]
    kd_train = np.reshape(kd_train,[len(kd_train),1])
    kd_dev = [train_kd[i] for i in index_dev]
    kd_dev = np.reshape(kd_dev,[len(kd_dev),1])

    protein_train = [train_protein[i] for i in index_train]
    protein_train = np.reshape(protein_train,[len(protein_train),protein_MAX_size])
    protein_dev = [train_protein[i] for i in index_dev]
    protein_dev = np.reshape(protein_dev,[len(protein_dev),protein_MAX_size])

    return compound_train, compound_dev, kd_train, kd_dev, protein_train, protein_dev


dev_perc = 0.1
batch_size = 64
random.seed(1234)

def normalize_labels(label):
   x = []
   micro=1000000
   for i in label:
      if i ==0:
         print(i)
      m = -math.log10(i)+math.log10(micro)
      x.append(m)
   return x


################# reading labels
data_path = "./data/"
data_type="ic50"
label_path_train=data_path+"train_"+data_type
label_train = []
f = open(label_path_train, "r")
length_train=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality!!!\n")
    else:
            label_train.append(float(line))
            length_train = length_train+1

f.close()
label_train = normalize_labels(label_train)


label_path_test=data_path+"test_"+data_type
label_test = []
f = open(label_path_test, "r")
length_test=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality!!!\n")
    else:
            label_test.append(float(line))
            length_test = length_test+1

f.close()
label_test = normalize_labels(label_test)

label_path_ER=data_path+"ER_"+data_type
label_ER = []
f = open(label_path_ER, "r")
length_ER=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality!!!\n")
    else:
            label_ER.append(float(line))
            length_ER = length_ER+1

f.close()
label_ER = normalize_labels(label_ER)



label_path_kinase=data_path+"kinase_"+data_type
label_kinase = []
f = open(label_path_kinase, "r")
length_kinase=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality!!!\n")
    else:
            label_kinase.append(float(line))
            length_kinase = length_kinase+1

f.close()
label_kinase = normalize_labels(label_kinase)

label_path_GPCR=data_path+"GPCR_"+data_type
label_GPCR = []
f = open(label_path_GPCR, "r")
length_GPCR=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality!!!\n")
    else:
            label_GPCR.append(float(line))
            length_GPCR = length_GPCR+1

f.close()
label_GPCR = normalize_labels(label_GPCR)

label_path_channel=data_path+"channel_"+data_type
label_channel = []
f = open(label_path_channel, "r")
length_channel=0
for line in f:
    if (line[0]=="<")or(line[0]==">"):
            print("Inequality in kd!!!\n")
    else:
            label_channel.append(float(line))
            length_channel = length_channel+1

f.close()
label_channel = normalize_labels(label_channel)


##################  reading compound features
Comp_name="_smile_RNN"
prot_name="_sps_RNN"
feature_ER_compound = np.zeros((length_ER,256))
feature_ER_protein = np.zeros((length_ER,512))
textfile1 = open(data_path+"ER"+Comp_name)
textfile2 = open(data_path+"ER"+prot_name)
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


feature_kinase_compound = np.zeros((length_kinase,256))
feature_kinase_protein = np.zeros((length_kinase,512))
textfile1 = open(data_path+"kinase"+Comp_name)
textfile2 = open(data_path+"kinase"+prot_name)
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

feature_channel_compound = np.zeros((length_channel,256))
feature_channel_protein = np.zeros((length_channel,512))
textfile1 = open(data_path+"channel"+Comp_name)
textfile2 = open(data_path+"channel"+prot_name)
count=0
while length_channel > count:
    x = textfile1.readline()
    y = textfile2.readline()
    x = x.strip()
    y = y.strip()
    result1 = np.array([list(map(float, x.split()))])
    result2 = np.array([list(map(float, y.split()))])
    feature_channel_compound[count,]=result1
    feature_channel_protein[count,]=result2
    count = count+1

feature_GPCR_compound = np.zeros((length_GPCR,256))
feature_GPCR_protein = np.zeros((length_GPCR,512))
textfile1 = open(data_path+"GPCR"+Comp_name)
textfile2 = open(data_path+"GPCR"+prot_name)
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



feature_train_compound = np.zeros((length_train,256))
feature_train_protein = np.zeros((length_train,512))
textfile1 = open(data_path+"train"+Comp_name)
textfile2 = open(data_path+"train"+prot_name)
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

feature_test_compound = np.zeros((length_test,256))
feature_test_protein = np.zeros((length_test,512))
textfile1 = open(data_path+"test"+Comp_name)
textfile2 = open(data_path+"test"+prot_name)
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



feature_train_compound,mean_feature_train_compound = tflearn.data_utils.featurewise_zero_center(feature_train_compound)
feature_train_protein,mean_feature_train_protein = tflearn.data_utils.featurewise_zero_center(feature_train_protein)

feature_train_compound,std_feature_train_compound = tflearn.data_utils.featurewise_std_normalization(feature_train_compound)
feature_train_protein,std_feature_train_protein = tflearn.data_utils.featurewise_std_normalization(feature_train_protein)

compound_train, compound_dev, kd_train, kd_dev, protein_train, protein_dev  =train_dev_split(feature_train_protein,feature_train_compound,label_train,dev_perc,256,512,batch_size)


feature_test_compound = tflearn.data_utils.featurewise_zero_center(feature_test_compound,mean_feature_train_compound)
feature_test_protein = tflearn.data_utils.featurewise_zero_center(feature_test_protein,mean_feature_train_protein)

feature_test_compound = tflearn.data_utils.featurewise_std_normalization(feature_test_compound,std_feature_train_compound)
feature_test_protein = tflearn.data_utils.featurewise_std_normalization(feature_test_protein,std_feature_train_protein)

feature_ER_compound = tflearn.data_utils.featurewise_zero_center(feature_ER_compound,mean_feature_train_compound)
feature_ER_protein = tflearn.data_utils.featurewise_zero_center(feature_ER_protein,mean_feature_train_protein)

feature_ER_compound = tflearn.data_utils.featurewise_std_normalization(feature_ER_compound,std_feature_train_compound)
feature_ER_protein = tflearn.data_utils.featurewise_std_normalization(feature_ER_protein,std_feature_train_protein)

feature_kinase_compound = tflearn.data_utils.featurewise_zero_center(feature_kinase_compound,mean_feature_train_compound)
feature_kinase_protein = tflearn.data_utils.featurewise_zero_center(feature_kinase_protein,mean_feature_train_protein)

feature_kinase_compound = tflearn.data_utils.featurewise_std_normalization(feature_kinase_compound,std_feature_train_compound)
feature_kinase_protein = tflearn.data_utils.featurewise_std_normalization(feature_kinase_protein,std_feature_train_protein)


feature_GPCR_compound = tflearn.data_utils.featurewise_zero_center(feature_GPCR_compound,mean_feature_train_compound)
feature_GPCR_protein = tflearn.data_utils.featurewise_zero_center(feature_GPCR_protein,mean_feature_train_protein)

feature_GPCR_compound = tflearn.data_utils.featurewise_std_normalization(feature_GPCR_compound,std_feature_train_compound)
feature_GPCR_protein = tflearn.data_utils.featurewise_std_normalization(feature_GPCR_protein,std_feature_train_protein)


feature_channel_compound = tflearn.data_utils.featurewise_zero_center(feature_channel_compound,mean_feature_train_compound)
feature_channel_protein = tflearn.data_utils.featurewise_zero_center(feature_channel_protein,mean_feature_train_protein)

feature_channel_compound = tflearn.data_utils.featurewise_std_normalization(feature_channel_compound,std_feature_train_compound)
feature_channel_protein = tflearn.data_utils.featurewise_std_normalization(feature_channel_protein,std_feature_train_protein)


# Sep model
prot_data = input_data(shape=[None, 512])
prot_reshape = tflearn.reshape(prot_data, [-1, 256,2])
conv_1 = conv_1d(prot_reshape, 64, 8,4, activation='leakyrelu', weights_init="xavier",regularizer="L2",name='conv1')
pool_1 = max_pool_1d(conv_1, 4,name='pool1')
prot_reshape_4 = tflearn.reshape(pool_1, [-1, 64*16])


drug_data = input_data(shape=[None,256])
drug_reshape = tflearn.reshape(drug_data, [-1, 128,2])
conv_3 = conv_1d(drug_reshape, 64, 4,2, activation='leakyrelu', weights_init="xavier",regularizer="L2",name='conv3')
pool_3 = max_pool_1d(conv_3, 4,name='pool3')
drug_reshape_4 = tflearn.reshape(pool_3, [-1, 64*16])

merging =  merge([prot_reshape_4,drug_reshape_4],mode='concat',axis=1)
fc_1 = fully_connected(merging, 300, activation='leakyrelu',weights_init="xavier",name='fully1')
drop_2 = dropout(fc_1, 0.8)
fc_2 = fully_connected(drop_2, 100, activation='leakyrelu',weights_init="xavier",name='fully2')
drop_3 = dropout(fc_2, 0.8)
linear = fully_connected(drop_3, 1, activation='linear',name='fully3')
reg = regression(linear, optimizer='adam', learning_rate=0.001,
                     loss='mean_square', name='target')

# Training
model = tflearn.DNN(reg, tensorboard_verbose=0,tensorboard_dir='./mytensor/',checkpoint_path="./checkpoints/")

######## training
model.fit([protein_train,compound_train], {'target': kd_train}, n_epoch=100,batch_size=batch_size,
           validation_set=([protein_dev,compound_dev], {'target': kd_dev}),
            show_metric=True, run_id='joint_model')

# saving model
model.save('my_model.tflearn')

print("error on ER")
size = 5000
num_bins = math.ceil(length_ER/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([feature_ER_protein[0:size,],feature_ER_compound[0:size,]])
        elif i < num_bins-1:
          temp = model.predict([feature_ER_protein[(i*size):((i+1)*size),],feature_ER_compound[(i*size):((i+1)*size),]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([feature_ER_protein[(i*size):length_ER,],feature_ER_compound[(i*size):length_ER,]])
          y_pred = np.concatenate((y_pred,temp), axis=0)

er=0
for i in range(length_ER):
  er += (y_pred[i]-label_ER[i])**2

mse = er/length_ER
print(mse)

results = sm.OLS(y_pred,sm.add_constant(label_ER)).fit()
print(results.summary())


print("error on kinase")
size = 5000
num_bins = math.ceil(length_kinase/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([feature_kinase_protein[0:size,],feature_kinase_compound[0:size,]])
        elif i < num_bins-1:
          temp = model.predict([feature_kinase_protein[(i*size):((i+1)*size),],feature_kinase_compound[(i*size):((i+1)*size),]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([feature_kinase_protein[(i*size):length_kinase,],feature_kinase_compound[(i*size):length_kinase,]])
          y_pred = np.concatenate((y_pred,temp), axis=0)

er=0
for i in range(length_kinase):
  er += (y_pred[i]-label_kinase[i])**2

mse = er/length_kinase
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


print("error on GPCR")
size = 5000
num_bins = math.ceil(length_GPCR/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([feature_GPCR_protein[0:size,],feature_GPCR_compound[0:size,]])
        elif i < num_bins-1:
          temp = model.predict([feature_GPCR_protein[(i*size):((i+1)*size),],feature_GPCR_compound[(i*size):((i+1)*size),]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([feature_GPCR_protein[(i*size):length_GPCR,],feature_GPCR_compound[(i*size):length_GPCR,]])
          y_pred = np.concatenate((y_pred,temp), axis=0)

er=0
for i in range(length_GPCR):
  er += (y_pred[i]-label_GPCR[i])**2

mse = er/length_GPCR
print(mse)

results = sm.OLS(y_pred,sm.add_constant(label_GPCR)).fit()
print(results.summary())


print("error on channel")
size = 5000
num_bins = math.ceil(length_channel/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([feature_channel_protein[0:size,],feature_channel_compound[0:size,]])
        elif i < num_bins-1:
          temp = model.predict([feature_channel_protein[(i*size):((i+1)*size),],feature_channel_compound[(i*size):((i+1)*size),]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([feature_channel_protein[(i*size):length_channel,],feature_channel_compound[(i*size):length_channel,]])
          y_pred = np.concatenate((y_pred,temp), axis=0)

er=0
for i in range(length_channel):
  er += (y_pred[i]-label_channel[i])**2

mse = er/length_channel
print(mse)

results = sm.OLS(y_pred,sm.add_constant(label_channel)).fit()
print(results.summary())

