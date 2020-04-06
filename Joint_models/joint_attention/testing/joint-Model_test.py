## importing part
from __future__ import division, print_function, absolute_import
import statsmodels.api as sm
import gzip
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
from tflearn.layers.conv import conv_2d, max_pool_2d,conv_1d,max_pool_1d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from tflearn.layers.merge_ops import merge
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell,GRUCell
#from recurrent import bidirectional_rnn, BasicLSTMCell,GRUCell

random.seed(1234)
#### data and vocabulary
label_type = 'ic50'
if len(sys.argv) == 2:
    checkpoint_file = sys.argv[1]
elif len(sys.argv) == 3:
    checkpoint_file = sys.argv[1]
    label_type = sys.argv[2].lower()
else:
    checkpoint_file = 'checkpoints-741400'

data_dir="./data"
vocab_size_compound=68
vocab_size_protein=76
comp_MAX_size=100
protein_MAX_size=152
vocab_compound="vocab_compound"
vocab_protein="vocab_protein"
batch_size = 64

GRU_size_prot=256
GRU_size_drug=128

dev_perc=0.1

## Padding part
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
_WORD_SPLIT = re.compile(b"(\S)")
_WORD_SPLIT_2 = re.compile(b",")
_DIGIT_RE = re.compile(br"\d")


## functions
def basic_tokenizer(sentence,condition):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    if condition ==0:
        l = _WORD_SPLIT.split(space_separated_fragment)
        del l[0::2]
    elif condition == 1:
        l = _WORD_SPLIT_2.split(space_separated_fragment)
    words.extend(l)
  return [w for w in words if w]

def sentence_to_token_ids(sentence, vocabulary,condition,normalize_digits=False):

  words = basic_tokenizer(sentence,condition)
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)



def data_to_token_ids(data_path, target_path, vocabulary_path, condition,normalize_digits=False):
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="rb") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          token_ids = sentence_to_token_ids(tf.compat.as_bytes(line), vocab, condition,normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def read_data(source_path,MAX_size):
  data_set = []
  length = []
  mycount=0
  with tf.gfile.GFile(source_path, mode="r") as source_file:
      source = source_file.readline()
      counter = 0
      while source:
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        if len(source_ids) < MAX_size:
           length.append(len(source_ids))
           pad = [PAD_ID] * (MAX_size - len(source_ids))
           data_set.append(list(source_ids + pad))
           mycount=mycount+1
        elif len(source_ids) == MAX_size:
           length.append(len(source_ids))
           data_set.append(list(source_ids))
           mycount=mycount+1
        else:
           print("there is a data with length bigger than the max\n")
           print(len(source_ids))
        source = source_file.readline()
  return data_set,length

def prepare_data(data_dir, train_path, vocabulary_size,vocab,max_size,condition):
  vocab_path = os.path.join(data_dir, vocab)

  train_ids_path = train_path + (".ids%d" % vocabulary_size)
  data_to_token_ids(train_path, train_ids_path, vocab_path,condition)
  train_set,length = read_data(train_ids_path,max_size)

  return train_set,length


def read_labels(path):
    x = []
    f = open(path, "r") 
    for line in f:
         if (line[0]=="<")or(line[0]==">"): 
            print("Inequality in IC50!!!\n")
         else:
            x.append(float(line)) 
 
    return x


def read_initial_state_weigths(path,size1,size2):
    x = []
    f = open(path, "r")
    count = 0;
    for line in f:
       y = [float(n) for n in line.split(" ")]
       if len(y) == size2:
          x.append(y)
          count = count+1
       else:
          print("not exactly equal to size2!!!!!!")
    
    return x

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

################ Reading initial states and weigths 
prot_init_state_1 = read_initial_state_weigths("./data/prot_init/first_layer_states.txt",batch_size,GRU_size_prot)
prot_init_state_1 = tf.convert_to_tensor(np.reshape(prot_init_state_1,[batch_size,GRU_size_prot]),dtype=tf.float32)

prot_init_state_2 = read_initial_state_weigths("./data/prot_init/second_layer_states.txt",batch_size,GRU_size_prot)
prot_init_state_2 = tf.convert_to_tensor(np.reshape(prot_init_state_2,[batch_size,GRU_size_prot]),dtype=tf.float32)

drug_init_state_1 = read_initial_state_weigths("./data/drug_init/first_layer_states.txt",batch_size,GRU_size_drug)
drug_init_state_1 = tf.convert_to_tensor(np.reshape(drug_init_state_1,[batch_size,GRU_size_drug]),dtype=tf.float32)

drug_init_state_2 = read_initial_state_weigths("./data/drug_init/second_layer_states.txt",batch_size,GRU_size_drug)
drug_init_state_2 = tf.convert_to_tensor(np.reshape(drug_init_state_2,[batch_size,GRU_size_drug]),dtype=tf.float32)

## preparing data 


test_protein,test_protein_length = prepare_data(data_dir,"./data/test_sps",vocab_size_protein,vocab_protein,protein_MAX_size,1)
test_compound,test_compound_length = prepare_data(data_dir,"./data/test_smile",vocab_size_compound,vocab_compound,comp_MAX_size,0)
test_IC50 = read_labels("./data/test_" + label_type)


## RNN for protein
prot_data = input_data(shape=[None, protein_MAX_size])
prot_embd = tflearn.embedding(prot_data, input_dim=vocab_size_protein, output_dim=GRU_size_prot)
prot_gru_1 = tflearn.gru(prot_embd, GRU_size_prot,initial_state= prot_init_state_1,trainable=True,return_seq=True,restore=True)
prot_gru_1 = tf.stack(prot_gru_1,axis=1)
prot_gru_2 = tflearn.gru(prot_gru_1, GRU_size_prot,initial_state= prot_init_state_2,trainable=True,return_seq=True,restore=True)
prot_gru_2=tf.stack(prot_gru_2,axis=1)

drug_data = input_data(shape=[None, comp_MAX_size])
drug_embd = tflearn.embedding(drug_data, input_dim=vocab_size_compound, output_dim=GRU_size_drug)
drug_gru_1 = tflearn.gru(drug_embd,GRU_size_drug,initial_state= drug_init_state_1,trainable=True,return_seq=True,restore=True)
drug_gru_1 = tf.stack(drug_gru_1,1)
drug_gru_2 = tflearn.gru(drug_gru_1, GRU_size_drug,initial_state= drug_init_state_2,trainable=True,return_seq=True,restore=True)
drug_gru_2 = tf.stack(drug_gru_2,axis=1)


W = tflearn.variables.variable(name="Attn_W_prot",shape=[GRU_size_prot,GRU_size_drug],initializer=tf.random_normal([GRU_size_prot,GRU_size_drug],stddev=0.1),restore=True)

b = tflearn.variables.variable(name="Attn_b_prot",shape=[protein_MAX_size,comp_MAX_size],initializer=tf.random_normal([protein_MAX_size,comp_MAX_size],stddev=0.1),restore=True)
V = tf.tensordot(prot_gru_2,W,axes=[[2],[0]])
for i in range(batch_size):
   temp = tf.expand_dims(tf.tanh(tf.tensordot(tflearn.reshape(tf.slice(V,[i,0,0],[1,protein_MAX_size,GRU_size_drug]),[protein_MAX_size,GRU_size_drug]),tflearn.reshape(tf.slice(drug_gru_2,[i,0,0],[1,comp_MAX_size,GRU_size_drug]),[comp_MAX_size,GRU_size_drug]),axes=[[1],[1]])+b),0)
   if i==0:
     VU = temp
   else:
     VU = merge([VU,temp],mode='concat',axis=0)

VU = tflearn.reshape(VU,[-1,comp_MAX_size*protein_MAX_size])
alphas_pair = tf.nn.softmax(VU,name='alphas')
alphas_pair = tflearn.reshape(alphas_pair,[-1,protein_MAX_size,comp_MAX_size])

U_size = 256
U_prot = tflearn.variables.variable(name="Attn_U_prot",shape=[U_size,GRU_size_prot],initializer=tf.random_normal([U_size,GRU_size_prot],stddev=0.1),restore=True)
U_drug = tflearn.variables.variable(name="Attn_U_drug",shape=[U_size,GRU_size_drug],initializer=tf.random_normal([U_size,GRU_size_drug],stddev=0.1),restore=True)
B = tflearn.variables.variable(name="Attn_B",shape=[U_size],initializer=tf.random_normal([U_size],stddev=0.1),restore=True)

prod_drug = tf.tensordot(drug_gru_2,U_drug,axes=[[2],[1]])
prod_prot = tf.tensordot(prot_gru_2,U_prot,axes=[[2],[1]])

Attn = tflearn.variables.variable(name="Attn",shape=[batch_size,U_size],initializer=tf.zeros([batch_size,U_size]),restore=True)

for i in range(comp_MAX_size):
        temp = tf.expand_dims(tflearn.reshape(tf.slice(prod_drug,[0,i,0],[batch_size,1,U_size]),[batch_size,U_size]),axis=1) + prod_prot + B
        Attn = Attn + tf.reduce_sum(tf.multiply(tf.tile(tflearn.reshape(tf.slice(alphas_pair,[0,0,i],[batch_size,protein_MAX_size,1]),[batch_size,protein_MAX_size,1]),[1,1,U_size]),temp),axis=1)


Attn_reshape = tflearn.reshape(Attn, [-1, U_size,1])
conv_1 = conv_1d(Attn_reshape, 64, 4,2, activation='leakyrelu', weights_init="xavier",regularizer="L2",name='conv1')
pool_1 = max_pool_1d(conv_1, 4,name='pool1')

pool_2 = tflearn.reshape(pool_1, [-1, 64*32])

fc_1 = fully_connected(pool_2, 600, activation='leakyrelu',weights_init="xavier",name='fully1')
drop_2 = dropout(fc_1, 0.8)
fc_2 = fully_connected(drop_2, 300, activation='leakyrelu',weights_init="xavier",name='fully2')
drop_3 = dropout(fc_2, 0.8)
linear = fully_connected(drop_3, 1, activation='linear',name='fully3')
reg = regression(linear, optimizer='adam', learning_rate=0.0001,
                     loss='mean_square', name='target')

model = tflearn.DNN(reg, tensorboard_verbose=0,tensorboard_dir='./mytensor/',checkpoint_path="./checkpoints/")
#checkpoints = 741400

#model.load('checkpoints-'+str(checkpoints))
model.load(checkpoint_file)
print("error on test")
size = 64
length_test = len(test_protein)
print(length_test)
num_bins = math.ceil(length_test/size)
for i in range(num_bins):
        if i==0:
          y_pred = model.predict([test_protein[0:size],test_compound[0:size]])
        elif i < num_bins-1:
          temp = model.predict([test_protein[(i*size):((i+1)*size)],test_compound[(i*size):((i+1)*size)]])
          y_pred = np.concatenate((y_pred,temp), axis=0)
        else:
          temp = model.predict([test_protein[length_test-size:length_test],test_compound[length_test-size:length_test]])
          y_pred = np.concatenate((y_pred,temp[size-length_test+(i*size):size]), axis=0)

er=0
for i in range(length_test):
  er += (y_pred[i]-test_IC50[i])**2

mse = er/length_test
print(mse)

results = sm.OLS(y_pred,sm.add_constant(test_IC50)).fit()
print(results.summary())

