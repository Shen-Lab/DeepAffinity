# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
#from tensorflow.python.ops import rnn_cell_impl
import data_utils
import seq2seq # MK added

from tensorflow.python.util.nest import is_sequence #MK
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.util import nest
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import nn_ops

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"


def _like_rnncell(cell):
  """Checks that a given object is an RNNCell by using duck typing."""
  conditions = [hasattr(cell, "output_size"), hasattr(cell, "state_size"),
                hasattr(cell, "zero_state"), callable(cell)]
  return all(conditions)


def _concat(prefix, suffix, static=False):
  """Concat that enables int, Tensor, or TensorShape values.

  This function takes a size specification, which can be an integer, a
  TensorShape, or a Tensor, and converts it into a concatenated Tensor
  (if static = False) or a list of integers (if static = True).

  Args:
    prefix: The prefix; usually the batch size (and/or time step size).
      (TensorShape, int, or Tensor.)
    suffix: TensorShape, int, or Tensor.
    static: If `True`, return a python list with possibly unknown dimensions.
      Otherwise return a `Tensor`.

  Returns:
    shape: the concatenation of prefix and suffix.

  Raises:
    ValueError: if `suffix` is not a scalar or vector (or TensorShape).
    ValueError: if prefix or suffix was `None` and asked for dynamic
      Tensors out.
  """
  if isinstance(prefix, ops.Tensor):
    p = prefix
    p_static = tensor_util.constant_value(prefix)
    if p.shape.ndims == 0:
      p = array_ops.expand_dims(p, 0)
    elif p.shape.ndims != 1:
     raise ValueError("prefix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % p)
  else:
    p = tensor_shape.as_shape(prefix)
    p_static = p.as_list() if p.ndims is not None else None
    p = (constant_op.constant(p.as_list(), dtype=dtypes.int32)
         if p.is_fully_defined() else None)
  if isinstance(suffix, ops.Tensor):
    s = suffix
    s_static = tensor_util.constant_value(suffix)
    if s.shape.ndims == 0:
      s = array_ops.expand_dims(s, 0)
    elif s.shape.ndims != 1:
      raise ValueError("suffix tensor must be either a scalar or vector, "
                       "but saw tensor: %s" % s)
  else:
    s = tensor_shape.as_shape(suffix)
    s_static = s.as_list() if s.ndims is not None else None
    s = (constant_op.constant(s.as_list(), dtype=dtypes.int32)
         if s.is_fully_defined() else None)

  if static:
    shape = tensor_shape.as_shape(p_static).concatenate(s_static)
    shape = shape.as_list() if shape.ndims is not None else None
  else:
    if p is None or s is None:
      raise ValueError("Provided a prefix or suffix of None: %s and %s"
                       % (prefix, suffix))
    shape = array_ops.concat((p, s), 0)
  return shape

def _zero_state_tensors(state_size, batch_size, dtype):
  """Create tensors of zeros based on state_size, batch_size, and dtype."""
  def get_state_shape(s):
    """Combine s with batch_size to get a proper tensor shape."""
    c = _concat(batch_size, s)
    c_static = _concat(batch_size, s, static=True)
    size = array_ops.zeros(c, dtype=dtype)
    size.set_shape(c_static)
    return size
  return nest.map_structure(get_state_shape, state_size)


class RNNCell(base_layer.Layer):

  def __call__(self, inputs, state, scope=None):
    if scope is not None:
      with vs.variable_scope(scope,
                             custom_getter=self._rnn_get_variable) as scope:
        return super(RNNCell, self).__call__(inputs, state, scope=scope)
    else:
      with vs.variable_scope(vs.get_variable_scope(),
                             custom_getter=self._rnn_get_variable):
        return super(RNNCell, self).__call__(inputs, state)

  def _rnn_get_variable(self, getter, *args, **kwargs):
    variable = getter(*args, **kwargs)
    trainable = (variable in tf_variables.trainable_variables() or
                 (isinstance(variable, tf_variables.PartitionedVariable) and
                  list(variable)[0] in tf_variables.trainable_variables()))
    if trainable and variable not in self._trainable_weights:
      self._trainable_weights.append(variable)
    elif not trainable and variable not in self._non_trainable_weights:
      self._non_trainable_weights.append(variable)
    return variable

  @property
  def state_size(self):
    """size(s) of state(s) used by this cell.

    It can be represented by an Integer, a TensorShape or a tuple of Integers
    or TensorShapes.
    """
    raise NotImplementedError("Abstract method")

  @property
  def output_size(self):
    """Integer or TensorShape: size of outputs produced by this cell."""
    raise NotImplementedError("Abstract method")

  def build(self, _):
    # This tells the parent Layer object that it's OK to call
    # self.add_variable() inside the call() method.
    pass

  def zero_state(self, batch_size, dtype):
    """Return zero-filled state tensor(s).

    Args:
      batch_size: int, float, or unit Tensor representing the batch size.
      dtype: the data type to use for the state.

    Returns:
      If `state_size` is an int or TensorShape, then the return value is a
      `N-D` tensor of shape `[batch_size x state_size]` filled with zeros.

      If `state_size` is a nested list or tuple, then the return value is
      a nested list or tuple (of the same structure) of `2-D` tensors with
      the shapes `[batch_size x s]` for each s in `state_size`.
    """
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      state_size = self.state_size
      return _zero_state_tensors(state_size, batch_size, dtype)





class GRUCell(RNNCell):
  """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078)."""

  def __init__(self,
               num_units,
               activation=None,
               reuse=None,
               kernel_initializer=None,
               bias_initializer=None):
    super(GRUCell, self).__init__(_reuse=reuse)
    self._num_units = num_units
    self._activation = activation or math_ops.tanh
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return self._num_units

  @property
  def output_size(self):
    return self._num_units

  def call(self, inputs, state):
    """Gated recurrent unit (GRU) with nunits cells."""
    with vs.variable_scope("gates"):  # Reset gate and update gate.
      # We start with bias of 1.0 to not reset and not update.
      bias_ones = self._bias_initializer
      if self._bias_initializer is None:
        dtype = [a.dtype for a in [inputs, state]][0]
        bias_ones = init_ops.constant_initializer(1.0, dtype=dtype)
      value = math_ops.sigmoid(
          _linear([inputs, state], 2 * self._num_units, True, bias_ones,
                  self._kernel_initializer))
      r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
    with vs.variable_scope("candidate"):
      c = self._activation(
          _linear([inputs, r * state], self._num_units, True,
                  self._bias_initializer, self._kernel_initializer))
    new_h = u * state + (1 - u) * c
    return new_h, new_h



class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.
  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self,
               source_vocab_size,
               target_vocab_size,
               buckets,
               size,
               num_layers,
               max_gradient_norm,
               batch_size,
               learning_rate,
               learning_rate_decay_factor,
               use_lstm=False,
               num_samples=512,
               forward_only=False,
               dtype=tf.float32):
    """Create the model.
    Args:
      source_vocab_size: size of the source vocabulary.
      target_vocab_size: size of the target vocabulary.
      buckets: a list of pairs (I, O), where I specifies maximum input length
        that will be processed in that bucket, and O specifies maximum output
        length. Training instances that have inputs longer than I or outputs
        longer than O will be pushed to the next bucket and padded accordingly.
        We assume that the list is sorted, e.g., [(2, 4), (8, 16)].
      size: number of units in each layer of the model.
      num_layers: number of layers in the model.
      max_gradient_norm: gradients will be clipped to maximally this norm.
      batch_size: the size of the batches used during training;
        the model construction is independent of batch_size, so it can be
        changed after initialization if this is convenient, e.g., for decoding.
      learning_rate: learning rate to start with.
      learning_rate_decay_factor: decay learning rate by this much when needed.
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
      dtype: the data type to use to store internal variables.
    """
    self.source_vocab_size = source_vocab_size
    self.target_vocab_size = target_vocab_size
    self.buckets = buckets
    self.batch_size = batch_size
    self.learning_rate = tf.Variable(
        float(learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(
        self.learning_rate * learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)
    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    if num_samples > 0 and num_samples < self.target_vocab_size:
      w_t = tf.get_variable("proj_w", [self.target_vocab_size, size], dtype=dtype)
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [self.target_vocab_size], dtype=dtype)
      output_projection = (w, b)

      def sampled_loss(labels, logits):
        labels = tf.reshape(labels, [-1, 1])
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(logits, tf.float32)
        return tf.cast(
            tf.nn.sampled_softmax_loss(
                weights=local_w_t,
                biases=local_b,
                labels=labels,
                inputs=local_inputs,
                num_sampled=num_samples,
                num_classes=self.target_vocab_size),
            dtype)
      softmax_loss_function = sampled_loss


    def get_lstm(): # MK add this function
       cell = tf.contrib.rnn.BasicLSTMCell(size, state_is_tuple=True,reuse=tf.get_variable_scope().reuse)
       return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)


    # Create the internal multi-layer cell for our RNN.
    def single_cell():
      #return tf.contrib.rnn.GRUCell(size) #MK add dropout
      cell = tf.contrib.rnn.GRUCell(size) 
      #cell = GRUCell(size) #MK
      return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.8)
    if use_lstm:
      def single_cell():
        return tf.contrib.rnn.BasicLSTMCell(size)
    cell = single_cell()
    if num_layers > 1: #MK change for testing
      cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
      #cell = tf.contrib.rnn.MultiRNNCell([get_lstm() for _ in range(num_layers)],state_is_tuple=True)  # MK change

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
       return seq2seq.embedding_rnn_seq2seq(
           encoder_inputs,
           decoder_inputs,
           cell,
           num_encoder_symbols=source_vocab_size,
           num_decoder_symbols=target_vocab_size,
           embedding_size=size,
           output_projection=output_projection,
           feed_previous=do_decode,
           dtype=dtype)



    # Feeds for inputs.
    self.encoder_inputs = []
    self.decoder_inputs = []
    self.target_weights = []
    for i in xrange(buckets[-1][0]):  # Last bucket is the biggest one.
      self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], #MK
                                                name="encoder{0}".format(i)))

    for i in xrange(buckets[-1][1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], #MK
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(dtype, shape=[None], #MK
                                                name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in xrange(len(self.decoder_inputs) - 1)]

    # Training outputs and losses.
    if forward_only:
      #self.outputs, self.losses,self.states = tf.contrib.legacy_seq2seq.model_with_buckets( #MK change
      self.outputs, self.losses,self.states,self.enc_outputs = seq2seq.model_with_buckets( #MK change
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets, lambda x, y: seq2seq_f(x, y, True),
          softmax_loss_function=softmax_loss_function)
      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        for b in xrange(len(buckets)):
          self.outputs[b] = [
              tf.matmul(output, output_projection[0]) + output_projection[1]
              for output in self.outputs[b]
          ]
    else:
     # self.outputs, self.losses,_ = tf.contrib.legacy_seq2seq.model_with_buckets( # MK change
      self.outputs, self.losses,self.states,self.enc_outputs = seq2seq.model_with_buckets( # MK change 
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, buckets,
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()

    #if not forward_only: #MK
    if True:   
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      for b in xrange(len(buckets)):
        gradients = tf.gradients(self.losses[b], params)
        clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                         max_gradient_norm)
        self.gradient_norms.append(norm)
        self.updates.append(opt.apply_gradients(
            zip(clipped_gradients, params), global_step=self.global_step))

   
    self.saver = tf.train.Saver(tf.global_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights,
           bucket_id, forward_only,is_saving):
    """Run a step of the model feeding the given inputs.
    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      bucket_id: which bucket of the model to use.
      forward_only: whether to do the backward step or only forward.
    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.
    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size for the specified bucket_id.
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.buckets[bucket_id]
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in xrange(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in xrange(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[bucket_id],  # Update Op that does SGD.
                     self.gradient_norms[bucket_id],  # Gradient norm.
                     self.losses[bucket_id]]  # Loss for this batch.
    else:
      output_feed = [self.losses[bucket_id]]  # Loss for this batch.
      output_feed.append(self.updates[bucket_id]) #MK
      for l in xrange(decoder_size):  # Output logits.
        output_feed.append(self.outputs[bucket_id][l])

      
    output_feed.append(self.states[bucket_id]) #MK change
    output_feed.append(self.enc_outputs[bucket_id]) #MK change
    outputs = session.run(output_feed, input_feed)

    if is_saving == 1:
     variables_names = []
     for v in tf.trainable_variables():
       if "embedding_rnn_seq2seq/rnn/embedding_wrapper/multi_rnn_cell/cell_0/gru_cell/gates/kernel" in v.name:
          variables_names.append(v)
       elif "embedding_rnn_seq2seq/rnn/embedding_wrapper/multi_rnn_cell/cell_0/gru_cell/gates/bias" in v.name:
          variables_names.append(v)
       elif "embedding_rnn_seq2seq/rnn/embedding_wrapper/multi_rnn_cell/cell_0/gru_cell/candidate/kernel" in v.name:
          variables_names.append(v)
       elif "embedding_rnn_seq2seq/rnn/embedding_wrapper/multi_rnn_cell/cell_0/gru_cell/candidate/bias" in v.name:
          variables_names.append(v)
       elif "embedding_rnn_seq2seq/rnn/embedding_wrapper/multi_rnn_cell/cell_1/gru_cell/gates/kernel" in v.name:
          variables_names.append(v)
       elif "embedding_rnn_seq2seq/rnn/embedding_wrapper/multi_rnn_cell/cell_1/gru_cell/gates/bias" in v.name:
          variables_names.append(v)
       elif "embedding_rnn_seq2seq/rnn/embedding_wrapper/multi_rnn_cell/cell_1/gru_cell/candidate/kernel" in v.name:
          variables_names.append(v)
       elif "embedding_rnn_seq2seq/rnn/embedding_wrapper/multi_rnn_cell/cell_1/gru_cell/candidate/bias" in v.name:
          variables_names.append(v)
       elif "embedding_rnn_seq2seq/rnn/embedding_wrapper/embedding" in v.name:
          variables_names.append(v)
     results = session.run(variables_names,input_feed)

     weigths = np.array(results[0])
     weigths_mat = np.matrix(weigths)
     with open('embedding_W.txt','wb') as f:
       for line in weigths_mat:
          np.savetxt(f, line, fmt='%.2f')

     weigths = np.array(results[1])
     weigths_mat = np.matrix(weigths)
     with open('cell_0_gates_kernel.txt','wb') as f:
       for line in weigths_mat:
          np.savetxt(f, line, fmt='%.2f')

     weigths = np.array(results[2])
     weigths_mat = np.matrix(weigths)
     with open('cell_0_gates_bias.txt','wb') as f:
       for line in weigths_mat:
          np.savetxt(f, line, fmt='%.2f')

     weigths = np.array(results[3])
     weigths_mat = np.matrix(weigths)
     with open('cell_0_candidate_kernel.txt','wb') as f:
       for line in weigths_mat:
          np.savetxt(f, line, fmt='%.2f')

     weigths = np.array(results[4])
     weigths_mat = np.matrix(weigths)
     with open('cell_0_candidate_bias.txt','wb') as f:
       for line in weigths_mat:
          np.savetxt(f, line, fmt='%.2f')

     weigths = np.array(results[5])
     weigths_mat = np.matrix(weigths)
     with open('cell_1_gates_kernel.txt','wb') as f:
       for line in weigths_mat:
          np.savetxt(f, line, fmt='%.2f')

     weigths = np.array(results[6])
     weigths_mat = np.matrix(weigths)
     with open('cell_1_gates_bias.txt','wb') as f:
       for line in weigths_mat:
          np.savetxt(f, line, fmt='%.2f')

     weigths = np.array(results[7])
     weigths_mat = np.matrix(weigths)
     with open('cell_1_candidate_kernel.txt','wb') as f:
       for line in weigths_mat:
          np.savetxt(f, line, fmt='%.2f')

     weigths = np.array(results[8])
     weigths_mat = np.matrix(weigths)
     with open('cell_1_candidate_bias.txt','wb') as f:
       for line in weigths_mat:
          np.savetxt(f, line, fmt='%.2f')
  
    if not forward_only:
      return outputs[1], outputs[2], None,outputs[-2],outputs[-1]  # Gradient norm, loss, no outputs,states.  # MK change
    else:
      return None, outputs[0], outputs[2:-2],outputs[-2],outputs[-1]  # No gradient norm, loss, outputs,states.   # MK change

  def get_batch(self, data, bucket_id):
    """Get a random batch of data from the specified bucket, prepare for step.
    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.
    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.
      bucket_id: integer, which bucket to get the batch for.
    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.buckets[bucket_id]
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in xrange(self.batch_size):
      encoder_input, decoder_input = random.choice(data[bucket_id])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [data_utils.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([data_utils.GO_ID] + decoder_input +
                            [data_utils.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in xrange(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in xrange(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in xrange(self.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.batch_size, dtype=np.float32)
      for batch_idx in xrange(self.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == data_utils.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights




def _linear(args,
            output_size,
            bias,
            bias_initializer=None,
            kernel_initializer=None):
  """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

  Args:
    args: a 2D Tensor or a list of 2D, batch x n, Tensors.
    output_size: int, second dimension of W[i].
    bias: boolean, whether to add a bias term or not.
    bias_initializer: starting value to initialize the bias
      (default is all zeros).
    kernel_initializer: starting value to initialize the weight.

  Returns:
    A 2D Tensor with shape [batch x output_size] equal to
    sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """
  if args is None or (nest.is_sequence(args) and not args):
    raise ValueError("`args` must be specified")
  if not nest.is_sequence(args):
    args = [args]

  # Calculate the total size of arguments on dimension 1.
  total_arg_size = 0
  shapes = [a.get_shape() for a in args]
  for shape in shapes:
    if shape.ndims != 2:
      raise ValueError("linear is expecting 2D arguments: %s" % shapes)
    if shape[1].value is None:
      raise ValueError("linear expects shape[1] to be provided for shape %s, "
                       "but saw %s" % (shape, shape[1]))
    else:
      total_arg_size += shape[1].value

  dtype = [a.dtype for a in args][0]

  # Now the computation.
  scope = vs.get_variable_scope()
  with vs.variable_scope(scope) as outer_scope:
    weights = vs.get_variable(
        _WEIGHTS_VARIABLE_NAME, [total_arg_size, output_size],
        dtype=dtype,
        initializer=kernel_initializer)
    if len(args) == 1:
      res = math_ops.matmul(args[0], weights)
    else:
      res = math_ops.matmul(array_ops.concat(args, 1), weights)
    if not bias:
      return res
    with vs.variable_scope(outer_scope) as inner_scope:
      inner_scope.set_partitioner(None)
      if bias_initializer is None:
        bias_initializer = init_ops.constant_initializer(0.0, dtype=dtype)
      biases = vs.get_variable(
          _BIAS_VARIABLE_NAME, [output_size],
          dtype=dtype,
          initializer=bias_initializer)
    return nn_ops.bias_add(res, biases)
