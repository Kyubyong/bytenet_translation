# -*- coding: utf-8 -*-
'''
This is a TensorFlow implementation of 
Character-Level Machine Translation in the paper 
'Neural Machine Translation in Linear Time' (version updated in 2017)
https://arxiv.org/abs/1610.10099. 

Note that I've changed a few lines in the file
`tensorflow/contrib/layers/python/layers/layer.py` for some reason.
Check below.

line 1532
Before: mean, variance = nn.moments(inputs, axis, keep_dims=True)
After: mean, variance = nn.moments(inputs, [-1], keep_dims=True)

lines 559-562 -> Commented out
inputs_shape[0:1].assert_is_compatible_with(batch_weights.get_shape())
# Reshape batch weight values so they broadcast across inputs.
nshape = [-1] + [1 for _ in range(inputs_rank - 1)]
batch_weights = array_ops.reshape(batch_weights, nshape)

By kyubyong park. kbpark.linguist@gmail.com. https://www.github.com/kyubyong
'''
from __future__ import print_function
from hyperparams import Hp
import tensorflow as tf
import numpy as np
from prepro import *
import os
from tqdm import tqdm

def get_batch_data():
    # Load data
    X, Y = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // Hp.batch_size
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])
            
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=Hp.batch_size, 
                                capacity=Hp.batch_size*64,   
                                min_after_dequeue=Hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch # (64, 100), (64, 100), ()

def embed(tensor, vocab_size, num_units):
    '''
    Args:
      tensor: A 2-D tensor of [batch, time].
      vocab_size: An int. The number of vocabulary.
      num_units: An int. The number of embedding units.

    Returns:
      An embedded tensor whose index zero is associated with constant 0. 
    '''
    lookup_table_for_zero = tf.zeros(shape=[1, num_units], dtype=tf.float32)
    lookup_table_for_others = tf.Variable(tf.truncated_normal(shape=[vocab_size-1, num_units], 
                                                   stddev=0.01))
    lookup_table = tf.concat((lookup_table_for_zero, lookup_table_for_others), 0)
    
    return tf.nn.embedding_lookup(lookup_table, tensor)
    
def normalize_activate(tensor, 
                       normalization_type="ln", 
                       is_training=True):
    '''
    Args:
      tensor: A 3-D or 4-D tensor.
      normalization_type: Either `ln` or `bn`.
      is_training: A boolean. Phase declaration for batch normalization.
    
    Returns:
      A tensor of the same shape as `tensor`, which has been 
      normalized and subsequently activated by Relu.
    '''
    if normalization_type == "ln": # layer normalization
        return tf.contrib.layers.layer_norm(inputs=tensor, center=True, scale=True, 
                                        activation_fn=tf.nn.relu)
    else: # batch normalization
        masks = tf.sign(tf.abs(tensor))
        return tf.contrib.layers.batch_norm(inputs=tensor, center=True, scale=True, 
                    activation_fn=tf.nn.relu, updates_collections=None,
                    is_training=is_training, batch_weights=masks)

def conv1d(tensor, 
           filters, 
           size=1, 
           rate=1, 
           padding="SAME", 
           causal=False,
           use_bias=False):
    '''
    Args:
      tensor: A 3-D tensor of [batch, time, depth].
      filters: An int. Number of outputs (=activation maps)
      size: An int. Filter size.
      rate: An int. Dilation rate.
      padding: Either `SAME` or `VALID`.
      causal: A boolean. If True, zeros of (kernel size - 1) * rate are prepadded
        for causality.
      use_bias: A boolean.
    
    Returns:
      A masked tensor of the sampe shape as `tensor`.
    '''
    
    # ! We need to get masks to zero out the outputs.
    masks = tf.sign(tf.abs(tf.reduce_sum(tensor, axis=-1, keep_dims=True)))
    
    if causal:
        # pre-padding for causality
        pad_len = (size - 1) * rate  # padding size
        tensor = tf.pad(tensor, [[0, 0], [pad_len, 0], [0, 0]])
        padding = "VALID"
        
    params = {"inputs":tensor, "filters":filters, "kernel_size":size,
            "dilation_rate":rate, "padding":padding, "activation":None, 
            "use_bias":use_bias}
    
    out = tf.layers.conv1d(**params)
    
    return out * masks

def _block(tensor, 
           size=3, 
           rate=1, 
           initial=False, 
           is_training=True, 
           normalization_type="ln",
           causal=False):
    '''
    Refer to Figure 3 on page 4 of the original paper.
    Args
      tensor: A 3-D tensor of [batch, time, depth].
      size: An int. Filter size.
      rate: An int. Dilation rate.
      initial: A boolean. If True, `tensor` will not be activated at first.
      is_training: A boolean. Phase declaration for batch normalization.
      normalization_type: Either `ln` or `bn`.
      causal: A boolean. If True, zeros of (kernel size - 1) * rate are prepadded
        for causality.
    
    Returns
      A tensor of the same shape as `tensor`.
    '''
    out = tensor
    
    # input dimension
    in_dim = out.get_shape().as_list()[-1]
    
    if not initial:
        out = normalize_activate(out, is_training=is_training, normalization_type=normalization_type)
    
    # 1 X 1 convolution -> Dimensionality reduction
    out = conv1d(out, filters=in_dim/2, size=1, causal=causal)
    
    # normalize and activate
    out = normalize_activate(out, is_training=is_training, normalization_type=normalization_type)
    
    # 1 X k convolution
    out = conv1d(out, filters=in_dim/2, size=size, rate=rate, causal=causal)
    
    # normalize and activate
    out = normalize_activate(out, is_training=is_training, normalization_type=normalization_type)
    
    # 1 X 1 convolution -> Dimension recovery
    out = conv1d(out, filters=in_dim, size=1, causal=causal)
    
    # Residual connection
    out += tensor
    
    return out 

class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data() # (N, T) (N, T)
                self.y_shifted = tf.concat([tf.zeros((Hp.batch_size, 1), tf.int32), self.y[:, :-1]], 1) # (16, 150) int32
            else: # inference
                self.x = tf.placeholder(tf.int32, shape=(None, Hp.maxlen))
                self.y_shifted = tf.placeholder(tf.int32, shape=(None, Hp.maxlen))
            
            # Load vocabulary    
            char2idx, idx2char = load_vocab()
             
            # Embedding
            self.X = embed(self.x, len(char2idx), Hp.hidden_units)
            self.Y_shifted = embed(self.y_shifted, len(char2idx), Hp.hidden_units)
             
            # Encoding
            for i in range(Hp.num_blocks):
                for rate in (1,2,4,8,16):
                    self.X = _block(self.X, 
                                    size=3, 
                                    rate=rate,
                                    normalization_type="bn",
                                    is_training=is_training,
                                    causal=False,
                                    initial=True if (i==0 and rate==1) else False) # (N, T, C)
                     
            # Decoding
            for i in range(Hp.num_blocks):
                for rate in (1,2,4,8,16):
                    if i==0 and rate==1:
                        self.X = _block(self.X, 
                                        size=3, 
                                        rate=rate, 
                                        causal=True,
                                        initial=False)
                        self.Y_shifted = _block(self.Y_shifted, 
                                                size=3, 
                                                rate=rate, 
                                                causal=True,
                                                initial=True) # (N, T, C)
                        self.dec = self.X + self.Y_shifted
                    else:
                        self.dec = _block(self.dec, 
                                          size=3, 
                                          rate=rate, 
                                          causal=True,
                                          initial=False) # (N, T, C)
             
            # final 1 X 1 convolutional layer for softmax
            self.logits = conv1d(self.dec, filters=len(char2idx), use_bias=True) # (N, T, V)
            
            if is_training:
                # Loss
                ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y) # (16, 100)
                istarget = tf.to_float(tf.not_equal(self.y, 0)) # zeros: 0, non-zeros: 1# (16, 100)
                self.loss = tf.reduce_sum(ce * istarget) / (tf.reduce_sum(istarget) + 0.0000001)
                 
                # Training
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.train_op = tf.train.AdamOptimizer(learning_rate=0.0001)\
                                        .minimize(self.loss, global_step=self.global_step)
                 
                # Summmary 
                tf.summary.scalar('loss', self.loss)
                self.merged = tf.summary.merge_all()
                
            # Predictions
            self.preds = tf.arg_max(self.logits, dimension=-1)

def main():   
    g = Graph("train"); print("Graph loaded")
    char2idx, idx2char = load_vocab()
    
    sv = tf.train.Supervisor(graph=g.graph, logdir=Hp.logdir,
                             save_model_secs=1800)
    
    with sv.managed_session() as sess:
        
        # Write initialized values
        sv.saver.save(sess, Hp.logdir + '/initial_values')
             
        # Training
        for epoch in range(1, 11): 
            if sv.should_stop(): break
            for step in tqdm(range(g.num_batch), total=g.num_batch, ncols=70, leave=False, unit='b'):
                sess.run(g.train_op)
                
#                 # Check ongoing samples
#                 if step % 100 == 0:
#                     _loss, _y, _preds = sess.run([g.loss, g.y, g.preds])
#                     print("loss:", _loss)
#                     for yy, pp in zip(_y, _pred):
#                         print("expected: ", "".join(idx2char[yyy] for yyy in yy))
#                         print("got: ", "".join(idx2char[ppp] for ppp in pp))
#                         print("")
                 
            # Write checkpoint files 
            loss, gs = sess.run([g.loss, g.global_step])  
            print("After epoch %02d, the training loss is %.2f" % (epoch, loss))
            sv.saver.save(sess, Hp.logdir + '/model_epoch_%02d_gs_%d_loss_%.2f' % (epoch, gs, loss))
        
if __name__ == '__main__':
    main()
    print("Done")

