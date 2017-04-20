#-*- coding: utf-8-*-
from __future__ import print_function
from hyperparams import Hp
import codecs
import tensorflow as tf
import numpy as np
from prepro import *
from train import Graph
from nltk.translate.bleu_score import corpus_bleu

# Hyperparameters
BATCH_SIZE = 128

def eval(): 
    # Load graph
    g = Graph(is_training=False)
    i = 0    
    with tf.Session(graph=g.graph) as sess:
        saver = tf.train.Saver()
        # Restore parameters
#         import glob
#         ckpt_files = glob.glob('asset/train/*.index')
#         for ckpt_file in ckpt_files:
#             saver.restore(sess, ckpt_file.replace(".index", ""))
#             print("Restored!")
#             mname = ckpt_file.replace(".index", "").split('/')[-1]
#             print(mname)
            
         
        saver.restore(sess, tf.train.latest_checkpoint(Hp.logdir))
        print("Restored!")
        mname = open('asset/train/checkpoint', 'r').read().split('"')[1] # model name
        if i==0:
            # Load data
            X, Sources, Targets = load_test_data()
            char2idx, idx2char = load_vocab()
             
            with codecs.open(mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                for i in range(len(X) // BATCH_SIZE):
                    
                    # Get mini-batches
                    x = X[i*BATCH_SIZE: (i+1)*BATCH_SIZE] # mini-batch
                    sources = Sources[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
                    targets = Targets[i*BATCH_SIZE: (i+1)*BATCH_SIZE]
                     
                    preds_prev = np.zeros((BATCH_SIZE, Hp.maxlen), np.int32)
                    preds = np.zeros((BATCH_SIZE, Hp.maxlen), np.int32)        
                    for j in range(Hp.maxlen):
                        # predict next character
                        outs = sess.run(g.preds, {g.x: x, g.y_shifted: preds_prev})
                        # update character sequence
                        if j < Hp.maxlen - 1:
                            preds_prev[:, j + 1] = outs[:, j]
                        preds[:, j] = outs[:, j]
                     
                    # Write to file
                    for source, target, pred in zip(sources, targets, preds): # sentence-wise
                        got = "".join(idx2char[idx] for idx in pred).split(u"␃")[0]
                        fout.write("- source: " + source +"\n")
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.flush()
                         
                        # For bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 2:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)
                 
                # Get bleu score
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100*score))
                                                
if __name__ == '__main__':
    eval()
    print("Done")
    
    
