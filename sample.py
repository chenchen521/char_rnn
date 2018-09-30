#!/usr/bin/env python

from __future__ import print_function

import os
from six.moves import cPickle
from collections import namedtuple

import tensorflow as tf
from model import Model

def sample():
#    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
#        saved_args = cPickle.load(f)
    with open(os.path.join('D:/presto/Python/Lib/site-packages/char/save/chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)

    Args = namedtuple("Args", ['model', 'num_layers', 'rnn_size', 'seq_length', 'batch_size', 'save_every', 'prime'])
    
    args = Args(model = 'lstm', num_layers = 2, rnn_size = 128, seq_length = 1, batch_size = 1, save_every = 1000, prime = ' ')
    
    #Use most frequent char if no prime is given
#    if args.prime == '':
#        args.prime = chars[0]
        
    model = Model(args, training=False)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('D:/presto/Python/Lib/site-packages/char/save/')
        if ckpt and ckpt.model_checkpoint_path:
            #同樣變量名的物件需要事先存在代碼中， 并且數據類型和長相必須一模一樣
            #restore()之後，儲存的參數起死回生一樣存在代碼中
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('>>>>>>>>>>>', model.sample(sess, chars, vocab, 500, args.prime,
                               1).encode('utf-8'), '>>>>>>>>>>>')

#if __name__ == '__main__':
#    sample()
