#!/usr/bin/env python

from __future__ import print_function

import time
import os
from six.moves import cPickle
from collections import namedtuple
import tensorflow as tf

from char.utils import TextLoader
from char.model import Model

def train():
    
    data_dir = 'D:/presto/Python/Lib/site-packages/char/data/tinyshakespeare/'
    batch_size = 50
    seq_length = 50
    data_loader = TextLoader(data_dir, batch_size, seq_length)

    with open(os.path.join('D:/presto/Python/Lib/site-packages/char/save/', 'chars_vocab.pkl'), 'wb') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)
    
    Args = namedtuple("Args", ['model', 'num_layers', 'rnn_size', 'seq_length', 'vocab_size', 'batch_size', 'save_every'])
    
    args = Args(model = 'lstm', num_layers = 2, rnn_size = 128, seq_length = 50, vocab_size = data_loader.vocab_size, batch_size = 50, save_every = 1000)
    
    model = Model(args)

    with tf.Session() as sess:
        # instrument for tensorboard
        summaries = tf.summary.merge_all()  #將圖形，訓練過程等數據合并在一起
        writer = tf.summary.FileWriter(
                os.path.join('logs', time.strftime("%Y-%m-%d-%H-%M-%S")))  #定義一個寫入summary的目標文件，tensorboard圖像在這裏顯示
        writer.add_graph(sess.graph)  #添加graph圖

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        # restore model
#        if args.init_from is not None:
#            saver.restore(sess, ckpt)
        
        for e in range(50): #num_epochs
            sess.run(tf.assign(model.lr,
                               0.002 * (0.97 ** e)))  # learning_rate = 0.002, decay_rate = 0.97 
            data_loader.reset_batch_pointer()
            
            #初始狀態激活
            state = sess.run(model.initial_state)
            for b in range(data_loader.num_batches):  # num_batches = 446
                start = time.time()
                
                #x.shape = (50,50), y.shape = (50, 50)
                x, y = data_loader.next_batch()
                feed = {model.input_data: x, model.targets: y}  #生成feed數據,每一個batch生成一個新的feed
                for i, (c, h) in enumerate(model.initial_state):
                    feed[c] = state[i].c
                    feed[h] = state[i].h

                # instrument for tensorboard
                summ, train_loss, state, _ = sess.run([summaries, model.cost, model.final_state, model.train_op], feed) #feed數據，運行流程圖
                writer.add_summary(summ, e * data_loader.num_batches + b)    #寫入文件

                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}"
                      .format(e * data_loader.num_batches + b,
                              50 * data_loader.num_batches,
                              e, train_loss, end - start))
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                        or (e == 50 - 1 and
                            b == data_loader.num_batches-1):
                    # save for the last result
                    checkpoint_path = os.path.join('D:/presto/Python/Lib/site-packages/char/save/', 'model.ckpt')
                    saver.save(sess, checkpoint_path,
                               global_step=e * data_loader.num_batches + b)   #保存模型
                    print("model saved to {}".format(checkpoint_path))

if __name__ == '__main__':
    train()