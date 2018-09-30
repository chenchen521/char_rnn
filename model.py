import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, args, training=True):
        self.args = args
#        if not training:
#            args.batch_size = 1
#            args.seq_length = 1
        cell_fn = rnn.LSTMCell
        # warp multi layered rnn cell into one cell with dropout
        cells = []
        for _ in range(args.num_layers):   #2
            cell = cell_fn(args.rnn_size)  #128
#            if training and (args.output_keep_prob < 1.0 or args.input_keep_prob < 1.0):  #1.0
#                cell = rnn.DropoutWrapper(cell,
#                                          input_keep_prob=args.input_keep_prob,
#                                          output_keep_prob=args.output_keep_prob)
            cells.append(cell)
            
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)
        # input/target data (int32 since input is char-level)
        self.input_data = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])     #(50, 50)
        self.targets = tf.placeholder(
            tf.int32, [args.batch_size, args.seq_length])     #(50, 50)
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)  #(50, 128)

        # softmax output layer, use softmax to classify
        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w",
                                        [args.rnn_size, 65])  # [128, 65]
            softmax_b = tf.get_variable("softmax_b", [65])  # vocab_size = 65

        # transform input to embedding
        embedding = tf.get_variable("embedding", [65, args.rnn_size])   # [65, 128]
        inputs = tf.nn.embedding_lookup(embedding, self.input_data)     # input_data = (50, 50)
 
        # dropout beta testing: double check which one should affect next line
#        if training and args.output_keep_prob:
#            inputs = tf.nn.dropout(inputs, args.output_keep_prob)

        # unstack the input to fits in rnn model
        inputs = tf.split(inputs, args.seq_length, 1)
        inputs = [tf.squeeze(input_, [1]) for input_ in inputs]  #50 * (50, 128)

        # loop function for rnn_decoder, which take the previous i-th cell's output and generate the (i+1)-th cell's input
        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        # rnn_decoder to generate the ouputs and final state. When we are not training the model, we use the loop function.
        outputs, last_state = legacy_seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if not training else None, scope='rnnlm')
        output = tf.reshape(tf.concat(outputs, 1), [-1, args.rnn_size])

        # output layer
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)

        # loss is calculate by the log loss and taking the average.
        loss = legacy_seq2seq.sequence_loss_by_example(
                [self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])])
        with tf.name_scope('cost'):
            self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()

        # calculate gradients
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                5)
        
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)

        # apply gradient change to the all the trainable variable.
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

        # instrument tensorboard
        tf.summary.histogram('logits', self.logits)
        tf.summary.histogram('loss', loss)
        tf.summary.scalar('train_loss', self.cost)

    def sample(self, sess, chars, vocab, num=200, prime='The ', sampling_type=1):
        
        #初始化初始狀態
        state = sess.run(self.cell.zero_state(1, tf.float32))
        for char in prime[:-1]:
            x = np.zeros((1, 1))
            x[0, 0] = vocab[char]
            feed = {self.input_data: x, self.initial_state: state}
            #feed數據，得到最終狀態值
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        #獲取prime最後的值
        char = prime[-1]
        #連續預測num個字符
        for _ in range(num):
            x = np.zeros((1, 1))
            #獲取char對應的編號
            x[0, 0] = vocab[char]
            #將char和prime之前的狀態作爲feed
            feed = {self.input_data: x, self.initial_state: state}
            #運行流程圖，獲取概率值， 將上一步的state值作爲作爲下一步的feed值
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            
            p = probs[0]
            #獲取最大概率的id
            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if char == ' ':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else:  # sampling_type == 1 default:
                sample = weighted_pick(p)
            #獲取id對應的字符
            pred = chars[sample]
            #將字符加入到ret中
            ret += pred
            #將上一步預測的字符作爲下一步的輸入值
            char = pred
        return ret
