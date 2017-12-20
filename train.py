#!/bin/python3
# -*- coding: utf-8 -*-
"""For training NMT models."""

import pickle
import os
import progressbar
import collections
import time

import tensorflow as tf
import numpy as np


import numpy as np


def param_generator(**kwargs):
    params_sample = dict()
    for param in kwargs:
        options = kwargs[param]
        if 'bool' in options and options['bool']:
            sample = np.random.rand() > 0.5
        elif 'range' in options:
            from_ = options['range'][0]
            to_ = options['range'][1]
            if 'scale' in options and options['scale'] == 'log':
                sample = np.exp(np.random.uniform(np.log(from_), np.log(to_)))
                if 'discrete' in options:
                    sample = int(np.round(sample))
            elif 'discrete' in options and options['discrete']:
                sample = np.random.randint(from_, to_ + 1)
            else:
                sample = np.random.uniform(from_, to_)
        elif 'list' in options and options['list']:
            sample_list = options['list']
            sample_len = len(sample_list)
            sample = sample_list[np.random.choice(sample_len)]
        elif 'ext_list' in options and options['ext_list']:
            sample_count = options['ext_list'][0]
            sample_list = options['ext_list'][1]
            sample_len = len(sample_list)
            sample = []
            for _ in range(sample_count):
                sample.append(sample_list[np.random.choice(sample_len)])
        params_sample[param] = sample
    return params_sample



# Prepare dataset
def prepare_dataset(data_path):
    print('Start dataset preparing.')
    with open(data_path, 'br') as f_des:
        pk_f = pickle.load(f_des)
    bar = progressbar.ProgressBar()
    question_embs = []
    answer_embs = []
    share_embs = []
    bar.init()
    for it in bar(pk_f):
        question_emb,answer_emb = it[2],it[3]
        question_embs.append(question_emb)
        answer_embs.append(answer_emb)
        share_embs.append(question_emb)
        share_embs.append(answer_emb)
    question_embs = [i[0] for i in question_embs]
    answer_embs = [i[0] for i in answer_embs]
    share_embs = [i[0] for i in share_embs]
    return question_embs, answer_embs, share_embs


def data_generator(data, memory_deep):
    start_context_replies = list(data[:memory_deep-1])
    context_size = start_context_replies[0].shape[0] * memory_deep
    context_replies = list(data[memory_deep-1:-1])
    context_deq = collections.deque(start_context_replies, maxlen = memory_deep)

    answers = list(data[memory_deep:])
    while True:
        for context_replie, answer in zip(context_replies,answers):
            context_deq.append(context_replie)
            yield np.array(context_deq).reshape((context_size)), np.array(answer)

def batch_generator(data, memory_deep, batch_size):
    data_gen = data_generator(data, memory_deep)
    while True:
        batch_xs, batch_ys = [], []
        for i, (xs, ys) in enumerate(data_gen):
            if i>= batch_size: break
            batch_xs.append(xs)
            batch_ys.append(ys)
        yield  np.array(batch_xs), np.array(batch_ys)

def split_data(data, rest_part):
    split_point = int(len(data)*rest_part)
    main_data, rest_data = data[:-split_point], data[-split_point:]
    return main_data, rest_data



########### hparams
    # data_dir = '/home/kuznetsov/pica-pica'
    # pickle_file = os.path.join(data_dir, 'heap_last_rep-457000.pkl')
    # batch_size = 256
    # context_maxlen = 10
    # embedding_size = 2048
    # units = 2048
    # layer_num = 2
    # step_per_stat= 1000
    # context_size = context_maxlen * embedding_size
    # test_part = 0.05

class Model():
    def __init__(
            self,
            hparams,
            base_componets,
            # scope=None,
            **kwargs
            ):
        self.hparams = hparams
        self.base_componets = base_componets
        self.hparams = kwargs
        self.dir = self.hparams['dir']
        tf.reset_default_graph()
        scope = None
        self.build_graph(hparams, scope=scope)

    def build_graph(self, hparams, scope=None):
        dtype = tf.float32
        activations = self.hparams['activations']

        self.memory_deep = self.hparams['memory_deep']
        memory_deep = self.memory_deep
        embedding_size = self.hparams['embedding_size']
        self.memory_size = memory_deep * embedding_size
        memory_size = self.memory_size

        bottom_layer_num = self.hparams['bottom_layer_num']
        mid_layer_num = self.hparams['mid_layer_num']
        mid_units = self.hparams['mid_units']

        loss_fn = self.hparams['loss_fn']

        def dense_layers(layer_input,units,layer_num, activations):
            layer_state = layer_input
            for i in range(layer_num):
                if activations[i] == 'tanh':
                    activation = tf.tanh
                elif activations[i] == 'sigm':
                    activation = tf.sigmoid
                elif activations[i] == 'relu':
                    activation = tf.nn.relu
                layer_state = tf.layers.dense(layer_state, units, activation=activation)
                print(layer_state)
            return layer_state
        # with tf.variable_scope(scope or "memory_checker", dtype=dtype):
        self.model_input = tf.placeholder(dtype, [None, memory_size], name = 'model_input')
        self.model_output = tf.placeholder(tf.float32, [None, embedding_size], name = 'model_output')
        model_input = self.model_input
        layer_state = dense_layers(model_input,memory_size,bottom_layer_num, activations)
        activations=activations[bottom_layer_num:]
        layer_state = dense_layers(layer_state,mid_units,mid_layer_num, activations)
        output_state = dense_layers(layer_state,embedding_size,1, ['tanh'])

        if loss_fn == 'mse':
            loss = tf.losses.mean_squared_error(labels=self.model_output, predictions=output_state)
        if loss_fn == 'cosine':
            loss = tf.abs(tf.losses.cosine_distance(labels=self.model_output, predictions=output_state, dim = 1))
        if loss_fn == 'abs':
            loss = tf.losses.absolute_difference(labels=self.model_output, predictions=output_state)
        self.loss = loss
        self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    def train(self):

        #Init prepare variables
        data_dir = '/home/kuznetsov/pica-pica'
        pickle_file = os.path.join(data_dir, 'heap_last_rep-457000.pkl')
        base_dir = '/home/kuznetsov/tmp/'
        log_dir = '/home/kuznetsov/tmp/memory_checker/' + self.dir
        os.mkdir(log_dir)
        params_file = log_dir + '/params.txt'
        with open(params_file, 'wt') as f_desc:
            f_desc.write(self.hparams.__str__())
            print(self.hparams.__str__())
        log_file = log_dir + '/log.txt'
        batch_size = 256
        step_per_stat= 1000
        test_part = 0.1
        _, _, dataset = prepare_dataset(pickle_file)

        train_dataset,test_dataset  = split_data(dataset, test_part)
        epoch_steps = len(train_dataset)//batch_size
        print('epoch_steps = {}'.format(epoch_steps))
        # epoch_steps = 20
        # raise

        sess = tf.InteractiveSession()
        tf.global_variables_initializer().run()
        # Train
        print('Start train')
        train_batch_generator = batch_generator(train_dataset, self.memory_deep, batch_size)
        test_batch_generator = batch_generator(test_dataset, self.memory_deep, batch_size)
        last_time = time.time()

        epoch_num = -1
        with open(log_file, 'wt') as f_desc:
            for i, (batch_xs, batch_ys) in enumerate(batch_generator(train_dataset, self.memory_deep, batch_size)):
               sess.run(self.train_step, feed_dict={self.model_input: batch_xs, self.model_output: batch_ys})
               if i%epoch_steps == 0:
                   epoch_num += 1
                   if epoch_num > self.hparams['epochs'] :
                       f_desc.write('end')
                       print('end')
                       break
               if i%step_per_stat == 0:
                   batch_xs, batch_ys = next(test_batch_generator)
                   test_loss = sess.run(self.loss, feed_dict={self.model_input: batch_xs, self.model_output: batch_ys})
                   f_desc.write('time = {:.2f}, step = {}, epoch = {}, loss = {}\n'.format(time.time() - last_time,i,i//epoch_steps,test_loss))
                   print('time = {:.2f}, step = {}, epoch = {}, loss = {}\n'.format(time.time() - last_time,i,i//epoch_steps,test_loss))
                   last_time = time.time()



def main():
    params = param_generator(activations={'ext_list': [20,['tanh','sigm','relu']]},
     memory_deep={'range': [10, 10], 'discrete': True},
     embedding_size={'range': [2048, 2048], 'discrete': True},
     bottom_layer_num={'range': [0, 3], 'discrete': True},
     mid_layer_num={'range': [0, 3], 'discrete': True},
     mid_units={'range': [2048, 8192], 'discrete': True},
     loss_fn={'list': ['mse','cosine','abs']},
     epochs={'range': [10, 10], 'discrete': True, 'scale': 'log'})
    path = ''
    for key in sorted(params.keys()):
        if key == 'activations': continue
        path += str(params[key])
    params['dir'] = path
    print(params.__str__())

    model = Model(1, 2,**params)
    model.train()


if __name__ == '__main__':
    main()
#     # Test trained model
#     correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     print(sess.run(accuracy, feed_dict={x: mnist.test.images,
#                                       y_: mnist.test.labels}))
#
# if __name__ == '__main__':
#   parser = argparse.ArgumentParser()
#   parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
#                       help='Directory for storing input data')
#   FLAGS, unparsed = parser.parse_known_args()
#   tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
