# -*- coding: utf-8 -*-

from __future__ import print_function
from hyperparams import Hyperparams as hp
import tensorflow as tf
import numpy as np
import codecs
import regex


def load_cn_vocab():
    path_vocab = r'E:\中科院\国科大\专业课\自然语言处理\homework\homework\作业5-机器翻译\作业5-机器翻译\transformer\data\cn.vocab.tsv'
    vocab = [line.split()[0] for line in codecs.open(path_vocab, 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def load_en_vocab():
    path_vocab = r'E:\中科院\国科大\专业课\自然语言处理\homework\homework\作业5-机器翻译\作业5-机器翻译\transformer\data\en.vocab.tsv'
    vocab = [line.split()[0] for line in codecs.open(path_vocab, 'r', 'utf-8').read().splitlines() if
             int(line.split()[1]) >= hp.min_cnt]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return word2idx, idx2word


def create_data(source_sents, target_sents):
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()

    # Index
    x_list, y_list, Sources, Targets = [], [], [], []
    for source_sent, target_sent in zip(source_sents, target_sents):
        x = [cn2idx.get(word, 1) for word in (source_sent + u" </S>").split()]  # 1: OOV, </S>: End of Text
        y = [en2idx.get(word, 1) for word in (target_sent + u" </S>").split()]
        if max(len(x), len(y)) <= hp.maxlen:
            x_list.append(np.array(x))
            y_list.append(np.array(y))
            Sources.append(source_sent)
            Targets.append(target_sent)

    # Pad      
    X = np.zeros([len(x_list), hp.maxlen], np.int32)
    Y = np.zeros([len(y_list), hp.maxlen], np.int32)
    for i, (x, y) in enumerate(zip(x_list, y_list)):
        X[i] = np.lib.pad(x, [0, hp.maxlen - len(x)], 'constant', constant_values=(0, 0))
        Y[i] = np.lib.pad(y, [0, hp.maxlen - len(y)], 'constant', constant_values=(0, 0))

    return X, Y, Sources, Targets


def load_train_data():
    cn_sents = [regex.sub("\\pP", "", line) for line in codecs.open(hp.source_train, 'r', 'utf-8').read().split("\n") if
                line and line[0] != "<"]
    en_sents = [regex.sub("\\pP", "", line) for line in codecs.open(hp.target_train, 'r', 'utf-8').read().split("\n") if
                line and line[0] != "<"]
    X, Y, Sources, Targets = create_data(cn_sents, en_sents)
    return X, Y


def load_test_data():
    def _refine(line):
        line = regex.sub("\\pP", "", line)
        line = regex.sub("\\pP", "", line)
        return line.strip()

    cn_sents = [_refine(line) for line in codecs.open(hp.source_test, 'r', 'utf-8').read().split("\n")]
    en_sents = [_refine(line) for line in codecs.open(hp.target_test, 'r', 'utf-8').read().split("\n")]
    X, Y, Sources, Targets = create_data(cn_sents, en_sents)

    return X, Sources, Targets  # (1064, 150)


def get_batch_data():
    # Load data
    X, Y = load_train_data()

    # calc total batch count
    num_batch = len(X) // hp.batch_size

    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)

    # Create Queues, Produces a slice of each `Tensor` in `tensor_list`.
    input_queues = tf.train.slice_input_producer([X, Y])

    # create batch queues by randomly shuffling tensors.
    x, y = tf.train.shuffle_batch(
        input_queues,  # The list or dictionary of tensors to enqueue.
        num_threads=8,  # The number of threads enqueuing `tensor_list`.
        batch_size=hp.batch_size,  # The new batch size pulled from the queue.
        # An integer. The maximum number of elements in the queue.
        capacity=hp.batch_size * 64,
        # Minimum number elements in the queue after a dequeue, used to ensure a level of mixing of elements.
        min_after_dequeue=hp.batch_size * 32,
        # Optional) Boolean. If `True`, allow the final batch to be smaller if there are insufficient items left in the queue.
        allow_smaller_final_batch=False)

    return x, y, num_batch  # (N, T), (N, T), ()
