# -*- coding: utf-8 -*-

class Hyperparams:
    '''Hyperparameters Setting'''
    # data_path
    path = r'E:\中科院\国科大\专业课\自然语言处理\homework\homework\作业5-机器翻译\作业5-机器翻译\transformer\data'
    path_log = r'E:\中科院\国科大\专业课\自然语言处理\homework\homework\作业5-机器翻译\作业5-机器翻译\transformer\results'
    source_train = path + '\cn_pre'
    target_train = path + '\en_pre'
    source_test = path + '\cn_test'
    target_test = path + '\en_test'
    loss_path = path_log + '\loss'
    # training
    batch_size = 32  # alias = N
    lr = 0.0001  # learning rate. In paper, learning rate is adjusted to the global step.
    logdir = path_log   # log directory

    # model
    maxlen = 50  # Maximum number of words in a sentence. alias = T.
    # Feel free to increase this if you are ambitious.
    min_cnt = 2  # words whose occurred less than min_cnt are encoded as <UNK>.
    hidden_units = 512  # alias = C
    num_blocks = 6  # number of encoder/decoder blocks
    num_epochs = 200
    num_heads = 8
    dropout_rate = 0.1  # the rate you want to drop out.
    sinusoid = False  # If True, use sinusoid. If false, positional embedding.
