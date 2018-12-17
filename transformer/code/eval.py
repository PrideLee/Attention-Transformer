# -*- coding: utf-8 -*-

from __future__ import print_function
import codecs
import os
import tensorflow as tf
import numpy as np
import re
from hyperparams import Hyperparams as hp
from data_load import load_test_data, load_cn_vocab, load_en_vocab
from train import Graph
from nltk.translate.bleu_score import corpus_bleu


def eval():
    # Load graph
    g = Graph(is_training=False)
    print("Graph loaded")

    # Load data
    X, Sources, Targets = load_test_data()
    cn2idx, idx2cn = load_cn_vocab()
    en2idx, idx2en = load_en_vocab()

    #     X, Sources, Targets = X[:33], Sources[:33], Targets[:33]

    # Start session         
    with g.graph.as_default():
        # A training helper that checkpoints models and computes summaries.
        sv = tf.train.Supervisor()
        with sv.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            ## Restore parameters
            sv.saver.restore(sess, tf.train.latest_checkpoint(hp.logdir))
            print("Restored!")

            ## Get model name
            mname = open(hp.logdir + '\checkpoint', 'r').read().split('"')[1]  # model name
            mname = re.findall(r'results(.*)', mname)[0]

            ## Inference
            result_trans = hp.logdir + '\\translation'
            if not os.path.exists(result_trans):
                os.mkdir(result_trans)
            with codecs.open(result_trans + mname, "w", "utf-8") as fout:
                list_of_refs, hypotheses = [], []
                if len(X) // hp.batch_size == 0:
                    iteration = 1
                else:
                    iteration =len(X) // hp.batch_size
                for i in range(iteration):
                    if iteration == 1:
                        x = X
                        sources = Sources
                        targets = Targets
                        preds = np.zeros((len(Sources), hp.maxlen), np.int32)
                    else:
                        ### Get mini-batches
                        x = X[i * hp.batch_size: (i + 1) * hp.batch_size]
                        sources = Sources[i * hp.batch_size: (i + 1) * hp.batch_size]
                        targets = Targets[i * hp.batch_size: (i + 1) * hp.batch_size]
                        ### Autoregressive inference
                        preds = np.zeros((hp.batch_size, hp.maxlen), np.int32)
                    for j in range(hp.maxlen):
                        _preds = sess.run(g.preds, {g.x: x, g.y: preds})
                        preds[:, j] = _preds[:, j]
                    ### Write to file
                    for source, target, pred in zip(sources, targets, preds):  # sentence-wise
                        got = " ".join(idx2en[idx] for idx in pred).split("</S>")[0].strip()
                        fout.write("- source: " + source + "\n")
                        fout.write("- expected: " + target + "\n")
                        fout.write("- got: " + got + "\n\n")
                        fout.flush()
                        # bleu score
                        ref = target.split()
                        hypothesis = got.split()
                        if len(ref) > 3 and len(hypothesis) > 3:
                            list_of_refs.append([ref])
                            hypotheses.append(hypothesis)

                ## Calculate bleu score
                score = corpus_bleu(list_of_refs, hypotheses)
                fout.write("Bleu Score = " + str(100 * score))


if __name__ == '__main__':
    eval()
    print("Done")
