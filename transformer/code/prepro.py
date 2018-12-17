# -*- coding: utf-8 -*-
from __future__ import print_function
from hyperparams import Hyperparams as hp
import codecs
import os
import regex
from collections import Counter


def make_vocab(fpath, pro_path, fname):
    '''Constructs vocabulary.
    
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `preprocessed/fname`
    '''
    text = codecs.open(fpath, 'r', 'utf-8').read()
    # replace "[^\s\p{Latin}']"  as "".
    text = regex.sub("\\pP", "", text)
    words = text.split()
    word2cnt = Counter(words)
    # Creating a folder to save the results.
    if not os.path.exists(pro_path):
        os.mkdir(pro_path)
    with codecs.open(pro_path + fname, 'w', 'utf-8') as fout:
        fout.write(
            "{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))


if __name__ == '__main__':
    make_vocab(hp.source_train, hp.path, "\cn.vocab.tsv")
    make_vocab(hp.target_train, hp.path, "\en.vocab.tsv")
    print("Done")
