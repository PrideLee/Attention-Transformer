import codecs


raw_path = r'E:\中科院\国科大\专业课\自然语言处理\homework\homework\作业5-机器翻译\作业5-机器翻译\data'
en_raw_data_path = raw_path + '\en_pre.txt'
en_vocab_path = raw_path + '\en.vocab'
en_train_path = raw_path + '\en.train'
cn_raw_data_path = raw_path + '\cn_pre.txt'
cn_vocab_path = raw_path + '\cn.vocab'
cn_train_path = raw_path + '\cn.train'


## Parameters setting
MODE = "TRANSLATE_EN"    # 将MODE设置为"PTB_TRAIN", "PTB_VALID", "PTB_TEST", "TRANSLATE_EN", "TRANSLATE_ZH"之一。
# MODE = "TRANSLATE_ZH"
if MODE == "PTB_TRAIN":        # PTB训练数据
    RAW_DATA = "///"  # 训练集数据文件
    VOCAB = "///"                                 # 词汇表文件
    OUTPUT_DATA = "///"                           # 将单词替换为单词编号后的输出文件
elif MODE == "PTB_VALID":      # PTB验证数据
    RAW_DATA = "///"
    VOCAB = "///"
    OUTPUT_DATA = "///"
elif MODE == "PTB_TEST":       # PTB测试数据
    RAW_DATA = "///"
    VOCAB = "///"
    OUTPUT_DATA = "///"
elif MODE == "TRANSLATE_ZH":   # 中文翻译数据
    RAW_DATA = cn_raw_data_path
    VOCAB = cn_vocab_path
    OUTPUT_DATA = cn_train_path
elif MODE == "TRANSLATE_EN":   # 英文翻译数据
    RAW_DATA = en_raw_data_path
    VOCAB = en_vocab_path
    OUTPUT_DATA = en_train_path


# 读取词汇表，并建立词汇到单词编号的映射。
with codecs.open(VOCAB, "r", "utf-8") as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

# 如果出现了不在词汇表内的低频词，则替换为"unk"。
def get_id(word):
    return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]


fin = codecs.open(RAW_DATA, "r", "utf-8")
fout = codecs.open(OUTPUT_DATA, 'w', 'utf-8')
for line in fin:
    words = line.strip().split() + ["<eos>"]  # 读取单词并添加<eos>结束符
    # 将每个单词替换为词汇表中的编号
    out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'
    fout.write(out_line)
fin.close()
fout.close()