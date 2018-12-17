import codecs
import collections
from operator import itemgetter


raw_path = r'E:\中科院\国科大\专业课\自然语言处理\homework\homework\作业5-机器翻译\作业5-机器翻译\data'
en_data_path = raw_path + '\en_pre.txt'
cn_data_path = raw_path + '\cn_pre.txt'
en_vocab_path = raw_path + '\en.vocab'
cn_vocab_path = raw_path + '\cn.vocab'
## Setting parameters

# MODE = "TRANSLATE_EN"    # 将MODE设置为"PTB", "TRANSLATE_EN", "TRANSLATE_ZH"之一。
MODE = "TRANSLATE_ZH"
if MODE == "PTB":             # PTB数据处理
    RAW_DATA = "///"  # 训练集数据文件
    VOCAB_OUTPUT = "ptb.vocab"                         # 输出的词汇表文件
elif MODE == "TRANSLATE_ZH":  # 翻译语料的中文部分
    RAW_DATA = cn_data_path
    VOCAB_OUTPUT = cn_vocab_path
    VOCAB_SIZE = 4000
elif MODE == "TRANSLATE_EN":  # 翻译语料的英文部分
    RAW_DATA = en_data_path
    VOCAB_OUTPUT = en_vocab_path
    VOCAB_SIZE = 10000


## Sorting the word by the frequent from large to small
counter = collections.Counter()  # 统计单词出现词频
with codecs.open(RAW_DATA, "r", "utf-8") as f:
    for line in f:
        for word in line.strip().split():  # 分词
            counter[word] += 1

# 按词频顺序对单词进行排序，降序排列。
sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

## Inserting the identifier
if MODE == "PTB":
    # 我们需要在文本换行处加入句子结束符"<eos>"，这里预先将其加入词汇表。
    sorted_words = ["<eos>"] + sorted_words
elif MODE in ["TRANSLATE_EN", "TRANSLATE_ZH"]:
    # "<eos>"句子结束符，"<unk>"低频词汇替换符,"<sos>"句子起始符
    sorted_words = ["<unk>", "<sos>", "<eos>"] + sorted_words
    if len(sorted_words) > VOCAB_SIZE:
        sorted_words = sorted_words[:VOCAB_SIZE]

## Saving the word table
with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + "\n")