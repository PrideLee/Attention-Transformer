import tensorflow as tf
import codecs
import sys

## parameter setting
path_model = r'E:\中科院\国科大\专业课\自然语言处理\homework\homework\作业5-机器翻译\作业5-机器翻译\result'
path_data = r'E:\中科院\国科大\专业课\自然语言处理\homework\homework\作业5-机器翻译\作业5-机器翻译\data'
test_en = path_data + '\en_test'
test_cn = path_data + '\cn_test'
translation = path_model + '\\translation'
# 读取checkpoint的路径。13600表示是训练程序在第13600步保存的checkpoint。
CHECKPOINT_PATH = path_model + '\\attention_ckpt-13600'

# 模型参数。必须与训练时的模型参数保持一致。
HIDDEN_SIZE = 1024  # LSTM的隐藏层规模。
DECODER_LAYERS = 2  # 解码器中LSTM结构的层数。
SRC_VOCAB_SIZE = 4000  # 源语言词汇表大小。
TRG_VOCAB_SIZE = 10000  # 目标语言词汇表大小。
SHARE_EMB_AND_SOFTMAX = True  # 在Softmax层和词向量层之间共享参数。

# 词汇表文件
SRC_VOCAB = path_data + '\cn.vocab'
TRG_VOCAB = path_data + '\en.vocab'

# 词汇表中<sos>和<eos>的ID。在解码过程中需要用<sos>作为第一步的输入，并将检查
# 是否是<eos>，因此需要知道这两个符号的ID。
SOS_ID = 1
EOS_ID = 2


## Decoding
# 定义NMTModel类来描述模型。
class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量。
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构。
        self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
             for _ in range(DECODER_LAYERS)])

        # 为源语言和目标语言分别定义词向量。
        self.src_embedding = tf.get_variable("src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        self.trg_embedding = tf.get_variable("trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # 定义softmax层的变量
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        else:
            self.softmax_weight = tf.get_variable("weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable("softmax_bias", [TRG_VOCAB_SIZE])

    def inference(self, src_input):
        # 虽然输入只有一个句子，但因为dynamic_rnn要求输入是batch的形式，因此这里将输入句子整理为大小为1的batch。
        src_size = tf.convert_to_tensor([len(src_input)], dtype=tf.int32)
        src_input = tf.convert_to_tensor([src_input], dtype=tf.int32)
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)

        with tf.variable_scope("encoder"):
            # 使用bidirectional_dynamic_rnn构造编码器。这一步与训练时相同。
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(self.enc_cell_fw, self.enc_cell_bw, src_emb,
                                                                     src_size, dtype=tf.float32)
            # 将两个LSTM的输出拼接为一个张量。
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)

        with tf.variable_scope("decoder"):
            # 定义解码器使用的注意力机制。
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(HIDDEN_SIZE, enc_outputs,
                                                                       memory_sequence_length=src_size)

            # 将解码器的循环神经网络self.dec_cell和注意力一起封装成更高层的循环神经网络。
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell, attention_mechanism,
                                                                 attention_layer_size=HIDDEN_SIZE)

        # 设置解码的最大步数。这是为了避免在极端情况出现无限循环的问题。
        MAX_DEC_LEN = 100
        # A context manager for defining ops that creates variables (layers).
        with tf.variable_scope("decoder/rnn/attention_wrapper"):
            # 使用一个变长的TensorArray来存储生成的句子。
            init_array = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True, clear_after_read=False)
            # 填入第一个单词<sos>作为解码器的输入。
            init_array = init_array.write(0, SOS_ID)
            # 调用attention_cell.zero_state构建初始的循环状态。循环状态包含循环神经网络的隐藏状态，保存生成句子的TensorArray，以及记录解码步数的一个整数step。
            init_loop_var = (attention_cell.zero_state(batch_size=1, dtype=tf.float32), init_array, 0)

            # tf.while_loop的循环条件：循环直到解码器输出<eos>，或者达到最大步数为止。
            def continue_loop_condition(state, trg_ids, step):
                return tf.reduce_all(
                    tf.logical_and(tf.not_equal(trg_ids.read(step), EOS_ID), tf.less(step, MAX_DEC_LEN - 1)))

            def loop_body(state, trg_ids, step):
                # 读取最后一步输出的单词，并读取其词向量。
                trg_input = [trg_ids.read(step)]
                trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)
                # 调用attention_cell向前计算一步。
                dec_outputs, next_state = attention_cell.call(state=state, inputs=trg_emb)
                # 计算每个可能的输出单词对应的logit，并选取logit值最大的单词作这一步的而输出。
                output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
                logits = (tf.matmul(output, self.softmax_weight) + self.softmax_bias)
                next_id = tf.argmax(logits, axis=1, output_type=tf.int32)
                # 将这一步输出的单词写入循环状态的trg_ids中。
                trg_ids = trg_ids.write(step + 1, next_id[0])
                return next_state, trg_ids, step + 1

            # 执行tf.while_loop，返回最终状态。
            state, trg_ids, step = tf.while_loop(continue_loop_condition, loop_body, init_loop_var)
            return trg_ids.stack()


## Translation

def main():
    # # 根据中文词汇表，将测试句子转为词ID。
    with codecs.open(SRC_VOCAB, "r", "utf-8") as f_vocab:
        src_vocab = [w.strip() for w in f_vocab.readlines()]
        src_id_dict = dict((src_vocab[x], x) for x in range(len(src_vocab)))
    output_ids_list = []
    # 测试句子。
    with codecs.open(test_cn, 'r', 'utf-8') as file_cn:
        for line in file_cn:
            test_en_text = line + ' ' + '<eos>'
            print(test_en_text)
            test_cn_ids = [(src_id_dict[token] if token in src_id_dict else src_id_dict['<unk>']) for token in
                           test_en_text.split()]
            # print(test_cn_ids)
            tf.reset_default_graph()
            # # 定义训练用的循环神经网络模型。
            with tf.variable_scope("nmt_model", reuse=None):
                model = NMTModel()
            output_op = model.inference(test_cn_ids)

            ## 建立解码所需的计算图。
            sess = tf.Session()
            saver = tf.train.Saver()
            saver.restore(sess, CHECKPOINT_PATH)

            # 读取翻译结果。
            output_ids = sess.run(output_op)
            output_ids_list.append(output_ids)
            sess.close()
    # 根据英文词汇表，将翻译结果转换为英文。
    with codecs.open(TRG_VOCAB, "r", "utf-8") as f_vocab:
        trg_vocab = [w.strip() for w in f_vocab.readlines()]
    predict = []
    for output in output_ids_list:
        output_text = ' '.join([trg_vocab[x] for x in output])
        # 输出翻译结果。
        temp_out = output_text.encode('utf-8').decode(sys.stdout.encoding)
        predict.append(temp_out + '\n')
        # print(temp_out)

    predict = predict[:len(predict) - 1]
    with codecs.open(translation, 'w', 'utf-8') as final_save:
        with codecs.open(test_cn, 'r', 'utf_8') as source_out:
            with codecs.open(test_en, 'r', 'utf-8') as target_out:
                for u in zip(predict, source_out, target_out):
                    final_save.write('source: ' + u[1] + '\n')
                    final_save.write('target:' + u[2] + '\n')
                    final_save.write('translate: ' + u[0] + '\n')


if __name__ == "__main__":
    main()
