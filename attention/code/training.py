import tensorflow as tf
import os


## The training data has already transformered into coded format.
path_data = r'E:\中科院\国科大\专业课\自然语言处理\homework\homework\作业5-机器翻译\作业5-机器翻译\data'
path_results = r'E:\中科院\国科大\专业课\自然语言处理\homework\homework\作业5-机器翻译\作业5-机器翻译\result'
train_en_path = path_data + '\en.train'
train_cn_path = path_data + '\cn.train'
checkpoint_path = path_results + '\\attention_ckpt'
path_cost = path_results + '\cost'
## Parameters setting
SRC_TRAIN_DATA = train_cn_path  # 源语言输入文件路径（中文）。
TRG_TRAIN_DATA = train_en_path  # 目标语言输入文件路径（英文）。
CHECKPOINT_PATH = checkpoint_path  # checkpoint保存路径。
HIDDEN_SIZE = 1024  # LSTM的隐藏层规模。
DECODER_LAYERS = 2  # 解码器中LSTM结构的层数。其中编码器固定使用单层的双向LSTM。
SRC_VOCAB_SIZE = 4000  # 源语言词汇表大小（中文）。
TRG_VOCAB_SIZE = 10000  # 目标语言词汇表大小（英文）。
BATCH_SIZE = 100  # 训练数据batch的大小。
NUM_EPOCH = 200  # 使用训练数据的轮数。
KEEP_PROB = 0.8  # 节点不被dropout的概率。
MAX_GRAD_NORM = 5  # 用于控制梯度膨胀的梯度大小上限。
SHARE_EMB_AND_SOFTMAX = True  # 在Softmax层和词向量层之间共享参数。
MAX_LEN = 50  # 限定句子的最大单词数量。
SOS_ID = 1  # 目标语言词汇表中<sos>的ID。


## Reading the train data and creating the Dataset
# 使用Dataset从一个文件中读取一个语言的数据。
# 数据的格式为每行一句话，单词已经转化为单词编号。
def MakeDataset(file_path):
    # tf.data.TextLineDataset()函数输出的每一个元素对应输入文件的一行
    dataset = tf.data.TextLineDataset(file_path)
    # 根据空格将单词编号切分开并放入一个一维向量。
    dataset = dataset.map(lambda string: tf.string_split([string]).values)
    # 将字符串形式的单词编号转化为整数。
    dataset = dataset.map(lambda string: tf.string_to_number(string, tf.int32))
    # 统计每个句子的单词数量，并与句子内容一起放入Dataset中，x为句子中个单词的编号序列，tf.size(x)个句子单词数。
    dataset = dataset.map(lambda x: (x, tf.size(x)))
    return dataset


# 从源语言文件src_path和目标语言文件trg_path中分别读取数据，并进行填充和batching操作。
def MakeSrcTrgDataset(src_path, trg_path, batch_size):
    # 首先分别读取源语言数据和目标语言数据（个句子中单词的编号及个句子中单词的数量）。
    src_data = MakeDataset(src_path)
    trg_data = MakeDataset(trg_path)
    # 通过zip操作将两个Dataset合并为一个Dataset即ds（ds为一个迭代器）。现在每个Dataset中每一项数据由4个张量组成：
    #   ds[0][0]是源句子
    #   ds[0][1]是源句子长度
    #   ds[1][0]是目标句子
    #   ds[1][1]是目标句子长度
    dataset = tf.data.Dataset.zip((src_data, trg_data))

    def FilterLength(src_tuple, trg_tuple):
        # src_tuple, trg_tuple = (x, size(x))
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        # tf.logical_and()函数返回True,False，如果1<src_len<MAX_LEN，则为True，否则为False.
        src_len_ok = tf.logical_and(tf.greater(src_len, 1), tf.less_equal(src_len, MAX_LEN))
        trg_len_ok = tf.logical_and(tf.greater(trg_len, 1), tf.less_equal(trg_len, MAX_LEN))
        # 当输入、输出句子长度均满足要求为True，否则为False
        return tf.logical_and(src_len_ok, trg_len_ok)

    # 调用FilterLength函数（），删除内容为空（只包含<EOS>）的句子和长度过长的句子。
    dataset = dataset.filter(FilterLength)

    # 解码器需要两种格式的目标句子：
    # 1.解码器的输入(trg_input)，形式如同"<sos> X Y Z"
    # 2.解码器的目标输出(trg_label)，形式如同"X Y Z <eos>"
    # 上面从文件中读到的目标句子是"X Y Z <eos>"的形式，我们需要从中生成"<sos> X Y Z"形式并加入到Dataset中。
    def MakeTrgInput(src_tuple, trg_tuple):
        ((src_input, src_len), (trg_label, trg_len)) = (src_tuple, trg_tuple)
        # 插入"<eos>"，输出trg_input
        trg_input = tf.concat([[SOS_ID], trg_label[:-1]], axis=0)
        return ((src_input, src_len), (trg_input, trg_label, trg_len))

    dataset = dataset.map(MakeTrgInput)

    # 随机打乱训练数据。
    dataset = dataset.shuffle(10000)

    # 规定填充后输出的数据维度。
    padded_shapes = (
        (tf.TensorShape([None]),  # 源句子是长度未知的向量
         tf.TensorShape([])),  # 源句子长度是单个数字
        (tf.TensorShape([None]),  # 目标句子（解码器输入）是长度未知的向量
         tf.TensorShape([None]),  # 目标句子（解码器目标输出）是长度未知的向量
         tf.TensorShape([])))  # 目标句子长度是单个数字
    # 调用padded_batch方法进行batching操作。
    batched_dataset = dataset.padded_batch(batch_size, padded_shapes)
    return batched_dataset


## Translation model
# 定义NMTModel类来描述模型。
class NMTModel(object):
    # 在模型的初始化函数中定义模型要用到的变量。
    def __init__(self):
        # 定义编码器和解码器所使用的LSTM结构。
        # Encoder
        # 前向LSTM
        self.enc_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        # 后向LSTM
        self.enc_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE)
        # Decoder stacking 2 LSTM layers
        self.dec_cell = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(DECODER_LAYERS)])

        # 为源语言和目标语言分别定义词向量。
        # input embedding
        self.src_embedding = tf.get_variable("src_emb", [SRC_VOCAB_SIZE, HIDDEN_SIZE])
        # output embedding
        self.trg_embedding = tf.get_variable("trg_emb", [TRG_VOCAB_SIZE, HIDDEN_SIZE])

        # 定义softmax层的变量
        # 在Softmax层和目标（输出）词向量层之间共享参数
        if SHARE_EMB_AND_SOFTMAX:
            self.softmax_weight = tf.transpose(self.trg_embedding)
        # 在Softmax层和词向量层之间不共享参数，Softmax层参数单独训练
        else:
            self.softmax_weight = tf.get_variable("weight", [HIDDEN_SIZE, TRG_VOCAB_SIZE])
        self.softmax_bias = tf.get_variable("softmax_bias", [TRG_VOCAB_SIZE])

    # 在forward函数中定义模型的前向计算图。
    # src_input, src_size, trg_input, trg_label, trg_size分别是上面MakeSrcTrgDataset函数产生的五种张量。
    def forward(self, src_input, src_size, trg_input, trg_label, trg_size):
        batch_size = tf.shape(src_input)[0]

        # 将输入和输出单词编号转为词向量。
        src_emb = tf.nn.embedding_lookup(self.src_embedding, src_input)
        trg_emb = tf.nn.embedding_lookup(self.trg_embedding, trg_input)

        # 在词向量上进行dropout。
        src_emb = tf.nn.dropout(src_emb, KEEP_PROB)
        trg_emb = tf.nn.dropout(trg_emb, KEEP_PROB)

        # 使用dynamic_rnn构造编码器。
        # 编码器读取源句子每个位置的词向量，输出最后一步的隐藏状态enc_state。
        # 因为编码器是一个双层LSTM，因此enc_state是一个包含两个LSTMStateTuple类张量的tuple，每个LSTMStateTuple对应编码器中的一层。
        # 张量的维度是 [batch_size, HIDDEN_SIZE]。
        # enc_outputs是顶层LSTM在每一步的输出，它的维度是[batch_size, max_time, HIDDEN_SIZE]。Seq2Seq模型中不需要用到enc_output。
        with tf.variable_scope("encoder"):
            # 构造编码器时，使用bidirectional_dynamic_rnn构造双向循环网络。
            # 双向循环网络的顶层输出enc_outputs是一个包含两个张量的tuple，每个张量的维度都是[batch_size, max_time, HIDDEN_SIZE]，代表两个LSTM在每一步的输出。
            # Creates a dynamic version of bidirectional recurrent neural network.
            # Return: A tuple (outputs, output_states) where: * outputs: A tuple (output_fw, output_bw) containing the forward and the backward rnn output Tensor
            enc_outputs, enc_state = tf.nn.bidirectional_dynamic_rnn(self.enc_cell_fw, self.enc_cell_bw, src_emb,
                                                                     src_size, dtype=tf.float32)
            # 将两个LSTM的输出拼接为一个张量。
            enc_outputs = tf.concat([enc_outputs[0], enc_outputs[1]], -1)

        with tf.variable_scope("decoder"):
            # 选择注意力权重的计算模型。BahdanauAttention是使用一个隐藏层的前馈神经网络。
            # memory_sequence_length是一个维度为[batch_size]的张量，代表batch中每个句子的长度，Attention需要根据这个信息把填充位置的注意力权重设置为0。
            attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(HIDDEN_SIZE, enc_outputs,
                                                                       memory_sequence_length=src_size)
            # 将解码器的循环神经网络self.dec_cell和注意力一起封装成更高层的循环神经网络。
            attention_cell = tf.contrib.seq2seq.AttentionWrapper(self.dec_cell, attention_mechanism,
                                                                 attention_layer_size=HIDDEN_SIZE)
            # 使用attention_cell和dynamic_rnn构造编码器。
            # 这里没有指定init_state，也就是没有使用编码器的输出来初始化输入，而完全依赖注意力作为信息来源。
            # Creates a recurrent neural network specified by RNNCell cell=attention_cell.
            dec_outputs, _ = tf.nn.dynamic_rnn(attention_cell, trg_emb, trg_size, dtype=tf.float32)

        # 计算解码器每一步的log perplexity。
        output = tf.reshape(dec_outputs, [-1, HIDDEN_SIZE])
        logits = tf.matmul(output, self.softmax_weight) + self.softmax_bias
        # Computes sparse softmax cross entropy between logits and labels.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(trg_label, [-1]), logits=logits)

        # 在计算平均损失时，需要将填充位置的权重设置为0，以避免无效位置的预测干扰模型的训练。
        label_weights = tf.sequence_mask(trg_size, maxlen=tf.shape(trg_label)[1], dtype=tf.float32)
        label_weights = tf.reshape(label_weights, [-1])
        # Computes the sum of elements across dimensions of a tensor.
        # sum all.
        cost = tf.reduce_sum(loss * label_weights)
        cost_per_token = cost / tf.reduce_sum(label_weights)

        # 定义反向传播操作。
        # Returns all variables created with `trainable=True`.
        trainable_variables = tf.trainable_variables()

        # 控制梯度大小，定义优化方法和训练步骤。
        grads = tf.gradients(cost / tf.to_float(batch_size), trainable_variables)
        # Gradient Clipping让权重的更新限制在一个合适的范围
        grads, _ = tf.clip_by_global_norm(grads, MAX_GRAD_NORM)
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0)
        train_op = optimizer.apply_gradients(zip(grads, trainable_variables))
        return cost_per_token, train_op


## main()
# 使用给定的模型model上训练一个epoch，并返回全局步数。
# 每训练200步便保存一个checkpoint。
def run_epoch(session, cost_op, train_op, saver, step, result_save):
    # 训练一个epoch。
    # 重复训练步骤直至遍历完Dataset中所有数据。
    while True:
        try:
            # 运行train_op并计算损失值。训练数据在main()函数中以Dataset方式提供。
            cost, _ = session.run([cost_op, train_op])
            if step % 10 == 0:
                temp_cost = "After %d steps, per token cost is %.3f" % (step, cost)
                print(temp_cost)
                result_save.write(temp_cost + '\n')
            # 每200步保存一个checkpoint。
            if step % 200 == 0:
                saver.save(session, CHECKPOINT_PATH, global_step=step)
            step += 1
        # Raised when an operation iterates past the valid input range.
        except tf.errors.OutOfRangeError:
            break
    return step


def main():
    # 定义初始化函数。
    initializer = tf.random_uniform_initializer(-0.05, 0.05)

    # 定义训练用的循环神经网络模型。
    with tf.variable_scope("nmt_model", reuse=None, initializer=initializer):
        # 训练模型
        train_model = NMTModel()

    # 定义输入数据。
    # batch & padding (delete the sentence which length smaller than 1 or larger than MAX_LEN)
    data = MakeSrcTrgDataset(SRC_TRAIN_DATA, TRG_TRAIN_DATA, BATCH_SIZE)
    # Creates an `Iterator` for enumerating the elements of this dataset.
    iterator = data.make_initializable_iterator()
    # Reading the tensor from iterator orderly.
    (src, src_size), (trg_input, trg_label, trg_size) = iterator.get_next()

    # 定义前向计算图。输入数据以张量形式提供给forward函数。
    cost_op, train_op = train_model.forward(src, src_size, trg_input, trg_label, trg_size)
    if not os.path.exists(path_results):
        os.mkdir(path_results)
    # 训练模型。
    # 定义saver保存checkpoint。
    saver = tf.train.Saver()
    step = 0
    with tf.Session() as sess:
        # 参数初始化
        tf.global_variables_initializer().run()
        with open(path_cost, 'a+') as result_file:
            for i in range(NUM_EPOCH):
                it = "In iteration: %d" % (i + 1)
                print(it)
                result_file.write(it + '\n')
                # A `tf.Operation` that should be run to initialize this iterator.
                sess.run(iterator.initializer)
                step = run_epoch(sess, cost_op, train_op, saver, step, result_file)


if __name__ == "__main__":
    main()
