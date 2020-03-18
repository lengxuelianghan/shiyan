import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")

"""
构建模型，模型的架构如下：
1，利用Bi-LSTM获得上下文的信息
2，将Bi-LSTM获得的隐层输出和词向量拼接[fwOutput;wordEmbedding;bwOutput]
3，将2所得的词表示映射到低维
4，hidden_size上每个位置的值都取时间步上最大的值，类似于max-pool
5，softmax分类
"""
class RCNN(object):
    """
    RCNN 用于文本分类
    """
    def __init__(self, config, wordEmbedding):

        # 定义模型的输入
        self.input_x = tf.placeholder(tf.int32, [config.batchSize, config.seq_len], name="inputX")
        self.input_y = tf.placeholder(tf.int32, [config.batchSize, config.numClasses], name="inputY")

        self.dropoutKeepProb = tf.placeholder(tf.float32, name="dropoutKeepProb")

        # 定义l2损失
        l2Loss = tf.constant(0.0)

        # 词嵌入层
        with tf.name_scope("embedding"):

            # 利用预训练的词向量初始化词嵌入矩阵
            self.W = tf.Variable(tf.cast(wordEmbedding, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            self.embeddedWords = tf.nn.embedding_lookup(self.W, self.input_x)
            # 复制一份embedding input
            self.embeddedWords_ = self.embeddedWords

        with tf.name_scope("Bi-LSTM"):
            for idx, hiddenSize in enumerate(config.model.hiddenSizes):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)
                    # 定义反向LSTM结构
                    lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hiddenSize, state_is_tuple=True),
                        output_keep_prob=self.dropoutKeepProb)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs_, self.current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell,
                                                                                   self.embeddedWords_,
                                                                                   dtype=tf.float32,
                                                                                   scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                    self.embeddedWords_ = tf.concat(outputs_, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        fwOutput, bwOutput = tf.split(self.embeddedWords_, 2, -1)

        with tf.name_scope("context"):
            shape = [tf.shape(fwOutput)[0], 1, tf.shape(fwOutput)[2]]
            self.contextLeft = tf.concat([tf.zeros(shape), fwOutput[:, :-1]], axis=1, name="contextLeft")
            self.contextRight = tf.concat([bwOutput[:, 1:], tf.zeros(shape)], axis=1, name="contextRight")

        # 将前向，后向的输出和最早的词向量拼接在一起得到最终的词表征
        with tf.name_scope("wordRepresentation"):
            self.wordRepre = tf.concat([self.contextLeft, self.embeddedWords, self.contextRight], axis=2)
            wordSize = config.model.hiddenSizes[-1] * 2 + config.model.embeddingSize

        with tf.name_scope("textRepresentation"):
            outputSize = config.model.outputSize
            textW = tf.Variable(tf.random_uniform([wordSize, outputSize], -1.0, 1.0), name="W2")
            textB = tf.Variable(tf.constant(0.1, shape=[outputSize]), name="b2")

            # tf.einsum可以指定维度的消除运算
            self.textRepre = tf.tanh(tf.einsum('aij,jk->aik', self.wordRepre, textW) + textB)

        # 做max-pool的操作，将时间步的维度消失
        output = tf.reduce_max(self.textRepre, axis=1)

        # 全连接层的输出
        with tf.name_scope("output"):
            outputW = tf.get_variable(
                "outputW",
                shape=[outputSize, config.numClasses],
                initializer=tf.contrib.layers.xavier_initializer())

            outputB = tf.Variable(tf.constant(0.1, shape=[config.numClasses]), name="outputB")
            l2Loss += tf.nn.l2_loss(outputW)
            l2Loss += tf.nn.l2_loss(outputB)
            self.logits = tf.nn.xw_plus_b(output, outputW, outputB, name="logits")

            if config.numClasses == 1:
                self.predictions = tf.cast(tf.greater_equal(self.logits, 0.0), tf.float32, name="predictions")
            elif config.numClasses > 1:
                self.predictions = tf.argmax(self.logits, axis=1, name="predictions")

        # 计算二元交叉熵损失
        with tf.name_scope("loss"):

            if config.numClasses == 1:
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                 labels=tf.cast(tf.reshape(self.input_y, [-1, 1]),
                                                                                dtype=tf.float32))
            elif config.numClasses > 1:
                losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.input_y)

            self.loss = tf.reduce_mean(losses) + config.model.l2RegLambda * l2Loss

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

