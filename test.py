import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import tensorflow.contrib.layers as layers

from ME.utils import BaseModelMixin

try: # import LSTM state
    from tensorflow.contrib.cnn import LSTMStateTuple, LSTMCell, GRUCell
except ImportError:
    LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple
    LSTMCell = tf.nn.rnn_cell.LSTMCell
    GRUCell = tf.nn.rnn_cell.GRUCell
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))

class BiLSTM_Att_CNN(BaseModelMixin):
    def __init__(self, config, emb_vec, l2_reg_lambda=0.0, warmup_steps=400, pos_encoding_type='sinusoid',
                 model_name='BLAC', tf_sess_config=None):

        assert config.model.embeddingSize % config.n_head == 0  # 若d_model能整除num_heads则继续
        assert pos_encoding_type in ('sinusoid', 'embedding')
        super().__init__(model_name, tf_sess_config=tf_sess_config)

        self.h = config.n_head
        self.embedingSize = config.model.embeddingSize
        self.seq_len = config.seq_len
        self.warmup_steps = warmup_steps
        self.batch_size = config.batchSize
        self.drop_rate = tf.placeholder(tf.float32, name="dropoutRate")
        self.pos_encoding_type = pos_encoding_type
        self.emb_vec = emb_vec
        self.hiddenSize = config.model.hiddenSize
        self.hiddenSizes = config.model.hiddenSizes
        self.dropoutProb = tf.placeholder(tf.float32, name="dropoutKeepProb")
        self.filter_sizes = config.cnn_config.filter_sizes
        self.num_filters = config.cnn_config.num_filters
        self.dropWordNum = config.cnn_config.dropWordNum
        self.residual = config.cnn_config.residual
        self.full_units = config.full_units
        self.classNum = config.numClasses
        self.l2_reg_lambda = l2_reg_lambda

        self.config = dict(
            num_heads=self.h,
            seq_len=self.seq_len,
            drop_rate=self.drop_rate,
            warmup_steps=self.warmup_steps,
            model_name=self.model_name,
            tf_sess_config=self.tf_sess_config,)

        # The following variables will be constructed in build_model(). 以下是build_model()函数创建的变量
        self.input_x = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_len], name='raw_input')
        self.input_y = tf.placeholder(tf.float32, [self.batch_size, self.classNum], name="input_y")
        self.input_back_x = tf.placeholder(tf.int32, shape=[self.batch_size, self.seq_len], name='raw_input_back')
        self._learning_rate = None
        self._is_training = tf.placeholder_with_default(True, shape=())
        self._raw_input = None
        self._raw_target = None
        self._output = None
        self._accuracy = None
        self._loss = None
        self._train_op = None
        self._is_init = False
        self.step = 0  # training step.
        self._L2loss = tf.constant(0.0)
    def build_model2(self,input_vocab, is_training=True, useBack=False, **train_params):
        self.config.update(dict(
            train_params=train_params,
        ))

        with tf.variable_scope(self.model_name):
            self._learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')
            self._is_training = tf.placeholder_with_default(is_training, shape=None, name="is_training")
            # Input embedding + positional encoding
            inp_mask = self.construct_padding_mask(self.input_x)
            ###inp_embed = self.preprocess(self.input_x, input_vocab, "input_preprocess")
            if useBack is True:
                W1 = tf.Variable(tf.cast(self.emb_vec, dtype=tf.float32, name="word2vec1"), name="W1")
                inp_embed1 = tf.nn.embedding_lookup(W1, self.input_back_x)

            Ww = tf.Variable(tf.cast(self.emb_vec, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            inp_embed = tf.nn.embedding_lookup(Ww, self.input_x)
            print(inp_embed)
            #bigru_out = self.bidirectional_rnn(inp_embed)
            #print(bigru_out)
            att_out_norm = self.encoder(inp_embed, inp_mask, num_layer=2)
            att_out_norm2 = self.encoder(inp_embed1, inp_mask, scope="e2", num_layer=2)
            att_out_norm_reve = tf.reverse(att_out_norm2, axis=[1])
            att_add = att_out_norm + att_out_norm_reve
            print(att_add)
            bigru_out2 = self.bidirectional_rnn(att_add, name=1)
            print(bigru_out2)

            output_conv_reshape = tf.reshape(bigru_out2, shape=[self.batch_size, -1])

            self.score, self.predictions = self.last_output(output_conv_reshape)
            print("score", self.score)
            print("predictions", self.predictions)

    def build_model(self, input_vocab, is_training=True, **train_params):

        self.config.update(dict(
            train_params=train_params,
        ))

        with tf.variable_scope(self.model_name):
            self._learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')
            self._is_training = tf.placeholder_with_default(is_training, shape=None, name="is_training")
            # Input embedding + positional encoding
            inp_mask = self.construct_padding_mask(self.input_x)
            ###inp_embed = self.preprocess(self.input_x, input_vocab, "input_preprocess")

            Ww = tf.Variable(tf.cast(self.emb_vec, dtype=tf.float32, name="word2vec"), name="W")
            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，维度[batch_size, sequence_length, embedding_size]
            inp_embed = tf.nn.embedding_lookup(Ww, self.input_x)
            print(inp_embed)
            bigru_out = self.bidirectional_rnn(inp_embed)
            print(bigru_out)
            att_out_norm = self.encoder(bigru_out, inp_mask, num_layer=2)
            print(att_out_norm)
            bigru_out2 = self.bidirectional_rnn(att_out_norm, name=1)
            print(bigru_out2)

            # input_conv = tf.expand_dims(att_out_norm, -1)
            # dim_input_conv = input_conv.shape[-2].value
            # print(input_conv)
            # output_conv = self.Conv2(input_conv, dim_input_conv, 4)
            # print("output_conv", output_conv)
            output_conv_reshape = tf.reshape(bigru_out2, shape=[self.batch_size, -1])
            ##conn_output = self.layer_norm(self.full_conn(conn_input))
            ##print(conn_output)
            # num = conn_output.shape[1] * conn_output.shape[2]
            # conn_output = tf.reshape(conn_output, shape=[-1, num])
            self.score, self.predictions = self.last_output(output_conv_reshape)
            print("score", self.score)
            print("predictions", self.predictions)

    def construct_padding_mask(self, inp):
        """
        Args: Original input of word ids, shape [batch, seq_len]
        Returns: a mask of shape [batch, seq_len, seq_len], where <pad> is 0 and others are 1s.
        """
        seq_len = inp.shape.as_list()[1]  # seq_len值
        # tf.not_equal 操作变为一个bool矩阵,相等的为False不相等的为True
        # tf.cast 操作将类型转换为float，False为0，True为1.0
        mask = tf.cast(tf.not_equal(inp, np.int32(0)), tf.float32)  # mask '<pad>'
        # tf.expand_dims 得到一个[batch_size, 1, seq_len]张量
        mask = tf.tile(tf.expand_dims(mask, 1), [1, seq_len, 1])  # 在第1维上复制seq_len次
        return mask  # batch_size * seq_len * seq_len

    def embedding(self, inp, vocab_size, emb_vec, emb_used=True):
        """嵌入词汇"""
        embed_size = self.embedingSize  # 嵌入矩阵维度大小等于self.d_model，为512
        if emb_used:  # 是用预训练过的嵌入矩阵
            #embed_lookup = tf.get_variable("embed_lookup", [vocab_size, embed_size], tf.float32,
            #                               initializer=tf.contrib.layers.xavier_initializer())
            embed_lookup = tf.Variable(tf.cast(emb_vec, dtype=tf.float32, name="word2vec"), name="embed_lookup")
        else:  # 随机初始化潜入矩阵
            # tf.contrib.layers.xavier_initializer()一个权重矩阵的初始化器 vocab_size, embed_size词汇数，嵌入维度
            embed_lookup = tf.get_variable("embed_lookup", [vocab_size, embed_size], tf.float32,
                                           initializer=tf.contrib.layers.xavier_initializer())
        # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
        # embed_lookup中inp对应的元素，即将序列值变换为one-hot向量。
        out = tf.nn.embedding_lookup(embed_lookup, inp)
        return out  # 返回 一个batch_size * seq_len * embed_size

    def _positional_encoding_embedding(self, inp):
        batch_size, seq_len = inp.shape.as_list()

        with tf.variable_scope('positional_embedding'):
            # Copy [0, 1, ..., `inp_size`] by `batch_size` times => matrix [batch, seq_len]
            pos_ind = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch_size, 1])
            return self.embedding(pos_ind, seq_len, emb_used=False)  # [batch, seq_len, d_model]

    def _positional_encoding_sinusoid(self, inp, back=False):  # 使用sinusoid的位置嵌入
        """
        PE(pos, 2i) = sin(pos / 10000^{2i/d_model})
        PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})
        """
        batch, seq_len = inp.shape.as_list()
        with tf.variable_scope('positional_sinusoid'):
            # Copy [0, 1, ..., `inp_size`] by `batch_size` times => matrix [batch, seq_len]
            # 生成一个batch_size * seq_len 的矩阵每一行是[0,...,seq_len - 1]
            if back is True:
                seq_query = np.array([seq_len - 1 - i for i in range(seq_len)])
                pos_ind = tf.tile(tf.expand_dims(tf.convert_to_tensor(seq_query, dtype=tf.int32), 0), [batch, 1])
            else:
                pos_ind = tf.tile(tf.expand_dims(tf.range(seq_len), 0), [batch, 1])

            # Compute the arguments for sin and cos: pos / 10000^{2i/d_model})
            # Each dimension is sin/cos wave, as a function of the position.
            pos_enc = np.array([
                [pos / np.power(10000., 2. * (i // 2) / self.embedingSize) for i in range(self.embedingSize)]
                for pos in range(seq_len)])  # [seq_len, d_model]

            # Apply the cosine to even columns and sin to odds. 将cos应用于偶数列，sin应用于奇数列。
            pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])  # dim 2i
            pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])  # dim 2i+1

            # Convert to a tensor
            lookup_table = tf.convert_to_tensor(pos_enc, dtype=tf.float32)  # [seq_len, d_model]
            # pos_ind每一行的数据是一样的，所以都是一样的
            out = tf.nn.embedding_lookup(lookup_table, pos_ind)  # [batch, seq_len, d_model]
            return out

    def positional_encoding(self, inp, back=False):
        if self.pos_encoding_type == 'sinusoid':  # 若使用正余弦计算嵌入位置信息
            pos_enc = self._positional_encoding_sinusoid(inp, back=back)
        else:  # 否则随机初始化位置权重矩阵。
            pos_enc = self._positional_encoding_embedding(inp)
        return pos_enc

    def preprocess(self, inp, inp_vocab, scope, back=False):  # Pre-processing: embedding + positional encoding
        # Output shape: [batch, seq_len, emb_size]
        with tf.variable_scope(scope):
            out = self.embedding(inp, inp_vocab, self.emb_vec, emb_used=True) + self.positional_encoding(inp, back=back)
            out = tf.layers.dropout(out, rate=self.drop_rate, training=self._is_training)
        return out  # [batch, seq_len, emb_size]

    def bidirectional_rnn(self, emb_vec, rnn_type=True, name=0):
        """Bidirecional RNN with concatenated outputs and states"""
        with tf.name_scope("BiGRU"+str(name)):
            # 定义前向LSTM结构
            if rnn_type is False:
                RNNCell = GRUCell(num_units=self.hiddenSizes[name])
            else:
                RNNCell = LSTMCell(num_units=self.hiddenSizes[name], state_is_tuple=True)
            lstmFwCell = tf.nn.rnn_cell.DropoutWrapper(RNNCell, output_keep_prob=self.dropoutProb)
            # 定义反向LSTM结构
            lstmBwCell = tf.nn.rnn_cell.DropoutWrapper(RNNCell, output_keep_prob=self.dropoutProb)
            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长outputs是一个元祖(output_fw, output_bw)，
            # 其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
            #self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元组(h, c)
            outputs_, current_state = tf.nn.bidirectional_dynamic_rnn(lstmFwCell, lstmBwCell, inputs=emb_vec,
                                                                      dtype=tf.float32, scope="bi-gru"+str(name))
            # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
            BiRNNOutputs = tf.concat(outputs_, 2)

        return BiRNNOutputs

    def feed_forwad(self, inp, scope='ff'):
        """
        Position-wise fully connected feed-forward network, applied to each position
        separately and identically. It can be implemented as (linear + ReLU + linear) or
        (conv1d + ReLU + conv1d).
        Args:
            inp (tf.tensor): shape [batch, length, d_model]
        """
        out = inp
        with tf.variable_scope(scope):
            # out = tf.layers.dense(out, self.d_ff, activation=tf.nn.relu)
            # out = tf.layers.dropout(out, rate=self.drop_rate, training=self._is_training)
            # out = tf.layers.dense(out, self.d_model, activation=None)

            # by default, use_bias=True d_ff 前馈层的神经元数目。卷积核为1
            out = tf.layers.conv1d(out, filters=800, kernel_size=1, activation=tf.nn.relu)
            out = tf.layers.conv1d(out, filters=200, kernel_size=1)
            ## out = self.bidirectional_rnn(inp)

        return out

    def scaled_dot_product_attention(self, Q, K, V, lats_dim, mask=None):
        """
        Args:
            Q (tf.tensor): of shape (h * batch, q_size, embedingSize) q_size = seq_len
            K (tf.tensor): of shape (h * batch, k_size, embedingSize) k_size = seq_len
            V (tf.tensor): of shape (h * batch, k_size, embedingSize)
            mask (tf.tensor): of shape (h * batch, q_size, k_size) k_size = q_size = seq_len
        """
        d = lats_dim // self.h  # 此处为200 / 8 = 25
        assert d == Q.shape[-1] == K.shape[-1] == V.shape[-1]

        out = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # [h*batch, q_size, k_size]
        out = out / tf.sqrt(tf.cast(d, tf.float32))  # scaled by sqrt(d_k)

        # if mask is not None:
        #     # masking out (0.0) => setting to -inf.  tf.multiply对应点相乘，掩盖<pad>标签
        #     out = tf.multiply(out, mask) + (1.0 - mask) * (-1e10)

        out = tf.nn.softmax(out)  # [h * batch, q_size, k_size]
        out = tf.layers.dropout(out, self.drop_rate, training=self._is_training)
        out = tf.matmul(out, V)  # [h * batch, q_size, d_model]

        return out  # h * batch, q_size（seq_len）, d_model

    def multihead_attention(self, query, lats_dim, memory=None, mask=None, scope='attn'):
        """
        Args:
            query (tf.tensor): of shape (batch, q_size, embed_size) ## batch_size * seq_len * embed_size(d_model)
            memory (tf.tensor): of shape (batch, m_size, embed_size) ##
            mask (tf.tensor): shape (batch, q_size, k_size) ## batch_size * seq_len * seq_len
        Returns:h
            a tensor of shape (bs, q_size, d_model)
        """
        if memory is None:
            memory = query

        with tf.variable_scope(scope):
            # Linear project to d_model dimension: [batch, q_size/k_size, d_model]
            # 经过全连接后得到一个 batch * （q_size/k_size）* embedingSize的矩阵 /是 或 的意思
            Q = tf.layers.dense(query, lats_dim, activation=tf.nn.relu)
            K = tf.layers.dense(memory, lats_dim, activation=tf.nn.relu)
            V = tf.layers.dense(memory, lats_dim, activation=tf.nn.relu)
            # Split the matrix to multiple heads and then concatenate to have a larger
            # batch size: [h*batch, q_size/k_size, embedingSize/num_heads]
            # self.h 是head数目。tf.split(Q, self.h, axis=2) 按每行划分出来d_model是200，所以划分出8个矩阵
            # 最后连接成 [h*batch, q_size/k_size, embedingSize/num_heads]，第一个/是或，第二个/是除
            Q_split = tf.concat(tf.split(Q, self.h, axis=2), axis=0)
            K_split = tf.concat(tf.split(K, self.h, axis=2), axis=0)
            V_split = tf.concat(tf.split(V, self.h, axis=2), axis=0)
            # h*batch * seq_len * seq_len。这里是在批次上扩充self.h次，其余不变。
            mask_split = tf.tile(mask, [self.h, 1, 1])

            # Apply scaled dot product attention h * batch, q_size(seq_len), embedingSize
            out = self.scaled_dot_product_attention(Q_split, K_split, V_split, lats_dim, mask=mask_split)

            # Merge the multi-head back to the original shape 合并多头回到原来的形状
            out = tf.concat(tf.split(out, self.h, axis=0), axis=2)  # [batch_size, seq_len, embedingSize]
        return out

    def encoder_layer(self, inp, input_mask, scope):
        """
        Args:
            inp: tf.tensor of shape (batch, seq_len, embed_size)
            input_mask: tf.tensor of shape (batch, seq_len, seq_len)
        """
        out = inp
        with tf.variable_scope(scope):
            # One multi-head attention + one feed-forword
            lats_dim = inp.shape[-1]
            out = self.layer_norm(out + self.multihead_attention(out, lats_dim, mask=input_mask))
            #out = self.layer_norm(out + self.feed_forwad(out))
        return out

    def encoder(self, inp, input_mask, scope='encoder', num_layer=2):
        """
        Args:
            inp (tf.tensor): shape (batch, seq_len, embed_size)
            input_mask (tf.tensor): shape (batch, seq_len, seq_len)
            scope (str): name of the variable scope.
        """
        out = inp  # now, (batch, seq_len, embed_size)
        with tf.variable_scope(scope):
            for i in range(num_layer):  # 默认六层
                out = self.encoder_layer(out, input_mask, 'enc_{}'.format(i))
        return out

    def layer_norm(self, inp):
        return tc.layers.layer_norm(inp, center=True, scale=True)

    def Conv(self, input_conv_expanded, dim_input_conv, num=None):
        with tf.variable_scope("CNN-{}".format(num)):
            conv_outputs = []
            pooled_outputs = []
            for i, filter_size in enumerate(self.filter_sizes): # 过滤器列表 3，4，5
                with tf.name_scope("gated-conv-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, dim_input_conv, 1, self.num_filters]

                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")

                    W_g = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_g")
                    b_g = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b_g")

                    conv_w = self.conv_op(W, b, input_conv_expanded, end="linear")
                    conv_v = self.conv_op(W_g, b_g, input_conv_expanded, end="gatad")
                    h = conv_w * tf.sigmoid(conv_v)

                    h_pool = self.K_Max_Pooling(h, 60)
                    conv_outputs.append(h_pool)
                    print("cov {}".format(i), h_pool)

            # batch * (sequence_length - filter_size + 1) * 1 * num_filters * 3
            h_pool = tf.concat(conv_outputs, 2)
            print(h_pool)
            h_drop = tf.layers.dropout(h_pool, self.dropoutProb, training=self._is_training)

        return h_drop

    def conv_op(self, W, b, inp, end=None):
        return tf.add(tf.nn.conv2d(inp, W, strides=[1, 1, 1, 1], padding='VALID', name="conv_{}".format(end)), b)

    def K_Max_Pooling(self, conv_outpot, k_num, pd=True):
        if pd:
            pool_input = tf.squeeze(conv_outpot, axis=2)
        else:
            pool_input = conv_outpot

        pool_output = tf.nn.top_k(tf.transpose(pool_input, [0, 2, 1]), k=k_num)[0]
        pool_output = tf.transpose(pool_output, [0, 2, 1])

        return pool_output

    def full_conn(self, inp):
        conn_output = layers.fully_connected(inp, num_outputs=self.embedingSize,
                                     weights_initializer=tf.contrib.layers.xavier_initializer(),
                                     biases_initializer=tf.constant_initializer(0.1))

        return conn_output

    def Conv2(self, input_conv_expanded, dim_input_conv, num_layers):
        with tf.variable_scope("Milt_CNN"):
            inpu = [input_conv_expanded, input_conv_expanded, input_conv_expanded]
            for z in range(num_layers):
                with tf.variable_scope("CNN-{}".format(str(z))):
                    conv_outputs = []
                    inputs = []
                    for i, filter_size in enumerate(self.filter_sizes): # 过滤器列表 3，4，5
                        with tf.name_scope("gated-conv-%s" % filter_size):
                            # Convolution Layer
                            filter_shape = [filter_size, dim_input_conv, 1, self.num_filters]

                            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                            b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")

                            W_g = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_g")
                            b_g = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b_g")

                            conv_w = self.conv_op(W, b, inpu[i], end="linear")
                            conv_v = self.conv_op(W_g, b_g, inpu[i], end="gatad")
                            h = conv_w * tf.sigmoid(conv_v)
                            print(h)
                            inputs.append(h)
                    for q in range(len(self.filter_sizes)):
                        if z < num_layers-1:
                            inpu[q] = tf.transpose(inputs[q], [0, 1, 3, 2])
                        else:
                            inpu[q] = inputs[q]

            for i in range(len(self.filter_sizes)):
                h_pool = self.K_Max_Pooling(inpu[i], 60)
                conv_outputs.append(h_pool)
                print("cov {}".format(i), h_pool)

            # batch * (sequence_length - filter_size + 1) * 1 * num_filters * 3
            h_pool = tf.concat(conv_outputs, 2)
            print(h_pool)
            h_drop = tf.layers.dropout(h_pool, self.dropoutProb, training=self._is_training)

        return h_drop

    def attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hiddenSize = self.hiddenSizes[-1] * 2

        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hiddenSize], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # 对W和M做矩阵运算，W=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hiddenSize]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.seq_len])

        # 用softmax做归一化处理[batch_size, time_step]
        alpha = tf.nn.softmax(restoreM)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, self.seq_len, 1]))

        # 将三维压缩成二维sequeezeR=[batch_size, hidden_size]
        sequeezeR = tf.reshape(r, [-1, hiddenSize])

        sentenceRepren = tf.tanh(sequeezeR)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentenceRepren, self.dropoutProb)

        return output

    def last_output(self, inp):
        with tf.name_scope("last_output"):
            # 获取W变量，若没有则初始化，
            W = tf.get_variable("W", shape=[inp.shape[1], self.classNum],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.classNum]), name="b")
            self._L2loss += tf.nn.l2_loss(W)
            self._L2loss += tf.nn.l2_loss(b)
            # tf.nn.xw_plus_b相当于matmul(self.h_drop, W) + b.
            scores = tf.nn.xw_plus_b(inp, W, b, name="scores")
            predictions = tf.argmax(scores, 1, name="predictions")

            return scores, predictions

    def Loss_func(self, scores):
        with tf.name_scope("loss"):
            self.loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=scores, labels=self.input_y)
            self.loss = tf.reduce_mean(self.loss) + self.l2_reg_lambda * self._L2loss
        return self.loss

    def accuracy_func(self, pre):
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(pre, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        return self.accuracy