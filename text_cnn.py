import tensorflow as tf
import numpy as np

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(self, sequence_length, num_classes, vocab_size, batch_size,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [batch_size, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [batch_size, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob") # 训练时变化

        # Keeping track of l2 regularization loss (optional)
        # 用于追踪l2正则化损失
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # 使用高斯分布初始化权重，并且限制在[-1,1]之间，大小为[vocab_size, embedding_size]
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            # 根据未知批次和序列大小选择嵌入向量。batch_size * seq_len * embedding_size
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 在行的每个元素上加一维。batch_size * seq_len * embedding_size * 1
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = [] # 连续不同尺寸的三个卷积池化层********重点*********
        for i, filter_size in enumerate(filter_sizes): # 过滤器列表 3，4，5
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # tf.truncated_normal 这是一个截断的产生正太分布的函数，
                # 就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成。
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # 经过卷积层得到的输出 batch * ((sequence_length - filter_size)/1 + 1) * 1 * num_filters
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W, # 过滤器权重数据
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity 使用非线性层
                # 以下输出batch * ((sequence_length - filter_size)/1 + 1) * 1 * num_filters
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                # 最大池化后的维度 batch * 1 * 1 * num_filters
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes) # 三种过滤器每种有128个
        # batch * (sequence_length - filter_size + 1) * 1 * num_filters * 3
        self.h_pool = tf.concat(pooled_outputs, 3)
        # (batch * (sequence_length - filter_size + 1) * 1 * num_filters * 3)/num_filters_total, num_filters_total
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        # 最终的分数和预测
        with tf.name_scope("output"):
            # 获取W变量，若没有则初始化，
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # tf.nn.xw_plus_b相当于matmul(self.h_drop, W) + b.
            # (batch * (sequence_length - filter_size + 1) * 1 * num_filters * 3) / num_filters_total, num_classes
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            #(batch * (sequence_length - filter_size + 1) * 1 * num_filters * 3) / num_filters_total的0,1向量
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
