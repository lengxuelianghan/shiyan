import warnings
import time
import numpy as np
import tensorflow as tf
import csv
import os
import datetime
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from ME.ME_below1.dataTest import Dataset, nextBatch
from ME.ME_below1.test import BiLSTM_Att_CNN
from ME.ME_below1.text_cnn import TextCNN
from ME.ME_below1.BLA import BiLSTMAttention
from ME.ME_below1.RCNN import RCNN
warnings.filterwarnings("ignore")

# 配置参数
class ConvConfig(object):
    filter_sizes = [3, 4, 5]
    num_filters = 200
    residual = 2
    dropWordNum = 20

class TrainingConfig(object):
    epoches = 20
    evaluateEvery = 100
    checkpointEvery = 100
    learningRate = 0.001

class ModelConfig(object):
    embeddingSize = 200
    hiddenSize = 100 # LSTM结构的神经元个数
    hiddenSizes = [100, 100]
    outputSize = 128
    dropoutProb = 0.5
    l2RegLambda = 0.0

class Config(object):
    #dataSource = "./data/preProcess/labeledTrain.csv"
    stopWordSource = "./data/english"

    numClasses = 2  # 二分类设置为2，多分类设置为类别的数目
    rate = 0.8  # 训练集的比例
    batchSize = 64
    seq_len = 100  # "Input sequence length."
    n_head = 8
    full_units = 200

    training = TrainingConfig()
    model = ModelConfig()
    cnn_config = ConvConfig()
is_training = tf.placeholder_with_default(False, shape=())
def train(num):
    # 实例化配置参数对象
    config = Config()
    data = Dataset(config)
    data.dataGen()
    trainReviews = data.trainReviews
    trainLabels = data.trainLabels
    evalReviews = data.evalReviews
    evalLabels = data.evalLabels
    testReviews = data.testReviews
    testLabels = data.testLabels

    wordEmbedding = data.wordEmbedding
    labelList = data.labelList
    input_vocab = len(wordEmbedding)
    print(input_vocab)

    train_params = dict(  # 训练参数
        learning_rate=1e-4,
        batch_size=64,
        seq_len=128,
        max_steps=300,)
    # 定义计算图
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session_conf.gpu_options.per_process_gpu_memory_fraction = 0.9  # 配置gpu占用率
        sess = tf.Session(config=session_conf)

        # 定义会话
        with sess.as_default():
            global_step = tf.Variable(0, name="global_step", trainable=False)
            if num == 1:
                lstm = BiLSTM_Att_CNN(config, wordEmbedding)
                lstm.build_model(input_vocab, **train_params)
                losses = lstm.Loss_func(lstm.score)
                accuracys = lstm.accuracy_func(lstm.predictions)
            elif num == 2:
                lstm = TextCNN(
                    sequence_length=config.seq_len,
                    num_classes=config.numClasses,
                    vocab_size=input_vocab,
                    batch_size=config.batchSize,
                    embedding_size=config.model.embeddingSize,
                    filter_sizes=config.cnn_config.filter_sizes,  # map是映射函数，变为整数
                    num_filters=config.cnn_config.num_filters,
                    l2_reg_lambda=config.model.l2RegLambda)
                losses = lstm.loss
                accuracys = lstm.accuracy
            elif num == 3:
                lstm = BiLSTMAttention(config, wordEmbedding)
                losses = lstm.loss
                accuracys = lstm.accuracy

            else:
                lstm = RCNN(config, wordEmbedding)
                losses = lstm.loss
                accuracys = lstm.accuracy

            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(losses)  # 返回gradient, variable
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # 用summary绘制tensorBoard
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            loss_summary = tf.summary.scalar("loss", losses)
            acc_summary = tf.summary.scalar("accuracy", accuracys)

            # Train Summaries 训练总结
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Test summaries
            test_summary_op = tf.summary.merge([acc_summary])
            test_summary_dir = os.path.join(out_dir, "summaries", "test")
            test_summary_writer = tf.summary.FileWriter(test_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")

            if not os.path.exists(checkpoint_dir):  # 若目录不存在则创建目录
                os.makedirs(checkpoint_dir)
            # tf.train.Saver提供保存和恢复模型的方法
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

            # 保存模型的一种方式，保存为pb文件
            savedModelPath = "./model/bilstm-atten/savedModel"
            if os.path.exists(savedModelPath):
                os.rmdir(savedModelPath)
            builder = tf.saved_model.builder.SavedModelBuilder(savedModelPath)

            sess.run(tf.global_variables_initializer())

            def trainStep(batchX, batchY):
                """
                训练函数
                """
                if num == 1:
                    feed_dict = {
                        lstm.input_x: batchX,
                        lstm.input_y: batchY,
                        lstm.dropoutProb: 0.5,
                        lstm.drop_rate: 0.1,
                        lstm._is_training: True
                    }
                elif num == 2:
                    feed_dict = {
                        lstm.input_x: batchX,
                        lstm.input_y: batchY,
                        lstm.dropout_keep_prob: 0.5
                    }
                else:
                    feed_dict = {
                        lstm.input_x: batchX,
                        lstm.input_y: batchY,
                        lstm.dropoutKeepProb: 0.5
                    }
                _, summary, step, loss, acc, pre = sess.run(
                    [train_op, train_summary_op, global_step, lstm.loss, accuracys, lstm.predictions], feed_dict)

                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {},{}".format(time_str, step, loss, np.sum(list(pre))))
                train_summary_writer.add_summary(summary, step)
                return loss, acc

            def devStep(batchX, batchY):
                """
                验证函数
                """
                if num == 1:
                    feed_dict = {
                        lstm.input_x: batchX,
                        lstm.input_y: batchY,
                        lstm.dropoutProb: 1.0,
                        lstm.drop_rate: 0.1,
                        lstm._is_training: True
                    }
                elif num == 2:
                    feed_dict = {
                        lstm.input_x: batchX,
                        lstm.input_y: batchY,
                        lstm.dropout_keep_prob: 1.0
                    }
                else:
                    feed_dict = {
                        lstm.input_x: batchX,
                        lstm.input_y: batchY,
                        lstm.dropoutKeepProb: 1.0
                    }
                summary, step, loss, accuracy, pre = sess.run([dev_summary_op, global_step, losses, accuracys, lstm.predictions], feed_dict)
                dev_summary_writer.add_summary(summary, step)
                print(np.sum(list(pre)))
                return loss, accuracy

            def testStep(batchX, batchY):
                """
                验证函数
                """
                if num == 1:
                    feed_dict = {
                        lstm.input_x: batchX,
                        lstm.input_y: batchY,
                        lstm.dropoutProb: 1.0,
                        lstm.drop_rate: 0.1,
                        lstm._is_training: True
                    }
                elif num == 2:
                    feed_dict = {
                        lstm.input_x: batchX,
                        lstm.input_y: batchY,
                        lstm.dropout_keep_prob: 1.0
                    }
                else:
                    feed_dict = {
                        lstm.input_x: batchX,
                        lstm.input_y: batchY,
                        lstm.dropoutKeepProb: 1.0
                    }
                step, predictions, accuracy = sess.run([global_step, lstm.predictions, accuracys], feed_dict)
                return predictions, accuracy

            for i in range(config.training.epoches):
                # 训练模型
                eval_acc = []
                eval_loss = []
                print("start training model ", i)
                for batchTrain in nextBatch(trainReviews, trainLabels, config.batchSize):
                    train_step_loss, train_step_acc = trainStep(batchTrain[0], batchTrain[1])
                    # currentStep = tf.train.global_step(sess, global_step) # 当前状态
                    # if currentStep % 20 == 0:
                    #     break

                for batchEval in nextBatch(evalReviews, evalLabels, config.batchSize):
                    eval_step_loss, eval_step_acc = devStep(batchEval[0], batchEval[1])
                    eval_acc.append(eval_step_acc)
                    eval_loss.append(eval_step_loss)
                    # print("acc:{}, loss:{}".format(eval_step_acc, eval_step_loss))
                    # currentStep = tf.train.global_step(sess, global_step)  # 当前状态
                    # if currentStep % 20 == 0:
                    #     break
                eval_acc_mean = np.mean(eval_acc)
                eval_loss_mean = np.mean(eval_loss)
                print("eval acc:{}, loss:{}".format(eval_acc_mean, eval_loss_mean))
                # 保存模型的另一种方法，保存checkpoint文件
                # My_path = "./model/Bi-LSTM-atten-Cnn/model/my-model"
                # if not os.path.exists(My_path):
                #     os.makedirs(My_path)
                #
                # currentStep = tf.train.global_step(sess, global_step)  # 当前状态
                # path = saver.save(sess, My_path, global_step=currentStep)
                # print("Saved model checkpoint to {}\n".format(path))

                preds = []
                y_test = []
                for batchTest in nextBatch(testReviews, testLabels, config.batchSize):
                    test_step_pred, test_step_acc = testStep(batchTest[0], batchTest[1])
                    preds += list(test_step_pred)
                    y_test += [0 if (y_l[0] - y_l[1] > 0) else 1 for y_l in batchTest[1]]
                    # for ij in test_step_pred:
                    #     print(ij, end="")
                    # print()
                    print("sum:",np.sum(list(test_step_pred)))
                    # currentStep = tf.train.global_step(sess, global_step)  # 当前状态
                    # if currentStep % 20 == 0:
                    #     break

                y_test = np.array(y_test)
                preds = np.array(preds)
                print("yuce:", np.sum(preds), np.sum(y_test))
                [precision_result, recall_result, F1_result, support_result] = \
                     precision_recall_fscore_support(y_test, preds, average='macro')
                acc_result = accuracy_score(y_test, preds)
                print(preds)
                print(y_test)
                print(precision_result, recall_result, F1_result, acc_result)
                append_csv(num, [i, precision_result, recall_result, F1_result, acc_result])

def create_csv(num):
    path = "./save_file/model" + str(num) + ".csv"
    if not os.path.exists("save_file"):
        os.makedirs("save_file")
    with open(path, "w+", newline='') as file:
        csv_file = csv.writer(file)
        head = ["each", "acc", "F1", "recall", "precision"]
        csv_file.writerow(head)

def append_csv(num, data):
    path = "./save_file/model" + str(num) + ".csv"
    with open(path, "a+", newline='') as file: # 处理csv读写时不同换行符  linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        csv_file.writerows([data])


if __name__ == "__main__":
    num = int(input("请输入序号：1，2，3"))
    create_csv(num)
    train(num)