import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import tensorrt as trt
from tensorrt.parsers import uffparser

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from random import randint # generate a random test case
import time #import system tools
import os

import uff

#log用のファイルとモデルセーブ用のディレクトリを作る
log_dir = "./logs"
save_dir = "./saves"
def check_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)

check_dir(log_dir)
check_dir(save_dir)

MAX_ITERATION = 19000
BATCH_SIZE = 125

class model():
    def __init__(self):
        self.data_set = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data')

        #LeNet でテスト
        self.inputs = tf.placeholder(tf.float32, [None, 28, 28, 1], name="Placeholder")
        self.labels = tf.placeholder(tf.float32, [None, 10])
        
        h_ = tf.layers.conv2d(self.inputs, filters=32, kernel_size=(5,5), padding="SAME", name="conv_0")
        h_ = tf.nn.relu(h_)
        h_ = tf.layers.max_pooling2d(h_, pool_size=(2,2), strides=(2,2))

        h_ = tf.layers.conv2d(h_, filters=64, kernel_size=(5,5), padding="SAME", name="conv_1")
        h_ = tf.nn.relu(h_)
        h_ = tf.layers.max_pooling2d(h_, pool_size=(2,2), strides=(2,2))

        h_ = tf.reshape(conv_layers[-1], (-1,7 * 7 *64))
        h_ = tf.layers.dense(flatten, 512, tf.nn.relu, name='dense_0')
        h_ = tf.layers.dropout(dense0, rate=0.65, training=training, name='dropout_0')
        h_ = tf.layers.dense(dropout, 10, name='dense_1')
        
        #モデルの出力が分かりやすいようにname_scopeで覆う
        with tf.name_scope("inference"):
            self.inference = tf.nn.softmax(logits)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
        with tf.name_scope("summary"):
            tf.summary.scalar("loss", self.loss)

        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(self.loss)

    def fit(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.log_device_placement = True
        with tf.Session(config=config) as sess:
            #変数の初期化
            tf.global_variables_initializer().run()
            #モデルの重みをセーブするためのセーバー
            saver = tf.train.Saver(tf.global_variables())
            #tensorBoard用のfile writer
            s_writer = tf.summary.FileWriter(logidir=logdir, graph=sess.graph_def())
            #summary
            summary = tf.summary.merge_all()

            for itr, (images_feed, labels_feed) in enumerate(self.data_set.train.next_batch(BATCH_SIZE)):
                feed_dict = {
                    self.inputs: images_feed,
                    self.labels: labels_feed
                }
                
                #lossの計算と, summary, モデルの最適化をする
                loss_, summary_, _ = sess.run([self.loss, summary, self.optimizer], feed_dict)
                
                if itr%10 == 0:
                    #log
                    print(itr, ":  ", loss_)
                    s_writer.add_summary(summary_, itr)
                
                if itr%50 == 0:
                    #modelのセーブ
                    saver.save(sess, os.path.join(save_dir, "model.ckpt"))

                if itr+1 == MAX_ITERATION:
                    break

    def inference(self):
        
        #モデルの読み込み
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, 'save/model.ckpt')

            graph_def = sess.graph_def()
            frozen_graph = tf.graph_util.convert_variables_to_constants(sess,
                                                                    graph_def,
                                                                    ["inference/softmax"])

            tf_model = tf.graph_util.remove_training_nodes(frozen_graph)

        # Tensorflowのモデル形式からUFFへ変換
        uff_model = uff.from_tensorflow(tf_model, ["inference/softmax"])

        # TensorRT EngineのためのUFF Streamを作る
        G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.ERROR)

        # uff parserを作り，モデルの入出力に関する情報を加える
        parser = uffparser.create_uff_parser()
        parser.register_input("Placeholder", (1,28,28), 0)
        parser.register_output("inference/softmax")

        # utility関数を用いてエンジンを作る(最後の引数はmax batch size and max workspace size?)
        engine = trt.utils.uff_to_trt_engine(G_LOGGER, uff_model, parser, 1, 1 << 20)

        # parserは使わないので解放
        parser.destroy()

        #runtime とengineのcontextを作成
        runtime = trt.infer.create_infer_runtime(G_LOGGER)
        context = engine.create_execution_context()

        #データ読み込み(１件のみ)
        img, label = MNIST_DATASETS.test.next_batch(1)
        img = img[0].astype(np.float32)
        label = label[0]

        # GPUにメモリ割り当てと，CPUにメモリ割り当て(推測後の結果を扱うために)
        # 結果受取用の変数
        output = np.empty(10, dtype = np.float32)

        #alocate device memory (The size of the allocations is the size of the input and expected output * the batch size.)
        d_input = cuda.mem_alloc(1 * img.size * img.dtype.itemsize)
        d_output = cuda.mem_alloc(1 * output.size * output.dtype.itemsize)

        bindings = [int(d_input), int(d_output)]

        #推測をするためのcuda Streamを作成
        stream = cuda.Stream()

        #データをGPUに，推測と結果のコピー
        cuda.memcpy_htod_async(d_input, img, stream)
        #推測
        context.enqueue(1, bindings, stream.handle, None)
        #結果のコピー
        cuda.memcpy_dtoh_async(output, d_output, stream)
        #スレッドの同期?
        stream.synchronize()

        print("Test Case: " + str(label))
        print ("Prediction: " + str(np.argmax(output)))

        # Engineのセーブ
        #trt.utils.write_engine_to_file("./tf_mnist.engine", engine.serialize())

        # Engineの読み込み
        #engine = trt.utils.load_engine(G_LOGGER, "./tf_mnist.engine")

        context.destroy()
        engine.destroy()
        new_engine.destroy()
        runtime.destroy()
