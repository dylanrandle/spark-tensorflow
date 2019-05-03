from pyspark import SparkConf, SparkContext
from pyspark.sql.types import *
import pyspark
from elephas.spark_model import SparkModel
# keras imports
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
# from youtube_model import create_model
import numpy as np

# ============
# SPARK SETUP
# ============

conf = SparkConf().setAppName('Youtube-8M') \
                  .set("spark.jars",
                       "ecosystem/spark/spark-tensorflow-connector/target/spark-tensorflow-connector_2.11-1.10.0.jar")
sc = SparkContext(conf = conf)
spark = pyspark.sql.SparkSession(sc)

# =============
# READING DATA
# =============

# train_df = spark.read.format("tfrecords").option("recordType", "Example").load("yt8pm_100th_shard/v2/video/train*.tfrecord")

# data_path = lambda set: "s3youtube/yt8pm_100th_shard/v2/video/{}*.tfrecord".format(set)
train_df = spark.read.format("tfrecords").option("recordType", "Example").load("yt8pm_100th_shard/v2/video/train*.tfrecord")
# val_df = spark.read.format("tfrecords").option("recordType", "Example").load(data_path('validate'))
# train_df = spark.read.format("tfrecords").option("recordType", "Example").load(data_path('test'))

# ==============
# PREPROCESSING
# ==============

# 1) take audio and rgb features
# 2) use only top 20 classes
train_df = train_df.select('mean_rgb', 'labels')
train_rdd = train_df.rdd
train_rdd = train_rdd.map(lambda x: (x[0], x[1]))
def convert_labels(labels):
    allowed=np.arange(20)
    one_hot=np.zeros(20)
    for l in labels:
        if l in allowed:
            one_hot[l]=1
    return one_hot
train_rdd = train_rdd.map(lambda x: (np.array(x[0]), convert_labels(x[1])))

# =========
# TRAINING
# =========

max_frame_rgb_sequence_length = 10
frame_rgb_embedding_size = 1024

max_frame_audio_sequence_length = 10
frame_audio_embedding_size = 128

number_dense_units = 1000
number_lstm_units = 100
rate_drop_lstm = 0.2
rate_drop_dense = 0.2
activation_function='relu'
validation_split_ratio = 0.2

label_feature_size = 20

def create_model():
    """Create and store best model at `checkpoint` path ustilising bi-lstm layer for frame level data of videos"""
    # Filip: without the frame-level data, we don't actually have a bi-LSTM

    # Creating 2 bi-lstm layer, one for rgb and other for audio level data
#     lstm_layer_1 = Bidirectional(LSTM(number_lstm_units, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))
#     lstm_layer_2 = Bidirectional(LSTM(number_lstm_units, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

    # creating input layer for frame-level data
    # FILIP: these below are frame-level features
#     frame_rgb_sequence_input = Input(shape=(max_frame_rgb_sequence_length, frame_rgb_embedding_size), dtype='float32')
#     frame_audio_sequence_input = Input(shape=(max_frame_audio_sequence_length, frame_audio_embedding_size), dtype='float32')

#     frame_x1 = lstm_layer_1(frame_rgb_sequence_input)
#     frame_x2 = lstm_layer_2(frame_audio_sequence_input)

    ### - Below un-deleted
    #creating input layer for video-level data
    vid_shape=(1024,)
    video_rgb_input = Input(shape=vid_shape)
    video_rgb_dense = Dense(int(number_dense_units/2), activation=activation_function, input_shape=vid_shape)(video_rgb_input)

#     aud_shape=(128,)
#     video_audio_input = Input(shape=aud_shape)
#     video_audio_dense = Dense(int(number_dense_units/2), activation=activation_function,input_shape = aud_shape)(video_audio_input)
#     ### - Above un-deleted

    # merging frame-level bi-lstm output and later passed to dense layer by applying batch-normalisation and dropout
#     merged_frame = concatenate([frame_x1, frame_x2])
#     merged_frame = BatchNormalization()(merged_frame)
#     merged_frame = Dropout(rate_drop_dense)(merged_frame)
#     merged_frame_dense = Dense(int(number_dense_units/2), activation=activation_function)(merged_frame)


    ### - Below un-deleted
    # merging video-level dense layer output
#     merged_video = concatenate([video_rgb_dense, video_audio_dense])
    merged_video = BatchNormalization()(video_rgb_dense)
    merged_video = Dropout(rate_drop_dense)(merged_video)
    merged_video_dense = Dense(int(number_dense_units/2), activation=activation_function)(merged_video)
    ### - Above un-deleted

    # merging frame-level and video-level dense layer output
    merged = merged_video_dense#merged_frame_dense#concatenate([merged_frame_dense, merged_video_dense])
    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)

    merged = Dense(number_dense_units, activation=activation_function)(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)
    preds = Dense(label_feature_size, activation='sigmoid')(merged)

    model = Model(inputs=video_rgb_input, outputs=preds)
    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['acc'])
    return model

keras_model = create_model()
spark_model = SparkModel(keras_model, frequency='batch', mode='synchronous')
history = spark_model.fit(train_rdd, epochs=10, batch_size=32, verbose=0)

# =========
# TESTING
# =========
# score = spark_model.master_network.evaluate(x_test, y_test, verbose=2)
# print('Test accuracy:', score[1])
