import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

import os
from glob import glob
from tqdm import tqdm
import sys

# keras imports
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
import time
import gc


def extract_video_files(video_files_path):
    '''
    Extraction of Youtube tfrecords video file features.

    Args: path to video files (note: developed with assumption of storing on s3 bucket and assessing with glob)

    Assumes each video in the tfrecord has following features:
    'id' : bytes_list
    'labels' : int64_list
    'mean_rgb': float_list
    'mean_audio': float_list

    returns:
    numpy arrays of video ids, video multi-labels, mean rgb and mean audio
    '''

    vid_ids = []
    labels = []
    mean_rgb = []
    mean_audio = []

    for file in tqdm(glob(video_files_path)):
        for example in tf.python_io.tf_record_iterator(file):
            tf_example = tf.train.Example.FromString(example)

            vid_ids.append(tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
            labels.append(tf_example.features.feature['labels'].int64_list.value)
            mean_rgb.append(tf_example.features.feature['mean_rgb'].float_list.value)
            mean_audio.append(tf_example.features.feature['mean_audio'].float_list.value)

    assert len(vid_ids) == len(labels),"The number of IDs does not match the number of labeled videos."
    return vid_ids, labels, mean_rgb, mean_audio


def extract_frame_level_features(frame_files_path,maximum_iter = False,stop_at_iter = 10,num_tf_records=1):
    '''
    Extraction of Youtube tfrecords frame file features.

    Args:
    path to video files (note: developed with assumption of storing on s3 bucket and assessing with glob)

    maximum_iter - flag- if True, will limit number of videos extracted from each TF record
    stop_at_iter - number of videos to extract
    num_tf_records - number of records to extract - WARNING!!! this is VERY slow, if bigger than 1

    Assumes each video in the tfrecord has following features:
    'id' : bytes_list
    'labels' : int64_list
    'audio': float arr, each frame 128
    'rgb', float arr, each frame 1024

    returns:
    numpy arrays of frame ids, frame multi-labels, frame audio, frame rgb
    '''
    frame_ids = []
    frame_labels = []
    feat_rgb = []
    feat_audio = []
    # ATTENTION: only use one TF record for debugging.
    for file in tqdm(glob(frame_files_path)[:num_tf_records]):
        print(f'There is {sum(1 for _ in tf.python_io.tf_record_iterator(file))} videos in this TF record.')
        iter_ = 0
        for example in tf.python_io.tf_record_iterator(file):
            if maximum_iter and iter_==stop_at_iter:
                break
            tf_example = tf.train.Example.FromString(example)

            frame_ids.append(tf_example.features.feature['id'].bytes_list.value[0].decode(encoding='UTF-8'))
            frame_labels.append(tf_example.features.feature['labels'].int64_list.value)

            tf_seq_example = tf.train.SequenceExample.FromString(example)
            n_frames = len(tf_seq_example.feature_lists.feature_list['audio'].feature)

            rgb_frame = []
            audio_frame = []
            # iterate through frames
            sys.stdout.flush()
            for i in range(n_frames):
                sess = tf.InteractiveSession()
                sys.stdout.write('\r'+'iterating video: ' + str(iter_)+ ' ,frames: ' + str(i)+'/'+str(n_frames))
                sys.stdout.flush()
                rgb_frame.append(tf.cast(tf.decode_raw(
                        tf_seq_example.feature_lists.feature_list['rgb'].feature[i].bytes_list.value[0],tf.uint8)
                               ,tf.float32).eval())
                audio_frame.append(tf.cast(tf.decode_raw(
                        tf_seq_example.feature_lists.feature_list['audio'].feature[i].bytes_list.value[0],tf.uint8)
                               ,tf.float32).eval())

                tf.reset_default_graph()
                sess.close()
            feat_rgb.append(rgb_frame)
            feat_audio.append(audio_frame)
            iter_+=1

    return frame_ids, frame_labels, feat_rgb, feat_audio
##########
# SPECIFY MODEL PARAMS

# 10 class problem for now?
label_feature_size = 10

# how many frames we will use from each video?
max_frame_rgb_sequence_length = 10
frame_rgb_embedding_size = 1024

# how many audio sequences we will use from each video?
max_frame_audio_sequence_length = 10
frame_audio_embedding_size = 128

number_dense_units = 1000
number_lstm_units = 100
rate_drop_lstm = 0.2
rate_drop_dense = 0.2
activation_function='relu'
validation_split_ratio = 0.2
##############



def create_model():
    """Create and store best model at `checkpoint` path ustilising bi-lstm layer for frame level data of videos"""

    # Creating 2 bi-lstm layer, one for rgb and other for audio level data
    lstm_layer_1 = Bidirectional(LSTM(number_lstm_units, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))
    lstm_layer_2 = Bidirectional(LSTM(number_lstm_units, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm))

    # creating input layer for frame-level data
    frame_rgb_sequence_input = Input(shape=(max_frame_rgb_sequence_length, frame_rgb_embedding_size), dtype='float32')
    frame_audio_sequence_input = Input(shape=(max_frame_audio_sequence_length, frame_audio_embedding_size), dtype='float32')
    frame_x1 = lstm_layer_1(frame_rgb_sequence_input)
    frame_x2 = lstm_layer_2(frame_audio_sequence_input)

    #creating input layer for video-level data
    vid_shape=(1024,)
    video_rgb_input = Input(shape=vid_shape)
    video_rgb_dense = Dense(int(number_dense_units/2), activation=activation_function, input_shape=vid_shape)(video_rgb_input)

    aud_shape=(128,)
    video_audio_input = Input(shape=aud_shape)
    video_audio_dense = Dense(int(number_dense_units/2), activation=activation_function,input_shape = aud_shape)(video_audio_input)

    # merging frame-level bi-lstm output and later passed to dense layer by applying batch-normalisation and dropout
    merged_frame = concatenate([frame_x1, frame_x2])
    merged_frame = BatchNormalization()(merged_frame)
    merged_frame = Dropout(rate_drop_dense)(merged_frame)
    merged_frame_dense = Dense(int(number_dense_units/2), activation=activation_function)(merged_frame)

    # merging video-level dense layer output
    merged_video = concatenate([video_rgb_dense, video_audio_dense])
    merged_video = BatchNormalization()(video_rgb_dense)
    merged_video = Dropout(rate_drop_dense)(merged_video)
    merged_video_dense = Dense(int(number_dense_units/2), activation=activation_function)(merged_video)


    # merging frame-level and video-level dense layer output
    merged = concatenate([merged_frame_dense, merged_video_dense])
    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)

    merged = Dense(number_dense_units, activation=activation_function)(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(rate_drop_dense)(merged)
    preds = Dense(label_feature_size, activation='sigmoid')(merged)

    model = Model(inputs=[frame_rgb_sequence_input, frame_audio_sequence_input, video_rgb_input, video_audio_input], outputs=preds)

    print(model.summary())

    model.compile(loss='categorical_crossentropy', optimizer='nadam', metrics=['acc'])

    return model

def create_train_dev_dataset(video_rgb, video_audio, vid_ids, frame_rgb, frame_audio, frame_labels, frame_ids):
    """
    Method to created training and validation data.
    We need to make sure we only use video IDs for which we have frames.
    This is handled below.

    """
    # we have to have the same video of for both video and frame-level features
    video_rgb_matching = []
    video_audio_matching = []

    for idx in frame_ids: # for each ID available on frame level, find matching video-level features
        for i, idx_vid in enumerate(vid_ids): # scan through video-level ids
            if idx == idx_vid:
                video_rgb_matching.append(video_rgb[i])
                video_audio_matching.append(video_audio[i])


    shuffle_indices = np.random.permutation(np.arange(len(frame_labels)))

    video_rgb_shuffled = np.array(video_rgb_matching)[shuffle_indices]
    video_audio_shuffled = np.array(video_audio_matching)[shuffle_indices]
    frame_rgb_shuffled = np.array(frame_rgb)[shuffle_indices]
    frame_audio_shuffled = np.array(frame_audio)[shuffle_indices]
    labels_shuffled = np.array(frame_labels)[shuffle_indices]

    dev_idx = max(1, int(len(labels_shuffled) * validation_split_ratio))

    # delete orig vars to clear some cache
    del video_rgb
    del video_audio
    del frame_rgb
    del frame_audio
    gc.collect()

    train_video_rgb, val_video_rgb = video_rgb_shuffled[:-dev_idx], video_rgb_shuffled[-dev_idx:]
    train_video_audio, val_video_audio = video_audio_shuffled[:-dev_idx], video_audio_shuffled[-dev_idx:]

    train_frame_rgb, val_frame_rgb = frame_rgb_shuffled[:-dev_idx], frame_rgb_shuffled[-dev_idx:]
    train_frame_audio, val_frame_audio = frame_audio_shuffled[:-dev_idx], frame_audio_shuffled[-dev_idx:]

    train_labels, val_labels = labels_shuffled[:-dev_idx], labels_shuffled[-dev_idx:]

    del video_rgb_shuffled, video_audio_shuffled, frame_rgb_shuffled, frame_audio_shuffled, labels_shuffled
    gc.collect()

    return (train_video_rgb, train_video_audio, train_frame_rgb, train_frame_audio, train_labels, val_video_rgb, val_video_audio,
            val_frame_rgb, val_frame_audio, val_labels)

# transform into final input in the model
def one_hot_y(raw_labels,label_size=20):
    '''
    Helper function to transform labels into one-hot TOP 20
    Uses np.unique(return_counts=True) as implicit sorter (first K labels are the most frequent)
    '''
    all_labels = []
    for i in list(raw_labels):
        for j in list(i):
            all_labels.append(j)

    results = np.unique(all_labels,return_counts=True)
    labels_vocab,counts = results

    labels = labels_vocab[:label_size-1] #last columns will be 1 if none of those labels found in a video
    output = []
    for set_of_labels in raw_labels:

        # preallocate numpy arr for each set of labels
        sequence = np.zeros(label_size)
        # loop through all the labels in one video and flip them to 1s
        for this_label in set_of_labels:
            designation = np.where(labels==this_label)
            for des in designation:
                sequence[des]=1
        # done with one training points
        if sequence.sum()==0:
            sequence[-1]=1
        output.append(sequence)
    return output

def transform_data_for_lstm(video_rgb,video_audio, frame_rgb, frame_audio,
                            labels,label_feature_size=10,max_frame_rgb_sequence_length = 10,\
                            max_frame_audio_sequence_length = 10):
    frames = []
    # need to transfrom to numpy (num_videos x max_frame_rgb_sequence_length x 1024)
    #print(len(frame_rgb))

    for frame in frame_rgb:
        # stack the frames in each video, only allowed number of first frams
        #print(np.vstack(frame).shape)
        frames.append(np.vstack(frame)[:max_frame_rgb_sequence_length,:])
    #print(len(frames))

    frames = np.reshape(np.array(frames),(len(frame_rgb),max_frame_rgb_sequence_length,1024))

    #print(frames.shape)

    frames_audio = []
    # need to transfrom to numpy (num_videos x max_frame_audio_sequence_length x 128)
    for frame in frame_audio:
        # stack the frames in each video, only allowed number of first frams
        #print(np.vstack(frame).shape)
        frames_audio.append(np.vstack(frame)[:max_frame_audio_sequence_length,:])

    frames_audio = np.reshape(np.array(frames_audio),(len(frame_audio),max_frame_audio_sequence_length,128))
    #print(frames_audio.shape)

    # deal with videos

    video_rgb = np.vstack(video_rgb)
    video_audio = np.vstack(video_audio)


    # labels - need to one-hot encode TOP - K label
    labels = one_hot_y(labels,label_feature_size)
    labels = np.vstack(labels)
    return frames,frames_audio, video_rgb,video_audio, labels

if __name__ == '__main__':

    # VIDEO-LEVEL DATA PROCESSING
    # note: this will ONLY work with small datasets, for large ones, we need to create iterators
    train_vid_ids, train_labels, train_mean_rgb, train_mean_audio = extract_video_files("mys3bucket/yt8pm_100th_shard/v2/video/train*"
    )

    test_vid_ids, test_labels, test_mean_rgb, test_mean_audio = extract_video_files("mys3bucket/yt8pm_100th_shard/v2/video/test*")

    val_vid_ids, val_labels, val_mean_rgb, val_mean_audio = extract_video_files("mys3bucket/yt8pm_100th_shard/v2/video/validate*")

    # FRAME LEVEL DATA PROCESSING - THE MOST TIME CONSUMING
    # this will cause out-of-memory issues when ran on the WHOLE data set

    train_frame_ids, train_frame_labels,train_frame_rgb,train_frame_audio \
    = extract_frame_level_features("mys3bucket/yt8pm_100th_shard/v2/frame/train*",maximum_iter=True,\
                                   stop_at_iter=100,num_tf_records=3)


    ################
    # CREATE MODEL

    model = create_model()
    ################

    # TRANSFORM DATA TO FIT MODEL ARCHITECTURE

    # split train/validation tests
    train_video_rgb, train_video_audio, train_frame_rgb, train_frame_audio, \
    train_labels, val_video_rgb, val_video_audio, val_frame_rgb, val_frame_audio, val_labels \
                = create_train_dev_dataset(train_mean_rgb, train_mean_audio, train_vid_ids, train_frame_rgb, \
                                            train_frame_audio, train_frame_labels, train_frame_ids )

    # create sequence-like model inputs
    train_frame_rgb, train_frame_audio, train_video_rgb, train_video_audio, train_labels = \
    transform_data_for_lstm(train_video_rgb, train_video_audio, train_frame_rgb, train_frame_audio,train_labels)

    val_frame_rgb, val_frame_audio, val_video_rgb, val_video_audio, val_labels = \
    transform_data_for_lstm( val_video_rgb, val_video_audio,val_frame_rgb, val_frame_audio, val_labels)

    ######
    # TRAIN MODEL
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    STAMP = 'lstm_%d_%d_%.2f_%.2f' % (number_lstm_units, number_dense_units, rate_drop_lstm, rate_drop_dense)

    checkpoint_dir = 'checkpoints/' + str(int(time.time())) + '/'

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    bst_model_path = checkpoint_dir + STAMP + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=False)
    tensorboard = TensorBoard(log_dir=checkpoint_dir + "logs/{}".format(time.time()))

    model.fit([train_frame_rgb, train_frame_audio, train_video_rgb, train_video_audio], train_labels,
              validation_data=([val_frame_rgb, val_frame_audio, val_video_rgb, val_video_audio], val_labels),
              epochs=200, batch_size=1, shuffle=True, callbacks=[early_stopping, model_checkpoint, tensorboard])
