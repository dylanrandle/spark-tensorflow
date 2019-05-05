import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import seaborn as sns
from IPython.display import YouTubeVideo
import matplotlib.pyplot as plt
import plotly.plotly as py
import multiprocessing as mp # if we want to parallelize i/o

# keras imports
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model
import operator
import time 
import gc
import os

import os
from glob import glob
from tqdm import tqdm
import sys
import timeit


from pyspark import SparkConf, SparkContext
from pyspark.sql.types import *
import pyspark
import numpy as np
from elephas.spark_model import SparkModel

###########
#SPECIFY PARAMS FIRST
###########

# 1000 class problem for now?
label_feature_size = 1000

# how many frames we will use from each video?
max_frame_rgb_sequence_length = 50
frame_rgb_embedding_size = 1024

# how many audio sequences we will use from each video?
max_frame_audio_sequence_length = 50
frame_audio_embedding_size = 128

number_dense_units = 1000
number_lstm_units = 100
rate_drop_lstm = 0.2
rate_drop_dense = 0.2
activation_function='relu'
validation_split_ratio = 0 # to use all

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

def extract_frame_level_features_per_tf_record(frame_file_path,maximum_iter = False,stop_at_iter = 10):
    '''
    Extraction of Youtube tfrecords frame file features.
    
    Args: 
    path to each tf_record (note: developed with assumption of storing on s3 bucket and assessing with glob)
    
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
    print(f'There is {sum(1 for _ in tf.python_io.tf_record_iterator(frame_file_path))} videos in this TF record.')
    iter_ = 0
    for example in tf.python_io.tf_record_iterator(frame_file_path):
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

def create_pandas(frame_rgb, frame_audio, video_rgb, video_audio, labels):
    '''
    Spark unfortunately does not work with numpy arrays - so we need to convert to traditional python types.
    '''
    return pd.DataFrame.from_dict({'frame_rgb':[[[float(k) for k in j] for j in i] for i in frame_rgb],\
                             'frame_audio':[[[float(k) for k in j] for j in i] for i in frame_audio],\
                             'mean_rgb':[[float(j) for j in i] for i in list(video_rgb)],\
                            'mean_audio':[[float(j) for j in i] for i in list(video_audio)],\
                            'labels':[[float(j) for j in i] for i in list(labels)]})


def convert_data(train_frame_shards):
    
    NUM_RECORDS_TO_LOAD = -1
    
    for tf_record in tqdm(train_frame_shards[:NUM_RECORDS_TO_LOAD]):
        # pull frames in memory
        try:
            train_frame_ids, train_frame_labels,train_frame_rgb,train_frame_audio \
                = extract_frame_level_features_per_tf_record(tf_record,maximum_iter=False,\
                                       stop_at_iter=5) # just pull 10 videos from each tf record for debugging
            # first transformation
            train_video_rgb, train_video_audio, train_frame_rgb, train_frame_audio, \
            train_labels, val_video_rgb, val_video_audio, val_frame_rgb, val_frame_audio, val_labels \
                        = create_train_dev_dataset(train_mean_rgb, train_mean_audio, train_vid_ids, train_frame_rgb, \
                                                    train_frame_audio, train_frame_labels, train_frame_ids )    

            # final transformation for LSTM
            train_frame_rgb, train_frame_audio, train_video_rgb, train_video_audio, train_labels = \
                    transform_data_for_lstm(train_video_rgb, train_video_audio, train_frame_rgb, train_frame_audio,train_labels)

            val_frame_rgb, val_frame_audio, val_video_rgb, val_video_audio, val_labels = \
                    transform_data_for_lstm( val_video_rgb, val_video_audio,val_frame_rgb, val_frame_audio, val_labels)

            #### BELOW WE ONLY USE THE TRAINING DATA AND NO VALIDATION DATA FOR SIMPLICITY
            df = create_pandas(train_frame_rgb, train_frame_audio, \
                           train_video_rgb, train_video_audio, train_labels)
            # create spark data frame

            df_spark = spark.createDataFrame(df)        

            path = f"{str(tf_record.split('/')[-1])}-converted.tfrecord"
            df_spark.write.format("tfrecords").option("recordType", "SequenceExample").save(path)
            del df_spark
            gc.collect()
            #print(path)
            os.system(f'sudo mv {path} mys3bucket/converted_records_for_spark/')
        except:
            continue # skip a tf record if something goes wrong

if __name__ == '__main__':

    conf = SparkConf().setAppName('Youtube-8M') \
                  .set("spark.jars",
                       "ecosystem/spark/spark-tensorflow-connector/target/spark-tensorflow-connector_2.11-1.10.0.jar")
    sc = SparkContext(conf = conf)
    spark = pyspark.sql.SparkSession(sc)
    # We just load all of the video data in memory since it is fairly small and manageable.
    train_vid_ids, train_labels, train_mean_rgb, train_mean_audio \
        = extract_video_files("mys3bucket/yt8pm_100th_shard/v2/video/train*")    
    # the fun starts here, pull frame data per tf-record
    train_frame_shards = glob('mys3bucket/yt8pm_100th_shard/v2/frame/train*')
    train_frame_shards.remove('mys3bucket/yt8pm_100th_shard/v2/frame/train0093.tfrecord')
    train_frame_shards.remove('mys3bucket/yt8pm_100th_shard/v2/frame/train0111.tfrecord')
    train_frame_shards.remove('mys3bucket/yt8pm_100th_shard/v2/frame/train0208.tfrecord')
    print(train_frame_shards)
    convert_data(train_frame_shards[1:]) 

