## creates model for Elephas training on YouTube-8M

# keras imports
from keras.layers import Dense, Input, LSTM, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.layers.embeddings import Embedding
from keras.layers.merge import concatenate
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import Model

# top 10 classes
label_feature_size = 10

# how many frames we will use from each video?
max_frame_rgb_sequence_length = 100
frame_rgb_embedding_size = 1024
number_dense_units = 1000
number_lstm_units = 100
rate_drop_lstm = 0.2
rate_drop_dense = 0.2
activation_function='relu'

def create_model():
    """Create and store best model at `checkpoint` path ustilising bi-lstm layer for frame level data of videos"""

    #creating input layer for video-level data
    vid_shape=(1024,)
    video_rgb_input = Input(shape=vid_shape)
    video_rgb_dense = Dense(int(number_dense_units/2), activation=activation_function, input_shape=vid_shape)(video_rgb_input)

    # merging video-level dense layer output
    # merged_video = concatenate([video_rgb_dense, video_audio_dense])
    merged_video = BatchNormalization()(video_rgb_dense)
    merged_video = Dropout(rate_drop_dense)(merged_video)
    merged_video_dense = Dense(int(number_dense_units/2), activation=activation_function)(merged_video)


    # merging frame-level and video-level dense layer output
    merged = merged_video_dense # concatenate([merged_frame_dense, merged_video_dense])
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
