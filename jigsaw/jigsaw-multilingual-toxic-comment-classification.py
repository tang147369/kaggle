# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from collections import Counter
import transformers
import tensorflow_hub as hub


# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None
if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)

def get_x_data_array(x_data):
    x_data_list = []
    for i in range(len(x_data)):
        if type(x_data[i]) != type('str'):
            print(type(x_data[i]), x_data[i])
        x_data[i] = x_data[i].replace('(', '').replace(')', '').replace(' ', '')
        temp = np.array(x_data[i].split(','))
        temp = list(map(int,temp))
        temp = np.array(temp)
        x_data_list.append(temp)
    x_data_array = np.array(x_data_list)
    return x_data_array

def data_preprocessing(jigsaw_toxic_comment_df):
    input_word_ids = jigsaw_toxic_comment_df['input_word_ids'].values
    input_mask = jigsaw_toxic_comment_df['input_mask'].values
    all_segment_id = jigsaw_toxic_comment_df['all_segment_id'].values
    train_labels = jigsaw_toxic_comment_df['toxic'].values
    
    print('input_word_ids_array')
    input_word_ids_array = get_x_data_array(input_word_ids)
    print('input_mask')
    input_mask_array = get_x_data_array(input_mask)
    print('all_segment_id')
    all_segment_id = get_x_data_array(all_segment_id)
    print('input_word_ids_array.shape', input_word_ids_array.shape)
    print(input_mask_array.shape)
    print(all_segment_id.shape)
    return input_word_ids_array, input_mask_array, all_segment_id, train_labels

# def get_training_model():
    
#     bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/2",
#                             trainable=True)
#     max_seq_length = 128
#     input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
#                                            name="input_word_ids")
#     input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
#                                        name="input_mask")
#     segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32,
#                                         name="segment_ids")
#     inputs = [input_word_ids, input_mask, segment_ids]
#     pooled_output, sequence_output = bert_layer(inputs)
#     print(pooled_output, sequence_output)
#     dense1 = tf.keras.layers.Dense(128, activation='relu')(pooled_output)
#     dropout1 = tf.keras.layers.Dropout(0.5)(dense1)
#     output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout1)
#     model = tf.keras.Model(inputs=inputs, outputs=output)
#     model.compile(optimizer=tf.optimizers.Adam(learning_rate=2e-5, epsilon=1e-08), 
#             loss='binary_crossentropy', metrics=['accuracy'])
#     print(model.summary())    
#     return model
    
def get_training_model():
    inp_id = tf.keras.layers.Input(shape=(128,), dtype=tf.int64, name="input_ids")
    inp_mask = tf.keras.layers.Input(shape=(128,), dtype=tf.int64, name="input_masks")
    inputs = [inp_id, inp_mask]
    
    last_hidden_state = transformers.TFDistilBertModel.from_pretrained('distilbert-base-multilingual-cased')(inputs)[0]
#     last_hidden_state, pooler_output = transformers.TFBertModel.from_pretrained('bert-base-uncased')(inputs)
#     print(pooler_output)
    print(last_hidden_state)
    pooled_output = last_hidden_state[:, 0]
    print(pooled_output)
    dense1 = tf.keras.layers.Dense(128, activation='relu', bias_regularizer=tf.keras.regularizers.l2(0.5))(pooled_output) #0.0002
    dropout1 = tf.keras.layers.Dropout(0.5)(dense1)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(dropout1)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer=tf.optimizers.Adam(learning_rate=2e-5, 
                                            epsilon=1e-08), 
                loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def get_data(file_name, columns):
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
             file_path = os.path.join(dirname, filename)
             if filename == file_name:
                jigsaw_toxic_comment_train = pd.read_csv(file_path)
                jigsaw_toxic_comment_train_df = jigsaw_toxic_comment_train[columns]
                return jigsaw_toxic_comment_train_df
            
if __name__ == '__main__':
    
    jigsaw_toxic_comment_train_df = get_data('jigsaw-toxic-comment-train-processed-seqlen128.csv', ['input_word_ids', 'input_mask', 'all_segment_id', 'toxic'])
    print('jigsaw_toxic_comment_train_df: ', jigsaw_toxic_comment_train_df.shape)
    print(jigsaw_toxic_comment_train_df.query('toxic==0').shape)
    print(jigsaw_toxic_comment_train_df.query('toxic==1').shape)
#     jigsaw_unintended_bias_train_df = get_data('jigsaw-unintended-bias-train-processed-seqlen128.csv', ['input_word_ids', 'input_mask', 'all_segment_id', 'toxic'])
#     print(jigsaw_unintended_bias_train_df.query('toxic==0').shape)
#     print(jigsaw_unintended_bias_train_df.query('toxic==1').shape)
#     jigsaw_toxic_comment_train_df = pd.concat([jigsaw_toxic_comment_train_df,
#                                               jigsaw_unintended_bias_train_df.query('toxic==0').sample(n=100000, random_state=0),
#                                               jigsaw_unintended_bias_train_df.query('toxic==1'),
#                                               ],axis=0,ignore_index=True)
    train_input_ids, train_input_masks, train_segment_ids, train_labels = data_preprocessing(jigsaw_toxic_comment_train_df)
    with strategy.scope():
        model = get_training_model()
    model.fit([train_input_ids, train_input_masks], train_labels, batch_size=64, epochs=1) #[0:223520]
    # validation
    validation_df = get_data('validation-processed-seqlen128.csv', ['input_word_ids', 'input_mask', 'all_segment_id', 'toxic'])
    print('validation_df', validation_df.query('toxic==0').shape)
    print(validation_df.query('toxic==1').shape)
    validation_input_ids, validation_input_masks, validation_segment_ids, validation_labels = data_preprocessing(validation_df)
    results = model.evaluate([validation_input_ids, validation_input_masks], validation_labels)
    print('results', results)
    
    # test predict
    test_jigsaw_toxic_comment_data_df = get_data('test-processed-seqlen128.csv', ['id', 'input_word_ids', 'input_mask', 'all_segment_id'])
    test_input_word_ids = test_jigsaw_toxic_comment_data_df['input_word_ids'].values
    test_input_mask = test_jigsaw_toxic_comment_data_df['input_mask'].values
    test_all_segment_id = test_jigsaw_toxic_comment_data_df['all_segment_id'].values
    
    x_test_id = test_jigsaw_toxic_comment_data_df['id'].values
    
    test_input_word_ids_array = get_x_data_array(test_input_word_ids)
    test_input_mask_array = get_x_data_array(test_input_mask)
    test_all_segment_id_array = get_x_data_array(test_all_segment_id)
    
    list_id_predict = []
    y_predict = model.predict([test_input_word_ids_array, test_input_mask_array])
    print('y_predict', y_predict[0:10])
    print(y_predict.shape, x_test_id.shape)
    predicted_result = list(zip(x_test_id, y_predict))
    
    # to_csv
    df = pd.DataFrame(predicted_result, columns=['id', 'toxic'])
    df.to_csv('submission.csv', index=None)
    df = pd.read_csv('submission.csv')
    df.columns = ['id','toxic']
    df = df.replace('[\\[\\]]', '', regex=True)
    print(df)
    df.to_csv('submission.csv', index=None)
    print('y_predict.flatten()', Counter(y_predict.flatten()))