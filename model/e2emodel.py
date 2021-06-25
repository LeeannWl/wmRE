#! -*- coding:utf-8 -*-
from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from keras.optimizers import Adam
from keras.regularizers import l2
# from keras_bert import load_trained_model_from_checkpoint
from bert4keras.models import build_transformer_model
from utils import seq_gather, extract_items, metric
from tqdm import tqdm
import numpy as np


def E2EModel(bert_config_path, bert_checkpoint_path, LR, num_rels):
    bert_model = build_transformer_model(bert_config_path, bert_checkpoint_path,seq_len=None)
    for l in bert_model.layers:
        l.trainable = True

    tokens_in = Input(shape=(None,))
    segments_in = Input(shape=(None,))
    gold_sub_heads_in = Input(shape=(None,))
    gold_sub_tails_in = Input(shape=(None,))
    sub_head_in = Input(shape=(1,))
    sub_tail_in = Input(shape=(1,))
    gold_obj_heads_in = Input(shape=(None, num_rels))
    gold_obj_tails_in = Input(shape=(None, num_rels))

    tokens, segments, gold_sub_heads, gold_sub_tails, sub_head, sub_tail, \
    gold_obj_heads, gold_obj_tails = tokens_in, segments_in, gold_sub_heads_in, \
                                     gold_sub_tails_in, sub_head_in, sub_tail_in, \
                                     gold_obj_heads_in, gold_obj_tails_in
    mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0),
                                   'float32'))(tokens)

    tokens_feature = bert_model([tokens, segments])
    pred_sub_heads = Dense(1, activation='sigmoid')(tokens_feature)
    pred_sub_tails = Dense(1, activation='sigmoid')(tokens_feature)

    subject_model = Model([tokens_in, segments_in], [pred_sub_heads, pred_sub_tails])

    sub_head_feature = Lambda(seq_gather)([tokens_feature, sub_head])
    sub_tail_feature = Lambda(seq_gather)([tokens_feature, sub_tail])
    sub_feature = Average()([sub_head_feature, sub_tail_feature])

    tokens_feature = Add()([tokens_feature, sub_feature])
    pred_obj_heads = Dense(num_rels, activation='sigmoid')(tokens_feature)
    pred_obj_tails = Dense(num_rels, activation='sigmoid')(tokens_feature)

    object_model = Model([tokens_in, segments_in, sub_head_in, sub_tail_in],
                         [pred_obj_heads, pred_obj_tails])

    hbt_model = Model([tokens_in, segments_in, gold_sub_heads_in, gold_sub_tails_in,
                       sub_head_in, sub_tail_in, gold_obj_heads_in, gold_obj_tails_in],
                      [pred_sub_heads, pred_sub_tails, pred_obj_heads, pred_obj_tails])

    gold_sub_heads = K.expand_dims(gold_sub_heads, 2)
    gold_sub_tails = K.expand_dims(gold_sub_tails, 2)

    sub_heads_loss = K.binary_crossentropy(gold_sub_heads, pred_sub_heads)
    sub_heads_loss = K.sum(sub_heads_loss * mask) / K.sum(mask)
    sub_tails_loss = K.binary_crossentropy(gold_sub_tails, pred_sub_tails)
    sub_tails_loss = K.sum(sub_tails_loss * mask) / K.sum(mask)

    obj_heads_loss = K.sum(K.binary_crossentropy(gold_obj_heads, pred_obj_heads),
                           2, keepdims=True)
    obj_heads_loss = K.sum(obj_heads_loss * mask) / K.sum(mask)
    obj_tails_loss = K.sum(K.binary_crossentropy(gold_obj_tails, pred_obj_tails),
                           2, keepdims=True)
    obj_tails_loss = K.sum(obj_tails_loss * mask) / K.sum(mask)

    loss = (sub_heads_loss + sub_tails_loss) + (obj_heads_loss + obj_tails_loss)

    hbt_model.add_loss(loss)
    hbt_model.compile(optimizer=Adam(LR))
    hbt_model.summary()

    return subject_model, object_model, hbt_model