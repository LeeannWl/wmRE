
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from data_loader import data_generator, load_data
import parse as args
from evaluator import Evaluator
from model.dgcnn import DGCNNModel
from model.e2emodel import E2EModel
# from evaluator import Evaluator
import json
from tqdm import tqdm
import numpy as np
from bert4keras.backend import keras
from bert4keras.snippets import open, to_array







if __name__ == '__main__':
    # 建立分词器
    tokenizer = Tokenizer(args.bert_vocab_path, do_lower_case=True)
    train_data, dev_data, test_data, id2predicate, predicate2id, num_predicate = load_data(args.train_path,
                                                                                           args.dev_path,
                                                                                           args.test_path,
                                                                                           args.rel_dict_path)
    train_generator = data_generator(train_data, tokenizer, predicate2id, args.maxlen, args.batch_size)
    # subject_model, object_model, train_model = DGCNNModel(args.bert_config_path, args.bert_checkpoint_path,args.lr,num_predicate)
    subject_model, object_model, train_model = E2EModel(args.bert_config_path, args.bert_checkpoint_path, args.lr,
                                                          num_predicate)
    # optimizer = extend_with_exponential_moving_average(Adam, name='AdamEMA')(args.lr)
    # train_model.compile(optimizer=optimizer)
    # evaluator = Evaluator(tokenizer, optimizer, train_model, object_model, subject_model, dev_data, id2predicate)
    evaluator = Evaluator(tokenizer, train_model, object_model, subject_model, dev_data, id2predicate)
    train_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=args.epochs,
        callbacks=[evaluator]
    )
