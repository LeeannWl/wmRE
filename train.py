
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from data_loader import data_generator, load_data
import parse as args
from model.dgcnn import DGCNNModel
# from evaluator import Evaluator
import json
from tqdm import tqdm
import numpy as np
from bert4keras.backend import keras
from bert4keras.snippets import open, to_array

# 建立分词器
tokenizer = Tokenizer(args.bert_vocab_path, do_lower_case=True)

class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox





class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    # def __init__(self,tokenizer,optimizer,train_model,object_model,subject_model,dev_data,id2rel):
    def __init__(self, tokenizer, train_model, object_model, subject_model, dev_data, id2rel):
        self.best_val_f1 = 0.
        self.train_model = train_model
        self.dev_data = dev_data
        # self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.object_model = object_model
        self.subject_model = subject_model
        self.id2rel = id2rel

    def extract_spoes(self,text):
        """抽取输入text所包含的三元组
        """
        tokens =  self.tokenizer.tokenize(text, maxlen=args.maxlen)
        mapping =  self.tokenizer.rematch(text, tokens)
        token_ids, segment_ids =  self.tokenizer.encode(text, maxlen=args.maxlen)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        # 抽取subject
        subject_preds = self.subject_model.predict([token_ids, segment_ids])
        start = np.where(subject_preds[0, :, 0] > 0.6)[0]
        end = np.where(subject_preds[0, :, 1] > 0.5)[0]
        subjects = []
        for i in start:
            j = end[end >= i]
            if len(j) > 0:
                j = j[0]
                subjects.append((i, j))
        if subjects:
            spoes = []
            token_ids = np.repeat(token_ids, len(subjects), 0)
            segment_ids = np.repeat(segment_ids, len(subjects), 0)
            subjects = np.array(subjects)
            # 传入subject，抽取object和predicate
            object_preds = self.object_model.predict([token_ids, segment_ids, subjects])
            for subject, object_pred in zip(subjects, object_preds):
                start = np.where(object_pred[:, :, 0] > 0.6)
                end = np.where(object_pred[:, :, 1] > 0.5)
                for _start, predicate1 in zip(*start):
                    for _end, predicate2 in zip(*end):
                        if _start <= _end and predicate1 == predicate2:
                            spoes.append(
                                ((mapping[subject[0]][0],
                                  mapping[subject[1]][-1]), predicate1,
                                 (mapping[_start][0], mapping[_end][-1]))
                            )
                            break
            return [(text[s[0]:s[1] + 1], self.id2rel[p], text[o[0]:o[1] + 1])
                    for s, p, o, in spoes]
        else:
            return []

    def evaluate(self,data):
        """评估函数，计算f1、precision、recall
        """
        X, Y, Z = 1e-10, 1e-10, 1e-10
        f = open(args.predfile, 'w', encoding='utf-8')
        pbar = tqdm()
        for d in data:
            R = set([SPO(spo) for spo in self.extract_spoes(d['text'])])
            T = set([SPO(spo) for spo in d['triple_list']])
            X += len(R & T)
            Y += len(R)
            Z += len(T)
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
            pbar.update()
            pbar.set_description(
                'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
            )
            s = json.dumps({
                'text': d['text'],
                'triple_list': list(T),
                'triple_list_pred': list(R),
                'new': list(R - T),
                'lack': list(T - R),
            },
                ensure_ascii=False,
                indent=4)
            f.write(s + '\n')
        pbar.close()
        f.close()
        return f1, precision, recall



    def on_epoch_end(self, epoch, logs=None):
        # self.optimizer.apply_ema_weights()
        f1, precision, recall = self.evaluate(self.dev_data,)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            self.train_model.save_weights('best_model.weights')
        # self.optimizer.reset_old_weights()
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )





if __name__ == '__main__':
    train_data, dev_data, test_data, id2predicate, predicate2id, num_predicate = load_data(args.train_path,
                                                                                           args.dev_path,
                                                                                           args.test_path,
                                                                                           args.rel_dict_path)
    train_generator = data_generator(train_data, tokenizer, predicate2id, args.maxlen, args.batch_size)
    subject_model, object_model, train_model = DGCNNModel(args.bert_config_path, args.bert_checkpoint_path,num_predicate,args.lr)

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
