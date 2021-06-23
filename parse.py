bert_model = '/root/kg/bert/chinese_L-12_H-768_A-12'
bert_config_path = bert_model + '/bert_config.json'
bert_vocab_path = bert_model + '/vocab.txt'
bert_checkpoint_path = bert_model + '/bert_model.ckpt'

dataset = "origin0603"
train = True
train_path = '../data/' + dataset + '/train_triples.json'
dev_path = '../data/' + dataset + '/dev_triples.json'
test_path = '../data/' + dataset + '/test_triples.json'
rel_dict_path = '../data/' + dataset + '/rel2id.json'
predfile= '../data/' + dataset + 'dev_pred.json'

lr=5e-6
epochs = 100
maxlen = 128
batch_size = 128