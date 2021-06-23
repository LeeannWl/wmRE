bert_model = '../chinese_L-12_H-768_A-12'
bert_config_path = bert_model + '/bert_config.json'
bert_vocab_path = bert_model + '/vocab.txt'
bert_checkpoint_path = bert_model + '/bert_model.ckpt'

dataset = "D:/allDataOfOurProject/origin0603_2"
train = True
train_path =  dataset + '/train_triples.json'
dev_path = dataset + '/dev_triples.json'
test_path = dataset + '/test_triples.json'
rel_dict_path = dataset + '/rel2id.json'
predfile= dataset + 'dev_pred.json'

lr=5e-6
epochs = 100
maxlen = 128
batch_size = 32