UNK = '<unk>'
PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'


# corpus
# 文本长
MAX_SOURCE_LEN = 3000
MAX_TARGET_LEN = 50

# 词表大小
VOCAB_SIZE = 80000

N_EPOCHS = 300
BATCH_SIZE = 100

EMBED_SIZE = 768
MODEL_DIM = 768

LR = 3e-5

HEAD_NUM = 8
LAYER_NUM = 6
FFN_DIM = 2048

DROPOUT = 0.1
ATT_DROPOUT = 0.1
FFN_DROPOUT = 0.1

VOCAB_PATH = '../data/vocab.pkl'
TRAIN_DATA_PATH = '../data/patent.tsv'
