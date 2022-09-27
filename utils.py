import torch
import numpy as np
import random
import os
import torchtext
import nltk
from tqdm import tqdm
import pickle


def set_random_seed(seed_value=1024):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True


def get_data_iter(task, batch_size, data_len=None, begin_word='<BOS>',
                  end_word='<EOS>', padding_word='<pad>', max_len=100,
                  min_freq=2, train_src_vocab=None, train_tgt_vocab=None):
    # 读取数据
    with open('data/{}_en'.format(task), 'r', encoding='utf-8') as en_file:
        en_list = en_file.readlines()
    with open('data/{}_cn'.format(task), 'r', encoding='utf-8') as cn_file:
        cn_list = cn_file.readlines()
    assert len(en_list) == len(cn_list), 'length does not match'
    # 构建Field
    SRC = torchtext.data.Field(tokenize=lambda x: nltk.word_tokenize(x),
                               pad_token=padding_word, lower=True,
                               batch_first=True)
    TGT = torchtext.data.Field(tokenize=lambda x: list(x), init_token=begin_word,
                               eos_token=end_word, pad_token=padding_word,
                               batch_first=True)
    # 减少数据集长度
    if data_len is not None and data_len < len(en_list):
        en_list = en_list[:data_len]
        cn_list = cn_list[:data_len]

    # 构建Examples
    fields = [('src', SRC), ('tgt', TGT)]
    examples = []
    for en_text, cn_text in tqdm(zip(en_list, cn_list),
                                 total=len(en_list),
                                 desc=task):
        examples.append(
            torchtext.data.Example.fromlist([en_text, cn_text], fields))
    # 构建Dataset
    data = torchtext.data.Dataset(examples, fields,
                                  filter_pred=lambda x:
                                  len(vars(x)['src']) <= max_len and
                                  len(vars(x)['tgt']) <= max_len)
    # 构建字典
    if train_src_vocab is not None:
        SRC.vocab = train_src_vocab
        TGT.vocab = train_tgt_vocab
    else:
        SRC.build_vocab(data.src, min_freq=min_freq)
        TGT.build_vocab(data.tgt, min_freq=min_freq)

    # 每次迭代中src维度为(batch_size, src_text_len), tgt维度为(batch_size, tgt_text_len)
    data_iter = torchtext.data.BucketIterator(data, batch_size=batch_size,
                                              sort_key=lambda x: len(x.src),
                                              shuffle=(task != 'test'))
    return SRC.vocab, TGT.vocab, data_iter


def id_to_sentence(x, vocab, is_english=False):
    # x: (length, batch_size)
    res = []
    for i in range(x.size(1)):
        sentence = []
        for id in x[:, i]:
            word = vocab.itos[id]
            if word == '<BOS>':
                continue
            if word == '<EOS>' or word == '<pad>':
                break
            sentence.append(word)
        if is_english:
            res.append(' '.join(sentence))
        else:
            res.append(''.join(sentence))
    return res


def id_to_word(tgt, vocab):
    # tgt: (batch_size, length)
    res = []
    for i in range(tgt.size(0)):
        sentence = []
        for id in tgt[i, :]:
            word = vocab.itos[id]
            if word == '<BOS>':
                continue
            if word == '<EOS>' or word == '<pad>':
                break
            sentence.append(word)
        res.append(sentence)
    return res


def save_obj(path, obj):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_obj(path):
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    return obj