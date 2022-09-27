import torch
import numpy as np
import random
import os
from model import Transformer
from utils import get_data_iter, id_to_sentence, save_obj, load_obj
import torchtext
import nltk
from tqdm import tqdm
import pickle


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    device = torch.device('cuda')
    batch_size = 256
    max_len = 70
    min_freq = 2
    epoch = 30
    train_data_len = 1000000
    # src_vocab = load_obj('src_vocab_1000000.pkl')
    # tgt_vocab = load_obj('tgt_vocab_1000000.pkl')
    # 读取数据
    src_vocab, tgt_vocab, train_iter = get_data_iter(task='train', batch_size=batch_size,
                                                     data_len=train_data_len, max_len=max_len, min_freq=min_freq)
    _, _, valid_iter = get_data_iter(task='valid', batch_size=batch_size, train_src_vocab=src_vocab,
                                     data_len=50000, train_tgt_vocab=tgt_vocab, max_len=max_len, min_freq=min_freq)
    _, _, test_iter = get_data_iter(task='test', batch_size=batch_size, train_src_vocab=src_vocab,
                                    data_len=None, train_tgt_vocab=tgt_vocab, max_len=max_len, min_freq=min_freq)
    # 保存字典
    save_obj('src_vocab_{}.pkl'.format(train_data_len), src_vocab)
    save_obj('tgt_vocab_{}.pkl'.format(train_data_len), tgt_vocab)

    # 初始化模型并训练
    model = Transformer(d_model=512, n_head=8, num_encoder_layers=3, num_decoder_layers=3,
                        dim_feedforward=1024, src_vocab=src_vocab, tgt_vocab=tgt_vocab)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[7, 14], gamma=0.1)

    model.fit(train_iter, valid_iter, epoch, optimizer, scheduler, verbose=False)
    # model.load_model('model_30.pt')

    # 将前50条翻译结果写入文件
    with open('translate_50.txt', 'w') as f:
        for i, batch in enumerate(test_iter):
            out = model.greedy_decode(batch.src[:50, :].cuda(), max_len)
            x = id_to_sentence(batch.src[:50, :].transpose(0, 1), src_vocab, is_english=True)
            t = id_to_sentence(batch.tgt[:50, :].transpose(0, 1), tgt_vocab)
            y = id_to_sentence(out.transpose(0, 1), tgt_vocab)
            for i in range(len(x)):
                print('第{}条：'.format(i))
                print('SRC: ' + x[i])
                print('TGT: ' + t[i])
                print('predict: ' + y[i])
                print()
                f.write(y[i] + '\n')
            break

    # 预测测试集并计算bleu
    bleu = model.predict_bleu(test_iter, max_len)
    print(bleu)

    print(model.translate('I want to eat an apple'), max_len)
