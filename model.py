import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import nltk
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu
from layers import EncoderLayer, Encoder, DecoderLayer, Decoder, PositionalEncoding
from utils import id_to_sentence, id_to_word


class Transformer(nn.Module):
    def __init__(self, d_model, n_head, num_encoder_layers, num_decoder_layers,
                 dim_feedforward, src_vocab, tgt_vocab, dropout=0.1):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_vocab_size = len(src_vocab)
        self.tgt_vocab_size = len(tgt_vocab)
        self.src_padding_id = src_vocab.stoi['<pad>']
        self.tgt_padding_id = tgt_vocab.stoi['<pad>']
        self.tgt_begin_id = tgt_vocab.stoi['<BOS>']

        # encoder and decoder
        encoder_layer = EncoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.encoder = Encoder(encoder_layer, num_encoder_layers)
        decoder_layer = DecoderLayer(d_model, n_head, dim_feedforward, dropout)
        self.decoder = Decoder(decoder_layer, num_decoder_layers)

        # embedding and positional encoding
        self.src_embed = nn.Embedding(self.src_vocab_size, d_model, padding_idx=self.src_padding_id)
        self.tgt_embed = nn.Embedding(self.tgt_vocab_size, d_model, padding_idx=self.tgt_padding_id)
        self.src_pos = PositionalEncoding(d_model)
        self.tgt_pos = PositionalEncoding(d_model)
        self.fc = nn.Linear(d_model, self.tgt_vocab_size)

        self.best_epoch = 0
        self.best_valid_loss = float('inf')

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, src, tgt):
        memory, src_padding_mask = self.encode(src)
        output = self.decode(tgt, memory, src_padding_mask)
        output = self.fc(output)
        return output

    def encode(self, src):
        # src_padding_mask: (batch_size, src_len)
        src_padding_mask = Transformer.get_padding_mask(src, self.src_padding_id)
        # (batch_size, src_len) -> (src_len, batch_size, d_model)
        src_embed = self.src_pos(self.src_embed(src) * math.sqrt(self.d_model)).transpose(0, 1)
        return self.encoder(src_embed, src_padding_mask), src_padding_mask

    def decode(self, tgt, memory, src_padding_mask):
        tgt_padding_mask = Transformer.get_padding_mask(tgt, self.tgt_padding_id)
        tgt_subsequent_mask = Transformer.get_square_subsequent_mask(tgt.size(1), tgt.device)
        tgt_embed = self.tgt_pos(self.tgt_embed(tgt) * math.sqrt(self.d_model)).transpose(0, 1)
        return self.decoder(tgt_embed, memory, src_padding_mask, tgt_padding_mask, tgt_subsequent_mask)

    def translate(self, sentence, max_len):
        self.eval()
        sentence = nltk.word_tokenize(sentence)
        sentence = [self.src_vocab.stoi[x] for x in sentence]
        input = torch.LongTensor(sentence).unsqueeze(0).cuda()
        out = self.greedy_decode(input, max_len)
        return id_to_sentence(out.transpose(0, 1), self.tgt_vocab)

    def predict_bleu(self, test_iter, max_len):
        self.eval()
        bleu = [0.0, 0.0, 0.0, 0.0]
        for i, batch in tqdm(enumerate(test_iter), desc='test',
                             total=math.ceil(len(test_iter.dataset)/test_iter.batch_size)):
            out = self.greedy_decode(batch.src.cuda(), max_len)
            y_pred = id_to_word(out, self.tgt_vocab)
            y_true = id_to_word(batch.tgt, self.tgt_vocab)
            for i in range(len(y_pred)):
                bleu[0] += sentence_bleu([y_true[i]], y_pred[i], weights=(1, 0, 0, 0))
                bleu[1] += sentence_bleu([y_true[i]], y_pred[i], weights=(0, 1, 0, 0))
                bleu[2] += sentence_bleu([y_true[i]], y_pred[i], weights=(0, 0, 1, 0))
                bleu[3] += sentence_bleu([y_true[i]], y_pred[i], weights=(0, 0, 0, 1))
        bleu = [x / len(test_iter.dataset) for x in bleu]
        print(bleu)
        return bleu

    def greedy_decode(self, src, max_len):
        self.eval()
        with torch.no_grad():
            memory, src_padding_mask = self.encode(src)
            dec_input = torch.ones(src.size(0), 1, dtype=src.dtype, device=src.device).fill_(self.tgt_begin_id)
            for i in range(max_len - 1):
                # y_prob: (batch_size, vocab_size)
                y_prob = F.softmax(self.fc(self.decode(dec_input, memory, src_padding_mask)[-1, :, :]), dim=-1)
                _, y_pred = torch.max(y_prob, dim=1)
                # next_word: (batch_size, 1)
                next_word = y_pred.unsqueeze(1)
                dec_input = torch.cat([dec_input, next_word], dim=1)
            return dec_input

    def __update_best_param(self, loss, current_epoch):
        if loss < self.best_valid_loss:
            self.best_epoch = current_epoch
            self.best_valid_loss = loss
            return True
        else:
            return False

    def __evaluate_dataset(self, valid_iter):
        self.eval()
        valid_loss = 0.0
        len_valid = len(valid_iter.dataset)
        with torch.no_grad():
            for i, batch in tqdm(enumerate(valid_iter), desc='valid',
                                 total=math.ceil(len_valid / valid_iter.batch_size)):
                dec_input = batch.tgt[:, :-1].cuda()
                y_true = batch.tgt[:, 1:].transpose(0, 1).cuda()
                # (tgt_len-1, batch_size, vocab_size)
                output = self.forward(batch.src.cuda(), dec_input)
                loss = self.__calculate_loss(output.contiguous().view(-1, output.size(-1)), y_true.contiguous().view(-1))
                valid_loss += float(loss) * batch.src.size(0) / len_valid
                if i == 0:
                    print('valid:')
                    _, y_pred = torch.max(output.detach()[:, :2, :], dim=2)
                    print(id_to_sentence(y_pred, self.tgt_vocab))
                    print(id_to_sentence(y_true[:, :2], self.tgt_vocab))
        self.train()
        return valid_loss

    def __calculate_loss(self, output, tgt, smoothing=0.1):
        # output: (tgt_len * batch_size, vocab_size)
        # tgt: (tgt_len * batch_size)
        # one-hot编码与标签平滑
        confidence = 1.0 - smoothing
        tgt_one_hot = torch.zeros_like(output)
        # 这里除将每个负例单词填充为smoothing / (vocab_size - 2), 因为下面将<pad>设为0, 因此不是vocab_size - 1
        tgt_one_hot.fill_(smoothing / (output.size(1) - 2))
        tgt_one_hot.scatter_(1, tgt.data.unsqueeze(1), confidence)

        # 将每个词向量中的<pad>对应位设为0
        tgt_one_hot[:, self.tgt_padding_id] = 0
        # 将填充单词对应的词向量全部设为0
        mask = torch.nonzero(tgt.data == self.tgt_padding_id)
        if mask.dim() > 0:
            tgt_one_hot.index_fill_(0, mask.squeeze(), 0.0)
        del mask
        # 交叉熵
        # output = F.log_softmax(output, dim=-1)
        # loss = torch.mean(torch.sum(output * tgt_one_hot * -1, dim=-1))
        torch.cuda.empty_cache()
        # KL散度
        output = F.log_softmax(output, dim=-1)
        kl_div = nn.KLDivLoss(reduction='batchmean')
        loss = kl_div(output, tgt_one_hot)
        return loss

    def fit(self, train_iter, valid_iter, epoch, optimizer, scheduler=None, verbose=False):
        self.train()
        self.best_epoch = 0
        self.best_valid_loss = float('inf')
        for e in range(1, epoch+1):
            for i, batch in tqdm(enumerate(train_iter),
                                 total=math.ceil(len(train_iter.dataset)/train_iter.batch_size),
                                 desc='epoch {}'.format(e)):
                dec_input = batch.tgt[:, :-1].cuda()
                y_true = batch.tgt[:, 1:].transpose(0, 1).cuda()
                # (tgt_len-1, batch_size, vocab_size)
                torch.cuda.empty_cache()
                output = self.forward(batch.src.cuda(), dec_input)
                torch.cuda.empty_cache()
                loss = self.__calculate_loss(output.contiguous().view(-1, output.size(-1)), y_true.contiguous().view(-1))
                torch.cuda.empty_cache()
                self.zero_grad()
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()
                if verbose:
                    print('iter {}: {}'.format(i, float(loss)))
                if i == 0:
                    print('train:')
                    _, y_pred = torch.max(output.detach()[:, :2, :], dim=2)
                    print(id_to_sentence(y_pred, self.tgt_vocab))
                    print(id_to_sentence(y_true[:, :2], self.tgt_vocab))
                    print('iter {}: {}'.format(i, float(loss)))
            if scheduler is not None:
                scheduler.step()
            valid_loss = self.__evaluate_dataset(valid_iter)
            print('Epoch {}: valid loss: {}'.format(e, valid_loss))
            # 更新参数时 以及 每5轮 保存一次模型
            if self.__update_best_param(valid_loss, e) or e % 5 == 0:
                self.save_model('model_{}.pt'.format(e))
        # 恢复模型参数为验证集loss最小的参数
        self.load_model('model_{}.pt'.format(self.best_epoch))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    @staticmethod
    def get_padding_mask(x, padding_id):
        # (batch_size, src_len)
        return x.data.eq(padding_id)

    @staticmethod
    def get_square_subsequent_mask(size, device):
        mask = ((torch.triu(torch.ones(size, size, device=device), diagonal=1)) == 1)
        # (tgt_len, tgt_len)
        return mask
