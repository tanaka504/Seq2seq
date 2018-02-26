import numpy as np
import chainer
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import MeCab
import pickle
import os
from utils import *

class LSTM_Encoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LSTM_Encoder, self).__init__(
            xe = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh = links.Linear(embed_size, 4 * hidden_size),
            hh = links.Linear(hidden_size, 4 * hidden_size)
        )

    def __call__(self, x, c, h):
        e = functions.tanh(self.xe(x))
        return functions.lstm(c, self.eh(e) + self.hh(h))

class LSTM_Decoder(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LSTM_Decoder, self).__init__(
            ye = links.EmbedID(vocab_size, embed_size, ignore_label=-1),
            eh = links.Linear(embed_size, 4 * hidden_size),
            hh = links.Linear(hidden_size, 4 * hidden_size),
            he = links.Linear(hidden_size, embed_size),
            ey = links.Linear(embed_size, vocab_size)
        )

    def __call__(self, y, c, h):
        e = functions.tanh(self.ye(y))
        c, h = functions.lstm(c, self.eh(e) + self.hh(h))
        t = self.ey(functions.tanh(self.he(h)))
        return t, c, h

class Seq2Seq(Chain):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, flag_gpu=True):
        super(Seq2Seq, self).__init__(
            encoder=LSTM_Encoder(vocab_size, embed_size, hidden_size),
            decoder=LSTM_Decoder(vocab_size, embed_size, hidden_size)
        )
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        if flag_gpu:
            self.ARR = cuda.cupy
        else:
            self.ARR = np

    def encode(self, input_batch):
        c = Variable(self.ARR.zeros((self.batch_size,self.hidden_size),dtype='float32'))
        h = Variable(self.ARR.zeros((self.batch_size,self.hidden_size),dtype='float32'))
        for batch_word in input_batch:
            batch_word = Variable(self.ARR.array(batch_word,dtype='int32'))
            c, h = self.encoder(batch_word, c, h)
        self.h = h
        self.c = Variable(self.ARR.zeros((self.batch_size,self.hidden_size),dtype='float32'))

    def decode(self, predict_id, teacher_id, train):
        batch_word = Variable(self.ARR.array(predict_id,dtype='int32'))
        predict_mat, self.c, self.h = self.decoder(batch_word, self.c, self.h)
        if train:
            t = Variable(self.ARR.array(teacher_id, dtype='int32'))
            return functions.softmax_cross_entropy(predict_mat, t), predict_mat
        else:
            return predict_mat

    def reset(self):
        self.h = Variable(self.ARR.zeros((self.batch_size,self.hidden_size),dtype='float32'))
        self.c = Variable(self.ARR.zeros((self.batch_size,self.hidden_size),dtype='float32'))

        self.cleargrads()

    def test_encode(self,words):
        for w in words:
            w = Variable(self.ARR.array([w],dtype='int32'))
            self.c, self.h = self.encoder(w, self.c, self.h)

    def test_decode(self,predict_id):
        word = Variable(self.ARR.array([predict_id],dtype='int32'))
        predict_vec, self.c, self.h = self.decoder(word, self.c, self.h)
        return predict_vec

    def generate(self,input_text,len_limit,vocab,id2word):
        self.reset()
        self.test_encode(input_text)

        response = ''
        word_id = vocab['<start>']
        for _ in range(len_limit):
            predict_vec = self.test_decode(predict_id=word_id)
            wid = self.ARR.argmax(predict_vec.data)
            wid = int(cuda.to_cpu(wid))
            word = id2word[wid]
            word_id = vocab[word]
            if word == '<eos>':
                break
            response = response + word + ' '
        return response

def main():
    if os.path.exists('./data/corpus/dictionary.dict'):
        corpus = ConvCorpus(create_flg=False, batch_size=BATCH_SIZE, size_filter=True)
        corpus.load(load_dir='./data/corpus/')
    else:
        corpus = ConvCorpus(create_flg=True, batch_size=BATCH_SIZE, size_filter=True)
        corpus.save(save_dir='./data/corpus/')

    model = Seq2Seq(vocab_size=len(corpus.dic.token2id),
                    embed_size=EMBED_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    batch_size=BATCH_SIZE,
                    flag_gpu=FLAG_GPU)
    model.reset()

    if FLAG_GPU:
        cuda.get_device(0).use()
        model.to_gpu(0)
        ARR = cuda.cupy
    else:
        ARR = np

    optimizer = optimizers.Adam(alpha=0.001)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.GradientClipping(5))

    input_mat = []
    output_mat = []
    max_input_len = max_output_len = 0

    for input_text, output_text in zip(corpus.rough_posts, corpus.rough_cmnts):
        output_text.append(corpus.dic.token2id['<eos>'])

        max_input_len = max(max_input_len, len(input_text))
        max_output_len = max(max_output_len, len(output_text))

        input_mat.append(input_text)
        output_mat.append(output_text)

    for li in input_mat:
        insert_num = max_input_len - len(li)
        for _ in range(insert_num):
            li.insert(0,corpus.dic.token2id['<pad>'])
    for li in output_mat:
        insert_num = max_output_len - len(li)
        for _ in range(insert_num):
            li.append(corpus.dic.token2id['<pad>'])
    input_mat = np.array(input_mat, dtype=np.int32).T
    output_mat = np.array(output_mat, dtype=np.int32).T

    accum_loss = 0
    for num, epoch in enumerate(range(EPOCH_NUM)):
        total_loss = 0
        batch_num = 0
        perm = np.random.permutation(len(corpus.rough_posts))

        #assert len(corpus.rough_posts)//BATCH_SIZE == 0
        for i in range(0, len(corpus.rough_posts), BATCH_SIZE):
            input_batch = input_mat[:, perm[i:i + BATCH_SIZE]]
            output_batch = output_mat[:, perm[i:i + BATCH_SIZE]]

            model.reset()
            model.encode(input_batch)

            end_batch = ARR.array([corpus.dic.token2id['<start>'] for _ in range(BATCH_SIZE)])
            first_words = output_batch[0]
            loss, predict_mat = model.decode(end_batch, first_words, train=True)
            next_ids = first_words
            accum_loss += loss
            for w_ids in output_batch[1:]:
                loss, predict_mat = model.decode(next_ids, w_ids, train=True)
                next_ids = w_ids
                accum_loss += loss

            model.cleargrads()
            accum_loss.backward()
            optimizer.update()
            total_loss += float(accum_loss.data)
            batch_num += 1
            print('Epoch:',num + 1,'batch:',batch_num,'batch loss:{:.2f}'.format(float(accum_loss.data)))
            accum_loss = 0

        if (epoch + 1) % 2 == 0:
            serializers.save_hdf5('./data/model/{}.model'.format(epoch + 1),model)
            serializers.save_hdf5('./data/model/{}.state'.format(epoch + 1),optimizer)


if __name__ == '__main__':
    main()
