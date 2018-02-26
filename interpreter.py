import numpy as np
import chainer
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import MeCab
import pickle
from utils import *
from seq2seq import *

model_path = './data/model/20.model'

def interpreter():
    corpus = ConvCorpus(create_flg=False)
    corpus.load(load_dir='./data/corpus/')


    model = Seq2Seq(vocab_size=len(corpus.dic.token2id),
                    embed_size=EMBED_SIZE,
                    hidden_size=HIDDEN_SIZE,
                    batch_size=1,
                    flag_gpu=FLAG_GPU)
    serializers.load_hdf5(model_path,model)

    if FLAG_GPU:
        cuda.get_device(0).use()
        model.to_gpu(0)
        ARR = cuda.cupy
    else:
        ARR = np

    print('準備出来ました！どうぞ話しかけてください！')
    print('お話を終わりたいときは"exit"と打ってね')

    while 1:
        print('>> ',end='')
        sentence = input()
        if sentence == 'exit':
            print('ばいばい')
            break

        try:
            input_ids = [corpus.dic.token2id[word] for word in sentence_to_words(sentence)] + [corpus.dic.token2id['<eos>']]
        except:
            print('ごめんね，まだ知らない単語があるんだ．')
            continue

        model.reset()

        response = model.generate(input_ids,
                                  len_limit=len(input_ids) + 30,
                                  vocab=corpus.dic.token2id,
                                  id2word=corpus.dic)

        print('-> ',response)
        print()

if __name__ == '__main__':
    interpreter()

