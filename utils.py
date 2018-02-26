import numpy as np
import chainer
from chainer import Chain, Variable, cuda, functions, links, optimizer, optimizers, serializers
import MeCab
import pickle
from gensim import corpora
from gensim.models import word2vec
import re

EMBED_SIZE = 300
HIDDEN_SIZE = 150
BATCH_SIZE = 128
EPOCH_NUM =30
FLAG_GPU = True

def padding(enc_words,dec_words):
    enc_max = np.max([len(row) for row in enc_words])
    enc_words = np.array([[-1]*(enc_max - len(row)) + row for row in enc_words],dtype='int32')
    enc_words = enc_words.T

    dec_max = np.max([len(row) for row in dec_words])
    dec_words = np.array([row + [-1]*(dec_max - len(row)) for row in dec_words], dtype='int32')
    dec_words = dec_words.T
    return enc_words, dec_words

def sentence_to_words(sentence):
    tagger = MeCab.Tagger('-Owakati')
    result = tagger.parse(sentence).split(' ')
    return result[:-1]

class ConvCorpus:
    def __init__(self, create_flg, batch_size=100, size_filter=True):
        self.rough_posts = []
        self.rough_cmnts = []
        self.fine_posts = []
        self.fine_cmnts = []
        self.dic = None

        if create_flg:
            self._construct_dict(batch_size,size_filter)

    def _construct_dict(self,batch_size, size_filter):
        max_len = 30

        rough_posts = []
        rough_cmnts = []
        r = re.compile(r'(.+?)(\t)(.+?)(\n|r\n)')
        for idx, line in enumerate(open('./data/corpus/post_cmnt_data.txt','r')):
            m = r.search(line)
            if m is not None:
                post = [word for word in m.group(1).split(' ')]
                cmnt = [word for word in m.group(3).split(' ')]
                if size_filter:
                    if len(post) <= max_len and len(cmnt) <= max_len:
                        rough_posts.append(post)
                        rough_cmnts.append(cmnt)
                else:
                    rough_posts.append(post)
                    rough_cmnts.append(cmnt)

        remove_num = len(rough_posts) - (int(len(rough_posts) / batch_size) * batch_size)
        del rough_posts[len(rough_posts) - remove_num:]
        del rough_cmnts[len(rough_cmnts) - remove_num:]

        self.dic = corpora.Dictionary(rough_posts + rough_cmnts, prune_at=None)
        self.dic.filter_extremes(no_below=1, no_above=1.0, keep_n=50000)

        # add symbols
        self.dic.token2id['<pad>'] = -1
        self.dic.token2id['<start>'] = len(self.dic.token2id)
        self.dic.token2id['<eos>'] = len(self.dic.token2id)
        self.dic.token2id['<unk>'] = len(self.dic.token2id)

        #self.rough_posts = [[self.dic.token2id.get(word, self.dic.token2id['<unk>']) for word in post] for post in rough_posts]
        #self.rough_cmnts = [[self.dic.token2id.get(word, self.dic.token2id['<unk>']) for word in cmnt] for cmnt in rough_cmnts]

        sim_th = 50
        model = word2vec.Word2Vec.load('./data/model/w2v_jawiki_alltag.model')
        self.rough_posts = self._token_to_id(token_data=rough_posts, model=model, sim_th=sim_th)
        self.rough_cmnts = self._token_to_id(token_data=rough_cmnts, model=model, sim_th=sim_th)

    def _token_to_id(self, token_data, model, sim_th):
        all_word_num = 0
        replace_num = 0
        unk_dic_num = 0
        unk_w2v_num = 0

        corpus_id = []
        for text in token_data:
            text_ids = []
            for word in text:
                all_word_num += 1
                if self.dic.token2id.get(word) is not None:
                    text_ids.append(self.dic.token2id.get(word))
                else:
                    try:
                        sim_words = model.most_similar(positive=[word], topn=sim_th)
                        for idx, candidate_tuple in enumerate(sim_words):
                            if self.dic.token2id.get(candidate_tuple[0]) is not None:
                                replace_num += 1
                                text_ids.append(self.dic.token2id.get(candidate_tuple[0]))
                                break
                            if idx == sim_th - 1:
                                unk_dic_num += 1
                                text_ids.append(self.dic.token2id['<unk>'])
                    except KeyError:
                        unk_w2v_num += 1
                        text_ids.append(self.dic.token2id['<unk>'])
            corpus_id.append(text_ids)

        print('全語彙数:',len(self.dic.token2id))
        print('全単語出現数:',all_word_num)
        print('置き換え成功数:',replace_num)
        print('unk出現数:',unk_dic_num + unk_w2v_num)
        return corpus_id

    def save(self, save_dir):
        self.dic.save(save_dir + 'dictionary.dict')
        with open(save_dir + 'rough_posts.list', 'wb') as f:
            pickle.dump(self.rough_posts, f)
        with open(save_dir + 'rough_cmnts.list', 'wb') as f:
            pickle.dump(self.rough_cmnts, f)

    def load(self,load_dir):
        self.dic = corpora.Dictionary.load(load_dir + 'dictionary.dict')
        with open(load_dir + 'rough_posts.list', 'rb') as f:
            self.rough_posts = pickle.load(f)
        with open(load_dir + 'rough_cmnts.list', 'rb') as f:
            self.rough_cmnts = pickle.load(f)
