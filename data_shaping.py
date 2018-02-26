import MeCab

def sentence_to_words(sentence):
    tagger = MeCab.Tagger('-Owakati')
    result = tagger.parse(sentence).split(' ')
    return ' '.join(result[:-1])

if __name__ == '__main__':

    post_lines = open('./data/corpus/post-data.txt','r').read().split('\n')
    cmnt_lines = open('./data/corpus/cmnt-data.txt','r').read().split('\n')
    out_f = open('./data/corpus/post_cmnt_data.txt','w')
    assert len(post_lines) == len(cmnt_lines)
    for line1,line2 in zip(post_lines,cmnt_lines):
        sentence1 = line1.split('\t')
        sentence2 = line2.split('\t')
        try:
            post_sentence = sentence_to_words(sentence1[1])
            cmnt_sentence = sentence_to_words(sentence2[1])
            out_f.write(post_sentence + '\t' + cmnt_sentence + '\n')
        except:
            pass
    out_f.close()