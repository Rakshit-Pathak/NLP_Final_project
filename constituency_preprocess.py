import stanfordnlp
import gensim.downloader as api
import numpy as np
from pos_tagging import onehot_pos

class WVC(object):  # word2vec converter
    def __init__(self, wv_dict=None):
        if wv_dict == None:   self.wv = api.load('word2vec-google-news-300')    # online huge dictionary
        else:                 self.wv = wv_dict                                 # custom dictionary
        
        self.size = len(self.wv)
        self.nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos')
        
    def const_preprocess(self, review):
        doc = self.nlp(review)
        review_vect = np.zeros((316,1))
        for sent in doc.sentences:
            for word in sent.words :
                if word.upos == 'PUNCT':
                    continue
                else :
                    vec = self.wv[word] if word in self.wv else self.wv["UNK"]
                    word_feat = np.concatenate([vec,onehot_pos(word.upos)])
                    word_feat = np.reshape(word_feat,(316,1))
                    #print(word_feat.size)
                    review_vect = np.concatenate([review_vect,word_feat],axis=1)
        return review_vect[:,1:]