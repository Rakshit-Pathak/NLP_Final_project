import stanfordnlp
import gensim.downloader as api
from pos_tagging import onehot_pos
import numpy as np

class WVC(object):  # word2vec converter
    def __init__(self, wv_dict=None):
        if wv_dict == None:   self.wv = api.load('word2vec-google-news-300')    # online huge dictionary
        else:                 self.wv = wv_dict                                 # custom dictionary
        
        self.size = len(self.wv)
        self.nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos')

#    def const_preprocess(self, review):
#        doc = self.nlp(review)
#        review_vect = []
#        for sent in doc.sentences:
#            for word in sent.words :
#                if word.upos == 'PUNCT':
#                    continue
#                else :
#                    word_feat = [word.text,word.upos]
#                    review_vect.append(word_feat)
#        return review_vect
       
    def const_preprocess(self, review):
        review_vect = []
        for word,pos in review :
            if pos == 'PUNCT':
                continue
            else :
                word_vec = self.wv[word] if word in self.wv else self.wv["UNK"]
                pos_vec = onehot_pos(pos)
                feature_vec = np.concatenate([word_vec,pos_vec],0)
                review_vect.append(feature_vec)
        return review_vect