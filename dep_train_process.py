import gensim.downloader as api
import numpy as np
from deprel_tagging import onehot_deprel

class WVC(object):  # word2vec converter
    def __init__(self, wv_dict=None):
        if wv_dict == None:   self.wv = api.load('word2vec-google-news-300')    # online huge dictionary
        else:                 self.wv = wv_dict                                 # custom dictionary
        
        self.size = len(self.wv)

        
    def dep_process(self, review):
        review_vect = []
        for word,gov_text,dep_rel in review:
            if dep_rel == 'punct':
                continue
            else :
                vec = self.wv[word] if word in self.wv else self.wv["UNK"]
                gov = self.wv[gov_text] if gov_text in self.wv else self.wv["UNK"]
                word_feat = np.concatenate([vec,gov,onehot_deprel(dep_rel)],axis=0)
                review_vect.append(word_feat)
        return review_vect
