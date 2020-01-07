import stanfordnlp
import gensim.downloader as api
import numpy as np
from deprel_tagging import onehot_deprel

class WVC(object):  # word2vec converter
    def __init__(self, wv_dict=None):
        if wv_dict == None:   self.wv = api.load('word2vec-google-news-300')    # online huge dictionary
        else:                 self.wv = wv_dict                                 # custom dictionary
        
        self.size = len(self.wv)
        self.nlp = stanfordnlp.Pipeline(processors='tokenize,mwt,pos,lemma,depparse')
        
    def dep_preprocess(self, review):
        doc = self.nlp(review)
        review_vect = np.zeros((647,1))
        for sent in doc.sentences:
            for word in sent.words :
                if word.dependency_relation == 'punct':
                    continue
                else :
                    vec = self.wv[word] if word in self.wv else self.wv["UNK"]
                    gov_text = sent.words[word.governor-1].text
                    gov = self.wv[gov_text] if gov_text in self.wv else self.wv["UNK"]
                    word_feat = np.concatenate([vec,gov,onehot_deprel(word.dependency_relation)])
                    word_feat = np.reshape(word_feat,(647,1))
                    review_vect = np.concatenate([review_vect,word_feat],axis=1)
        return review_vect[:,1:]