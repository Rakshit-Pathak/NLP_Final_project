from dependency_preprocess import WVC
import json
import pickle
from tqdm import tqdm


with open('./trainlist.txt', 'r') as f:
    data_train = json.loads(f.read())
    
with open('./dict_compressed.pickle', 'rb') as f:
    wv_dict = pickle.load(f)
    f.close()
    
train_text,train_labels = zip(*data_train)
wvc = WVC(wv_dict)


dep_preprocess_review = []
for reviews,score in tqdm(zip(train_text,train_labels)):
    review = wvc.dep_preprocess(reviews)
    dep_preprocess_review.append([review,score])
    
with open('dep_preprocess_review.json','w') as f:
    json.dump(dep_preprocess_review,f)