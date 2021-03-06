from constituency_preprocess import WVC
import json
import pickle
from tqdm import tqdm


with open('./testlist.txt', 'r') as f:
    data_train = json.loads(f.read())
    
with open('./models_in_progress/dict_compressed.pickle', 'rb') as f:
    wv_dict = pickle.load(f)
    f.close()
    
train_text,train_labels = zip(*data_train)
wvc = WVC(wv_dict)


const_preprocess_review = []
for reviews,score in tqdm(zip(train_text,train_labels)):
    review = wvc.const_preprocess(reviews)
    const_preprocess_review.append([review,score])
    
with open('const_preprocess_test_review.pkl','wb') as f:
    pickle.dump(const_preprocess_review,f)