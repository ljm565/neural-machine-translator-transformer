from utils_func import preprocessing
from tqdm import tqdm
import random


    
def preprocessing_data(r_path):
    with open(r_path, 'r') as f:
        data = f.readlines()
  
    data = [d.split('\t')[:2] for d in data]
    data = [[preprocessing(d[0]), preprocessing(d[1])] for d in tqdm(data)]

    random.seed(999)
    all_id = list(range(len(data)))
    testset_id = random.sample(all_id, 1000)
    trainset_id = list(set(all_id) - set(testset_id))
    trainset = [data[i] for i in trainset_id]
    testset = [data[i] for i in testset_id]

    return trainset, testset