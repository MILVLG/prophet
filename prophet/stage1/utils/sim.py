import json,random
import numpy as np
import multiprocessing as mp
import pickle
EXAMPLE_NUM=100
def calculate_similarity(val_item):
    #val_qid = val_item["question_id"]
    val_feat = np.load(val_item["path"])
    val_feat = val_feat / np.linalg.norm(val_feat, axis=-1, keepdims=True)
    val_temp_dict = {}

    for train_qid, train_feat in train_feats.items():
        sim = np.matmul(train_feat, val_feat.T)

        #sim_temp = (np.sum(np.max(sim, axis=0)) + np.sum(np.max(sim, axis=1))) / (sim.shape[0] + sim.shape[1])
        sim_temp = (np.sum(np.mean(sim, axis=0)) + np.sum(np.mean(sim, axis=1))) / (sim.shape[0] + sim.shape[1])
        val_temp_dict[train_qid] = sim_temp

    sorted_items = dict(sorted(val_temp_dict.items(), key=lambda x: x[1], reverse=True))
    temp_dict = {"question_id": val_item["question_id"], "similar": list(sorted_items.keys())[:EXAMPLE_NUM]}
    return temp_dict

if __name__ == '__main__':
    train_file = json.load(open('/data2/ouyangxc/NEW_P/out/heuristics_okvqa/train_laten_dict.json', 'r'))
    val_file = json.load(open('/data2/ouyangxc/NEW_P/out/heuristics_okvqa/test_laten_dict.json', 'r'))
    print(len(train_file))
    print(len(val_file))


    train_feats = {}
    for item in train_file:
        train_feat = np.load(item['path'])
        train_feat = train_feat / np.linalg.norm(train_feat, axis=-1, keepdims=True)
        train_feats[item['question_id']] = train_feat

        
    train_sim_list = []
    num = 0
    print('test')

    with mp.Pool() as pool:
        results = pool.map(calculate_similarity, val_file)

    train_sim_list.extend(results)
    
    #print(train_sim_list)

    print(len(train_sim_list))
    sim_dict={x["question_id"]: x["similar"] for x in train_sim_list}
        

    pickle.dump(sim_dict, open('/data2/ouyangxc/NEW_P/out/heuristics_okvqa/sim_test.pkl', 'wb'))
    json.dump(sim_dict, open('/data2/ouyangxc/NEW_P/out/heuristics_okvqa/sim_test.json', 'w'),indent=4)