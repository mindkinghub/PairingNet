import os
import pickle
import random
import numpy as np

def judge_where(dot1, dot2, cur):
    if cur < dot1:
        return 1
    elif cur >= dot2:
        return 3
    else:
        return 2

def divide(data_path, R_PATH):
    # 加载原始数据
    with open(data_path, 'rb') as file:
        data = pickle.load(file)

    print("Data field lengths:", {k: len(v) for k, v in data.items()})

    # 准备输出集
    train_set = {k: [] for k in ['full_pcd_all','img_all','extra_img','shape_all','GT_pairs','source_ind','target_ind','intersection_len']}
    valid_set = {k: [] for k in ['full_pcd_all','img_all','extra_img','shape_all','GT_pairs','source_ind','target_ind','intersection_len']}
    test_set = {k: [] for k in ['full_pcd_all','img_all','extra_img','shape_all','GT_pairs','source_ind','target_ind','intersection_len']}

    # ------------------------
    # 1. 按 img_all 分割 train/valid/test
    nums_img = len(data['img_all'])
    shuffle_ind = np.arange(nums_img)
    random.shuffle(shuffle_ind)
    cur_ind = np.arange(nums_img)
    shuffle_cur_dic = {shuffle_ind[i]: cur_ind[i] for i in range(nums_img)}

    dot1, dot2 = (nums_img * 5)//10, (nums_img * 6)//10

    # 仅对 img_all、full_pcd_all、shape_all、extra_img（如果有）分割
    for key in ['img_all','full_pcd_all','shape_all','extra_img']:
        key_len = len(data[key])
        train_set[key] = [data[key][idx] for idx in shuffle_ind[:min(dot1,key_len)]]
        valid_set[key] = [data[key][idx] for idx in shuffle_ind[min(dot1,key_len):min(dot2,key_len)]]
        test_set[key]  = [data[key][idx] for idx in shuffle_ind[min(dot2,key_len):key_len]]

    # ------------------------
    # 2. 分配 GT_pairs
    for i, pair in enumerate(data['GT_pairs']):
        cur1 = shuffle_cur_dic.get(pair[0], -1)
        cur2 = shuffle_cur_dic.get(pair[1], -1)

        # 如果索引不在 img_all 范围内，跳过
        if cur1 == -1 or cur2 == -1:
            continue

        where1 = judge_where(dot1, dot2, cur1)
        where2 = judge_where(dot1, dot2, cur2)

        if where1 != where2:
            continue

        if where1 == 1:
            train_set['GT_pairs'].append([cur1, cur2])
            train_set['source_ind'].append(data['source_ind'][i])
            train_set['target_ind'].append(data['target_ind'][i])
        elif where1 == 2:
            valid_set['GT_pairs'].append([cur1-dot1, cur2-dot1])
            valid_set['source_ind'].append(data['source_ind'][i])
            valid_set['target_ind'].append(data['target_ind'][i])
        else:
            test_set['GT_pairs'].append([cur1-dot2, cur2-dot2])
            test_set['source_ind'].append(data['source_ind'][i])
            test_set['target_ind'].append(data['target_ind'][i])

    # ------------------------
    # 保存
    if not os.path.exists(R_PATH):
        os.makedirs(R_PATH)

    with open(os.path.join(R_PATH,'ori_train_set.pkl'),'wb') as f:
        pickle.dump(train_set, f)
    with open(os.path.join(R_PATH,'ori_valid_set.pkl'),'wb') as f:
        pickle.dump(valid_set, f)
    with open(os.path.join(R_PATH,'ori_test_set.pkl'),'wb') as f:
        pickle.dump(test_set, f)

    print("Divide finished!")
    print("Train GT pairs:", len(train_set['GT_pairs']))
    print("Valid GT pairs:", len(valid_set['GT_pairs']))
    print("Test GT pairs:", len(test_set['GT_pairs']))

if __name__ == '__main__':
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    R_PATH = os.path.join(ROOT_DIR, 'data', 'pkl')
    root = os.path.join(ROOT_DIR, 'data', 'pkl', 'matching_set.pkl')
    divide(root, R_PATH)