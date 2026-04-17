import cv2
import time
import torch
import tqdm
import psutil
import os
import numpy as np
from torch.utils.data import Dataset
from encoder import pre_encoder1, pre_encoder2, pre_encoder3, img_patch_encoder
from torchvision import transforms as tf
import math
from PIL import Image

def get_area(poly, max_length):
    empty = np.zeros((max_length, max_length), dtype=np.uint8)
    color = 255
    mask = cv2.fillPoly(empty, [poly], (color))
    mask = (mask == color).sum()

    return mask


def get_adjacent(boundary, max_len, k=1):
    """
    The input is a set of contour points, and the function will get an adjacency matrix constructed
    from the set of points.
    :param boundary: type = Ndarray.
    :param sparse: type = bool
    :return: an adjacency matrix.
    """
    n = len(boundary)
    adjacent_matrix = np.eye(n)
    temp = np.eye(n)
    for i in range(k):
        adjacent_matrix += np.roll(temp, i + 1, axis=0)
        adjacent_matrix += np.roll(temp, -i - 1, axis=0)
    temp = np.zeros((max_len, max_len))
    temp[:n, :n] = adjacent_matrix
    return torch.from_numpy(temp).to_sparse()

def get_adjacent2(boundary, max_len, k=1):
    """
    The input is a set of contour points, and the function will get an adjacency matrix constructed
    from the set of points.
    :param boundary: type = Ndarray.
    :param sparse: type = bool
    :return: an adjacency matrix.
    返回非稀疏矩阵
    """
    n = len(boundary)
    adjacent_matrix = np.eye(n)
    temp = np.eye(n)
    for i in range(k):
        adjacent_matrix += np.roll(temp, i + 1, axis=0)
        adjacent_matrix += np.roll(temp, -i - 1, axis=0)
    temp = np.zeros((max_len, max_len))
    temp[:n, :n] = adjacent_matrix

    return torch.from_numpy(temp)

def generate_tensor(n, max_length):

    tensor = torch.zeros((max_length, max_length), dtype=torch.bool)
    tensor[:n, :n] = True

    return tensor 


class MyDataSet(Dataset):
    def __init__(self, GT_config, args):
        super(MyDataSet, self).__init__()
        # ======================
        # raw data (only keep raw)
        # ======================
        self.raw_pcd = GT_config['full_pcd_all']
        self.raw_img = GT_config['img_all']
        self.GT_pairs = np.array(GT_config['GT_pairs'], dtype=np.int32)
        self.long = list(map(len, self.raw_pcd))

        # initial parameters
        self.model = GT_config['model_type']  # train, test, matching
        self.patch_size = GT_config['patch_size']  # 3x3, 7x7, 11x11
        self.c_model = GT_config['c_model']  # l, io, ilo

        # get max point nums

        max_points = args.max_length
        self.max_points = max_points

        self.mask_all = torch.zeros(
            (len(self.GT_pairs), self.max_points, self.max_points),
            dtype=torch.bool
        )

        self.adj_all = [self.get_adj(i) for i in range(len(self.raw_img))]

        for i in range(len(self.GT_pairs)):
            s = GT_config['source_ind'][i]
            t = GT_config['target_ind'][i]
            self.mask_all[i][s, t] = True
        n = len(self.raw_img)
    
        c = GT_config['channel']
        self.trans = tf.Compose([
            tf.ToTensor(),
            tf.Resize((224, 224)),
            tf.CenterCrop((224, 224)),
            tf.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        # get max img shape
        shape_all = np.array(GT_config['shape_all'])
        shape_all = torch.from_numpy(shape_all)[:, :2]
        self.height_max = 1319  # length of max length
        self.width_max = self.height_max
        self.mid_area = self.height_max ** 2

        self.att_mask_s = torch.zeros(len(self.GT_pairs), 1, 1)
        self.att_mask_t = torch.zeros(len(self.GT_pairs), 1, 1)

        if self.model != 'searching':
            pair_n = len(self.GT_pairs)

            self.mask_all = torch.zeros((pair_n, self.max_points, self.max_points), dtype=torch.bool)

            for i in range(pair_n):
                s, t = GT_config['source_ind'][i], GT_config['target_ind'][i]
                self.mask_all[i][s, t] = True

        self.full_s_buf = np.zeros((self.max_points, 2), dtype=np.float32)
        self.full_t_buf = np.zeros((self.max_points, 2), dtype=np.float32)

    def get_adj(self, i):
        n = self.long[i]
        adj = np.eye(n, dtype=np.float32)

        k = 8
        for j in range(k):
            adj += np.roll(np.eye(n), j + 1, axis=0)
            adj += np.roll(np.eye(n), -j - 1, axis=0)

        full_adj = np.zeros((self.max_points, self.max_points), dtype=np.float32)
        full_adj[:n, :n] = adj

        return torch.from_numpy(full_adj)

    def __len__(self):
        if self.model in ["matching_train", "matching_test"]:
            return len(self.GT_pairs)
        else:
            return len(self.raw_img)

    def __getitem__(self, idx):
        # print("RAM:", psutil.Process(os.getpid()).memory_info().rss / 1024**3, "GB")
        if self.model in ['matching_train', 'matching_test']:

            idx_s, idx_t = self.GT_pairs[idx]
            idx_s, idx_t = int(idx_s), int(idx_t)

            # ---- point cloud ----

            n_s = int(self.long[idx_s])
            n_t = int(self.long[idx_t])

            full_s = self.full_s_buf.copy()
            full_t = self.full_t_buf.copy()
            full_s[:n_s] = self.raw_pcd[idx_s][:n_s]
            full_t[:n_t] = self.raw_pcd[idx_t][:n_t]

            full_s = torch.from_numpy(full_s)
            full_t = torch.from_numpy(full_t)

            full_s = full_s / (self.height_max / 2.) - 1
            full_t = full_t / (self.height_max / 2.) - 1

            # ---- image (on demand) ----

            img_s = self.trans(self.raw_img[idx_s])
            # print("img_s tensor:", img_s.shape, img_s.element_size() * img_s.nelement() / 1024**2, "MB")
            img_t = self.trans(self.raw_img[idx_t])
            # print("img_t tensor:", img_t.shape, img_t.element_size() * img_t.nelement() / 1024**2, "MB")

            # ---- adjacency (lazy) ----
            adj_s = self.adj_all[idx_s]
            adj_t = self.adj_all[idx_t]

            c_s = torch.zeros((self.max_points, self.patch_size, self.patch_size))
            c_t = torch.zeros((self.max_points, self.patch_size, self.patch_size))

            t_s = torch.zeros((self.max_points, 3, self.patch_size, self.patch_size))
            t_t = torch.zeros((self.max_points, 3, self.patch_size, self.patch_size))

            factors = (1.0, 1.0)

            mask = self.mask_all[idx]

            att_mask = (self.att_mask_s[idx], self.att_mask_t[idx])

            return (
            (mask, n_s, n_t, idx_s, idx_t),        
            (img_s, img_t),   
            (full_s, full_t), 
            (c_s, c_t),       
            (t_s, t_t),       
            (adj_s, adj_t),
            factors,
            att_mask        
            )
        elif self.model == 'save_stage1_feature':
            n = self.long[idx]

            # ===== 点云（raw + normalized）=====
            full_raw_np = np.zeros((self.max_points, 2), dtype=np.float32)
            full_raw_np[:n] = self.raw_pcd[idx]

            full = torch.from_numpy(full_raw_np.copy())
            full = full / (self.height_max / 2.) - 1

            full_raw = torch.from_numpy(full_raw_np)

            # ===== 图像 =====
            img_np = self.raw_img[idx]
            img = self.trans(img_np)

            # ===== 邻接矩阵 =====
            adj = self.get_adj(idx)

            # ===== t_input =====
            img_pad = np.zeros((self.height_max, self.width_max, 3), dtype=np.uint8)
            h, w = img_np.shape[:2]
            img_pad[:h, :w] = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            img_pad = torch.from_numpy(img_pad).permute(2, 0, 1).unsqueeze(0) / 255.0

            t_input = img_patch_encoder(
                img_pad,
                full_raw.unsqueeze(0),
                self.patch_size
            )[0]

            # ===== c_input =====
            mask_np = np.zeros((self.height_max, self.width_max), dtype=np.float32)
            mask_np[:h, :w] = (img_np.sum(-1) > 0)
            mask = torch.from_numpy(mask_np).unsqueeze(0)

            if self.c_model == 'l':
                c_input = pre_encoder1(mask, full_raw.unsqueeze(0), self.patch_size)[0]
            elif self.c_model == 'io':
                c_input = pre_encoder2(mask, full_raw.unsqueeze(0), self.patch_size)[0]
            else:
                c_input = pre_encoder3(mask, full_raw.unsqueeze(0), self.patch_size)[0]

            factor = 1.0

            return full, img, t_input, adj, factor, c_input


class MyDataSet_searching(Dataset):
    def __init__(self, stage1_feature, args):
        
        raw_feat = stage1_feature["saved_feature"]
        raw_pcd = stage1_feature["full_pcd"]
        raw_pairs = stage1_feature["GT_pairs"]

        self.model = args.model_type
        self.max_points = args.max_length
        # =========================
        # 1. 过滤 None 样本
        # =========================
        self.stage1_feature = []
        self.full_pcd = []
        for i in range(len(raw_feat)):
            if raw_feat[i] is None or raw_pcd[i] is None:
                continue
            self.stage1_feature.append(raw_feat[i])
            self.full_pcd.append(raw_pcd[i])

        print(f"✅ valid samples: {len(self.stage1_feature)}")
        n = len(self.full_pcd)
        # =========================
        # 2. padding point cloud
        # =========================
        self.full_pcd_all = torch.zeros((n, self.max_points, 2))

        for i in range(n):
            num = min(len(self.full_pcd[i]), self.max_points)
            self.full_pcd_all[i, :num] = torch.tensor(self.full_pcd[i][:num])

        # =========================
        # 3. 重新构建 GT_pairs（防越界）
        # =========================
        self.GT_pairs = []

        for s, t in raw_pairs:
            if s < n and t < n:
                self.GT_pairs.append((s, t))

        print(f"✅ valid pairs: {len(self.GT_pairs)}")
    
    def __len__(self):
        return len(self.GT_pairs)


    def __getitem__(self, idx):

        s, t = self.GT_pairs[idx]

        feat_s = self.stage1_feature[s]
        feat_t = self.stage1_feature[t]

        pcd_s = self.full_pcd_all[s]
        pcd_t = self.full_pcd_all[t]

        if feat_s is None or feat_t is None:
            raise ValueError(f"None feature at pair {idx}: {s},{t}")

        # =========================
        # ① all_data（6字段）
        # =========================
        all_data = (
            feat_s,
            feat_t,
            s,
            t,
            pcd_s,
            pcd_t
        )

        # =========================
        # ② mask_para（训练用辅助信息）
        # =========================
        mask_para = (self.full_pcd[s].shape[0], self.full_pcd[t].shape[0])

        return all_data, mask_para


class MyRealDataSet(Dataset):
    def __init__(self, GT_config, args):
        super(MyRealDataSet, self).__init__()
        self.inputs = {
            'full_pcd_all': [],
            'img_all': [],
            'c_input': [],
            't_input': [],
            'GT_pairs': GT_config['GT_pairs'],
        }
        # initial parameters
        self.model = GT_config['model_type']  # train, test, matching
        patch_size = GT_config['patch_size']  # 3x3, 7x7, 11x11
        c_model = GT_config['c_model']  # l, io, ilo
        n = len(GT_config['img_all'])  # nums of fragments
        pair_n = len(GT_config['GT_pairs'])  # nums of gt pairs
        # for i in tqdm.trange(n):
        #     '''0~360随机旋转'''
        #     new_pcd,new_extra_img,new_ori_img = self.rotate_func(GT_config['full_pcd_all'][i], GT_config['extra_img'][i], GT_config['img_all'][i])
        #     GT_config['full_pcd_all'][i], GT_config['extra_img'][i], GT_config['img_all'][i] = new_pcd,new_extra_img,new_ori_img
        #     GT_config['shape_all'][i] = np.array((new_ori_img.shape[1],new_ori_img.shape[0],new_ori_img.shape[2]), dtype=np.int64)

        c = GT_config['channel']
        trans = tf.Compose([
            tf.ToTensor(),
            tf.Resize(224),
            tf.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
        ])
        # get max point nums
        self.long = list(map(lambda x: len(x), GT_config['full_pcd_all'])) #每个轮廓的长度
        # self.long = list(map(lambda x: len(x), GT_config['down_sample_pcd'])) #每个轮廓的长度
        # max_points = max(self.long)
        max_points = args.max_length
        self.max_points = max_points

        # get max img shape
        shape_all = np.array(GT_config['shape_all'])
        shape_all = torch.from_numpy(shape_all)[:, :2]
        height_max = 1319  # length of max length
        # height_max = 1119  # length of max length
        # height_max = 296  # length of mid area
        # height_max = 305  # length of mid length
        width_max = height_max
        mid_area = height_max ** 2

        print("更新邻接矩阵")
        for i in tqdm.trange(n):
            # GT_config['adj_all'][i] = get_adjacent(GT_config['down_sample_pcd'][i], max_points, k = 8)
            # GT_config['adj_all'][i] = get_adjacent2(GT_config['down_sample_pcd'][i], max_points, k = 8)
            GT_config['adj_all'].append(get_adjacent2(GT_config['full_pcd_all'][i], max_points, k = 8))

        # initial inputs
        self.inputs['full_pcd_all'] = torch.zeros((n, max_points, 2))  # input contours
        self.inputs['c_input'] = torch.zeros((n, max_points, patch_size, patch_size))  # input patches
        self.inputs['t_input'] = torch.zeros((n, max_points, 3, patch_size, patch_size))  # input patches
        self.inputs['img_all'] = torch.zeros((n, c, 224, 224))  # input image
        self.inputs['adj_all'] = GT_config['adj_all']
        # self.inputs['adj_all'] = []
        self.inputs['factor'] = torch.zeros(n)  # resize factor

        #  deal with each inputs of fragments
        print('dealing with fragments')
        for i in tqdm.trange(n):
            '''points'''
            full_pcd = GT_config['full_pcd_all'][i]
            self.inputs['full_pcd_all'][i][0:self.long[i]] = torch.from_numpy(full_pcd)


            '''img_ori'''
            temp_empty_img = np.zeros((height_max, width_max, c), dtype=np.uint8)  # initial
            # normalization，把原来碎片放置到一个统一的最大画幅上，然后进行缩放可以保持不同碎片之间的相对尺度一致（这里用的是外推的碎片）
            temp_empty_img[:shape_all[i][1], :shape_all[i][0]] = cv2.cvtColor(GT_config['img_all'][i],
                                                                                          cv2.COLOR_BGR2RGB)                                                                            
            # resize 2 224 x 224
            new_img = trans(temp_empty_img)
            self.inputs['img_all'][i] = new_img
            self.inputs['factor'][i] = 1

            # input patches
            img = cv2.cvtColor(GT_config['img_all'][i], cv2.COLOR_BGR2RGB) # 311，298，3
            img = np.pad(img, ((0, 20), (0, 20), (0, 0)), 'constant', constant_values=(0, 0)) # 331，318，3
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0) / 255 # 1,3,331,318
            t_input = img_patch_encoder(img, self.inputs['full_pcd_all'][i].unsqueeze(0), patch_size)
            self.inputs['t_input'][i] = t_input[0]
            img = (GT_config['img_all'][i] != 0).all(-1)  # get extracted template
            img = np.pad(img, ((0, 20), (0, 20)), 'constant', constant_values=(0, 0))
            img = torch.from_numpy(img).float().unsqueeze(0) #这一步之后，有像素的地方都是1，无像素的地方都是0

            if c_model == 'l':  # only contour line
                c_input = pre_encoder1(img, self.inputs['full_pcd_all'][i].unsqueeze(0), patch_size) # 1,2778,7,7
            elif c_model == 'io':  # Interior and exterior of contour
                c_input = pre_encoder2(img, self.inputs['full_pcd_all'][i].unsqueeze(0), patch_size)
            else:  # Interior, exterior and contour
                c_input = pre_encoder3(img, self.inputs['full_pcd_all'][i].unsqueeze(0), patch_size)
            self.inputs['c_input'][i] = c_input[0]
        self.inputs['shape'] = [height_max, height_max]

        # points normalization
        self.inputs['full_pcd_all'] = self.inputs['full_pcd_all'] / (height_max / 2.) - 1

        #  deal with correctly matched pairs into matrix
        if GT_config['model_type'] != 'searching':
            print("dealing with gt pairs into matrix")
            self.inputs['mask_all'] = []
            # self.inputs['att_mask_s'] = []
            # self.inputs['att_mask_t'] = []
            for i in tqdm.trange(pair_n):
                mask = torch.zeros((max_points, max_points), dtype=torch.bool)
                self.inputs['mask_all'].append(mask)

            self.inputs['mask_all'] = torch.stack(self.inputs['mask_all'], 0)

    def __len__(self):
        # if self.model == 'searching' or self.model == 'stage1':
        #     return len(self.inputs['img_all'])
        # elif  self.model == 'train' or self.model == 'test':
        #     return len(self.inputs['GT_pairs'])
        if  self.model == 'real_dataset_test':
            return len(self.inputs['GT_pairs'])
        elif self.model == 'stage1_real':
            return len(self.inputs['img_all'])

    def __getitem__(self, idx):
        self.inputs['GT_pairs'] = np.array(self.inputs['GT_pairs'])
        if  self.model == 'real_dataset_test':
            idx_s, idx_t = self.inputs['GT_pairs'][idx]
            full_s, full_t = self.inputs['full_pcd_all'][idx_s], self.inputs['full_pcd_all'][idx_t]

            return \
                (self.inputs['mask_all'][idx], self.long[idx_s], self.long[idx_t],idx_s, idx_t), \
                (self.inputs['img_all'][idx_s], self.inputs['img_all'][idx_t]), \
                (full_s, full_t), \
                (self.inputs['c_input'][idx_s], self.inputs['c_input'][idx_t]), \
                (self.inputs['t_input'][idx_s], self.inputs['t_input'][idx_t]), \
                (self.inputs['adj_all'][idx_s], self.inputs['adj_all'][idx_t]), \
                (self.inputs['factor'][idx_s], self.inputs['factor'][idx_t])
        
        elif self.model == 'stage1_real' :
            return self.inputs['full_pcd_all'][idx], self.inputs['img_all'][idx], self.inputs['t_input'][idx], \
                   self.inputs['adj_all'][idx], self.inputs['factor'][idx], self.inputs['c_input'][idx]


        if self.model == 'train' or self.model == 'test':
            idx_s, idx_t = self.inputs['GT_pairs'][idx]
            full_s, full_t = self.inputs['full_pcd_all'][idx_s], self.inputs['full_pcd_all'][idx_t]
            # a = self.inputs['adj_all'][idx_s].to_dense()
            return \
                (self.inputs['mask_all'][idx], self.long[idx_s], self.long[idx_t],idx_s, idx_t), \
                (self.inputs['img_all'][idx_s], self.inputs['img_all'][idx_t]), \
                (full_s, full_t), \
                (self.inputs['c_input'][idx_s], self.inputs['c_input'][idx_t]), \
                (self.inputs['t_input'][idx_s], self.inputs['t_input'][idx_t]), \
                (self.inputs['adj_all'][idx_s], self.inputs['adj_all'][idx_t]), \
                (self.inputs['factor'][idx_s], self.inputs['factor'][idx_t]), \
                (self.inputs['att_mask_s'][idx], self.inputs['att_mask_t'][idx])

        elif self.model == 'searching_refine':
            # idx_s, idx_t = self.inputs['GT_pairs'][idx]
            full_pcd = self.inputs['full_pcd_all'][idx]
            # a = self.inputs['adj_all'][idx_s].to_dense()
            return \
                (self.inputs['img_all'][idx]), \
                (full_pcd), \
                (self.inputs['c_input'][idx]), \
                (self.inputs['t_input'][idx]), \
                (self.inputs['adj_all'][idx]), \
                (self.inputs['factor'][idx])

        elif self.model == 'searching' :
            # return self.inputs['full_pcd_all'][idx], self.inputs['img_all'][idx], self.inputs['c_input'][idx], \
            #        self.inputs['t_input'][idx], self.inputs['adj_all'][idx], self.inputs['factor'][idx]
            return self.inputs['full_pcd_all'][idx], self.inputs['img_all'][idx], self.inputs['t_input'][idx], \
                   self.inputs['adj_all'][idx], self.inputs['factor'][idx], self.inputs['c_input'][idx]
            # \
            #     (self.inputs['img_all'][idx_s], self.inputs['img_all'][idx_t]), \
            #     (full_s, full_t), \
            #     (self.inputs['c_input'][idx_s], self.inputs['c_input'][idx_t]), \
            #     (self.inputs['t_input'][idx_s], self.inputs['t_input'][idx_t]), \
            #     (self.inputs['adj_all'][idx_s], self.inputs['adj_all'][idx_t]), \
            #     (self.inputs['factor'][idx_s], self.inputs['factor'][idx_t])
        elif self.model == 'stage1' :
            return self.inputs['full_pcd_all'][idx], self.inputs['img_all'][idx], self.inputs['t_input'][idx], \
                   self.inputs['adj_all'][idx], self.inputs['factor'][idx], self.inputs['c_input'][idx]


    def rotate_func(self, pcd, new_extra, new, pad_=10):
        # angle = np.random.uniform(start, end)
        # angle = 0
        new_extra = new_extra.transpose(1, 0, 2)
        new = new.transpose(1, 0, 2)
        angle = np.random.uniform(0, 2*np.pi)
        cos_, sin_ = np.cos(angle), np.sin(angle)
        # x, y = (pcd[:, 0].max() + pcd[:, 0].min()) * 0.5, (pcd[:, 1].max() + pcd[:, 1].min()) * 0.5
        x, y = 0, 0
        # temp_matrix = np.array([[cos_, -sin_, -x * cos_ + y * sin_],
        #                         [sin_, cos_, -x * sin_ - y * cos_]])
        temp_matrix = np.array([[cos_, -sin_, -x * cos_ + y * sin_],
                                [sin_, cos_, -x * sin_ - y * cos_]])
        temp_pcd = np.matmul(np.hstack((pcd, np.ones((len(pcd), 1)))), temp_matrix.T)
        shift_x = (0 - temp_pcd[:, 0].min())
        shift_y = (0 - temp_pcd[:, 1].min())
        # pcd = np.hstack((pcd[:, 1].reshape(-1, 1), pcd[:, 0].reshape(-1, 1)))
        rotate_matrix = np.array([[cos_, -sin_, -x * cos_ + y * sin_ + shift_x + pad_],
                                [sin_, cos_, -x * sin_ - y * cos_ + shift_y + pad_]])

        pcd = np.matmul(np.hstack((pcd, np.ones((len(pcd), 1)))), rotate_matrix.T)
        # pcd = np.hstack((pcd[:, 1].reshape(-1, 1), pcd[:, 0].reshape(-1, 1)))
        width_max, height_max = pcd[:, 0].max(), pcd[:, 1].max()
        
        
        # cv2.imwrite(os.path.join(test_save_path, 'fragment {}.png'.format(str(3).zfill(4))), new[:, :, :3])
        
        new_extra = \
            cv2.warpAffine(new_extra, rotate_matrix, (int(width_max) + pad_, int(height_max) + pad_), flags=cv2.INTER_NEAREST,
                        borderValue=0)
        new = \
            cv2.warpAffine(new, rotate_matrix, (int(width_max) + pad_, int(height_max) + pad_), flags=cv2.INTER_NEAREST,
                        borderValue=0)

        # cv2.imwrite(os.path.join(test_save_path, 'fragment {}.png'.format(str(4).zfill(4))), new[:, :, :3])

        new_extra = new_extra.transpose(1, 0, 2)
        new = new.transpose(1, 0, 2)
        
        return pcd.astype(int), new_extra, new



class MyRealDataSet_searching(Dataset):
    def __init__(self, stage1_feature, args):
        self.stage1_feature = stage1_feature["saved_feature"]
        self.GT_pairs = stage1_feature["GT_pairs"]
        self.full_pcd = stage1_feature["full_pcd"]
        self.model = args.model_type
        self.adj = []
        self.inputs = {
            'full_pcd_all': [],
        }

        n = len(self.full_pcd)
        max_points = args.max_length
        self.inputs['full_pcd_all'] = torch.zeros((n, max_points, 2))
        self.long = list(map(lambda x: len(x), self.full_pcd))
        height_max = 1319

        self.inputs['full_pcd_all'] = self.inputs['full_pcd_all'] / (height_max / 2.) - 1
        print("更新邻接矩阵 Over")

    
    def __len__(self):
        if self.model == 'stage2':
            return len(self.GT_pairs)
        elif self.model == 'stage2_real_searching':
            return len(self.stage1_feature)

    def __getitem__(self, idx):
        if self.model == 'stage2':
            idx_s, idx_t = self.GT_pairs[idx]

            return (self.stage1_feature[idx_s], self.stage1_feature[idx_t], idx_s, idx_t, self.inputs['full_pcd_all'][idx_s], self.inputs['full_pcd_all'][idx_t]), \
                  (self.long[idx_s], self.long[idx_t])

        elif self.model == 'stage2_real_searching':
            return self.stage1_feature[idx], self.inputs['full_pcd_all'][idx]
        


