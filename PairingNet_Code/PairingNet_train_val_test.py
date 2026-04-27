import __init__
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import cv2
import torch
import math
import random
import time
import pickle
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import numpy as np
from glob import glob
from tqdm import tqdm
from hausdorff import hausdorff_distance
from utils.loss import FocalLoss
from utils.evaluation import e_rmse
from utils.utilz import affine_transform
from utils import pipeline, config, data_preprocess, visualization
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
import re
global c
from torch.utils.data.distributed import DistributedSampler
from utils.infornce_loss import InfoNCE
import torch.nn.functional as F
import numpy as np
import pytorch_warmup as warmup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchsummary import summary
from utils import calute_NDCG 

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu
    torch.cuda.manual_seed_all(seed)  # all gpus

def pad_tensor(t1, t2):
    
    s1 = t1.shape
    s2 = t2.shape
    if s1[-1] >= s2[-1]:
        size_tensor = torch.full((s1[0], 1), s1[-1], dtype=torch.int)
        padded_t1 = torch.cat([t1, size_tensor], dim=-1)
        return padded_t1
    else:
        padding = s2[-1] - s1[-1]
        padded_t1 = torch.cat([t1, torch.zeros(*s1[:-1], padding, dtype=torch.int)], dim=-1)
        size_tensor = torch.full((s1[0], 1), s1[-1], dtype=torch.int)
        padded_t1 = torch.cat([padded_t1, size_tensor], dim=-1)
        return padded_t1

def unpad_tensor(padded_t1):
    original_size = int(padded_t1[0, -1])
    padded_t1 = padded_t1[:, :-1]
    unpadded_t1 = torch.narrow(padded_t1, -1, 0, original_size)
    return unpadded_t1

class Train_model(object):
    def __init__(self, net, args, temperature, case_name):
        """"""
        '''initial tensorboard'''
        self.log_save_path = EXP_path+'/EXP/{}/summary'.format(case_name)
        self.checkpoint_path = EXP_path+'/EXP/{}/checkpoint'.format(case_name)
        self.case_name = case_name
        if not os.path.exists(self.log_save_path):
            os.makedirs(self.log_save_path, exist_ok=True)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path, exist_ok=True)
        self.writer = SummaryWriter(self.log_save_path)

        
        '''set training dataset'''
        print('set training dataset')
        self.train_data, _ = self.set_dataset(args.train_set, args)
        self.train_loader = DataLoader(self.train_data, args.matching_batch_size, num_workers=0,shuffle=True)

        '''set test set in training model'''
        self.val_data, _ = self.set_dataset(args.valid_set, args)
        self.val_loader = DataLoader(self.val_data, 1, num_workers=0, shuffle=False)
        

        '''set training model'''
        print('set training model')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temperature = temperature

        args.flattenNet_config['input_dim'] = args.patch_size ** 2
        self.models = net(args)
        self.models.to(self.device)
        # if torch.cuda.device_count() > 1:
        #     self.models = torch.nn.DataParallel(self.models)

        self.optimizer = torch.optim.Adam(self.models.parameters(),
                                          lr=args.lr,
                                          weight_decay=args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.epoch)
        self.epoch = args.epoch
        self.args = args
        self.best_loss = float('inf')
        self.loss_fn = FocalLoss()

    def save_checkpoint(self, epoch, val_loss=None):
        state = {
            'epoch': epoch,
            'model_state_dict': self.models.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss
        }

        # ===== latest（始终覆盖）=====
        latest_path = os.path.join(self.checkpoint_path, "latest.tar")
        torch.save(state, latest_path)

        # ===== best（更优才保存）=====
        if val_loss is not None and val_loss < self.best_loss:
            self.best_loss = val_loss
            best_path = os.path.join(self.checkpoint_path, "best.tar")
            torch.save(state, best_path)
            print(f"Save BEST model at epoch {epoch}, loss={val_loss:.4f}")

    def load_checkpoint(self):
        latest_path = os.path.join(self.checkpoint_path, "latest.tar")

        if not os.path.exists(latest_path):
            print(f'No checkpoint found at {self.checkpoint_path}')
            return 0

        print(f'Loaded checkpoint from: {latest_path}')
        checkpoint = torch.load(latest_path, map_location=self.device)

        self.models.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        return start_epoch

    @staticmethod
    def get_pad_mask(mask_para):
        """
        input points in each fragments are padded to a fixed nums.
        padded mask denotes the padded part in similarity matrix which
        calculated by source and target point feature.
        """
        bs = mask_para[0].shape[0]
        maxs = mask_para[0].shape[1]
        pad_mask = torch.zeros((bs, maxs, maxs), dtype=torch.bool)
        for m in range(bs):
            a = mask_para[2][m]
            b = mask_para[1][m]
            pad_mask[m][:, mask_para[2][m]:] = True
            pad_mask[m][mask_para[1][m]:, :] = True

        return pad_mask

    @staticmethod
    def get_concat_adj(adj, max_len):
        device = adj.device

        all_edges = []

        for i in range(len(adj)):
            idx = torch.nonzero(adj[i]).t()   # [2, E]

            if idx.numel() == 0:
                continue

            idx = idx + i * max_len
            all_edges.append(idx)

        if len(all_edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long, device=device)

        return torch.cat(all_edges, dim=1)
    

    def get_similarity_matrix(self, feature1, feature2, pad_mask):
        similarity_matrix = torch.bmm(feature1, feature2.permute(0, 2, 1)) / self.temperature
        similarity_matrix[pad_mask] -= 1e9  # give a very small value to the padded part for softmax operation
        s_i = torch.softmax(similarity_matrix, dim=1)  # row softmax
        s_j = torch.softmax(similarity_matrix, dim=-1)  # column softmax
        similarity_matrix = torch.multiply(s_i, s_j)

        return similarity_matrix

    @staticmethod
    def set_dataset(data_path, args):
        with open(data_path, 'rb') as gt_file:
            gt_config = pickle.load(gt_file)

        gt_config['model_type'] = args.model_type
        gt_config['channel'] = args.channel
        gt_config['c_model'] = args.c_model
        gt_config['patch_size'] = args.patch_size
        dataset = data_preprocess.MyDataSet(gt_config, args)
        return dataset, gt_config

    def train_start(self):
        device = self.device
        '''start training'''
        print('start!!!')
        start_epoch = self.load_checkpoint()
        for i in range(start_epoch, self.epoch):
            loss_m_all=[]
            p_all=[]
            v_loss_np_all = []
            v_p_all = []
            self.models.train()
            self.models.requires_grad_(True)

            for _, (mask_para, imgs, pcd, c_input, t_input, adjs, factors, att_mask) in enumerate(tqdm(self.train_loader)):
                max_point_nums = pcd[0].shape[1]
                adj_s = self.get_concat_adj(adjs[0], max_point_nums)
                adj_t = self.get_concat_adj(adjs[1], max_point_nums)
                # adj_s = adj_s.to(device)

                source_input = {
                    'pcd': pcd[0].to(device), 'img': imgs[0].to(device), 'c_input': c_input[0].to(device),
                    'adj': adj_s.to(device), 'factor': factors[0].to(device), 't_input': t_input[0].to(device), "att_mask":att_mask[0].to(device)
                }

                target_input = {
                    'pcd': pcd[1].to(device), 'img': imgs[1].to(device), 'c_input': c_input[1].to(device),
                    'adj': adj_t.to(device), 'factor': factors[1].to(device), 't_input': t_input[1].to(device), "att_mask":att_mask[1].to(device)
                }


                pad_mask = self.get_pad_mask(mask_para).to(device)  # mark the padded part in similarity matrix
                gt_mask = mask_para[0].to(device)  # mark the gt corresponding in similarity matrix
                final_mask = pad_mask | gt_mask

                feature_s, _, w_s = self.models(source_input) # 15,2778,64
                feature_t, _, w_t = self.models(target_input)
                similarity_matrix = self.get_similarity_matrix(feature_s, feature_t, pad_mask) #bs, n, n
                '''matching loss'''
                loss_np, loss_p = self.loss_fn(similarity_matrix, gt_mask, final_mask)


                self.optimizer.zero_grad()
                loss_np.backward()
                self.optimizer.step()
                loss_m_all.append(loss_np.item())
                p_all.append(loss_p.item())
            train_loss=np.mean(loss_m_all)
            self.scheduler.step()
            self.writer.add_scalar('train_loss', train_loss, i)

            if (i + 1) % 2 != 0:
                continue

            '''validation'''
            self.models.eval()
            self.models.requires_grad_(False)
            with torch.no_grad():
                for _, (mask_para, imgs, pcd, c_input, t_input, adjs, factors, att_mask) in enumerate(tqdm(self.val_loader)):
                    max_point_nums = pcd[0].shape[1]
                    adj_s = self.get_concat_adj(adjs[0], max_point_nums)
                    adj_t = self.get_concat_adj(adjs[1], max_point_nums)

                    source_input = {
                        'pcd': pcd[0].to(device), 'img': imgs[0].to(device), 'c_input': c_input[0].to(device),
                        'adj': adj_s.to(device), 'factor': factors[0].to(device), 't_input': t_input[0].to(device), "att_mask":att_mask[0].to(device)
                    }

                    target_input = {
                        'pcd': pcd[1].to(device), 'img': imgs[1].to(device), 'c_input': c_input[1].to(device),
                        'adj': adj_t.to(device), 'factor': factors[1].to(device), 't_input': t_input[1].to(device), "att_mask":att_mask[1].to(device)
                    }

                    pad_mask = self.get_pad_mask(mask_para).to(device)
                    gt_mask = mask_para[0].to(device)
                    final_mask = pad_mask | gt_mask

                    feature_s, _, w_s = self.models(source_input)
                    feature_t, _, w_t = self.models(target_input)
                    similarity_matrix = self.get_similarity_matrix(feature_s, feature_t, pad_mask)
                    '''matching loss'''
                    v_loss_np, v_loss_p = self.loss_fn(similarity_matrix, gt_mask, final_mask)

                    v_loss_np_all.append(v_loss_np.item())
                    v_p_all.append(v_loss_p.item())

            # ===== 记录验证指标 =====
            val_loss = sum(v_loss_np_all) / len(v_loss_np_all)
            val_pos_loss = sum(v_p_all) / len(v_p_all)

            self.writer.add_scalar('valid_loss', val_loss, i)
            self.writer.add_scalar('valid_positive_loss', val_pos_loss, i)
            self.save_checkpoint(i, val_loss)
           
            print('epoch = {}, match_loss = {}, loss_p = {}, v_match_loss = {}, v_loss_p = {}'.format(
                i, np.mean(loss_m_all), np.mean(p_all), val_loss, val_pos_loss
            ))


class TestModel(Train_model):
    def __init__(self, net, args, temperature, case_name, save_img=True, save_corres=False, save_w=True,
                 save_gt=False):
        self.save_img = save_img
        self.save_corres = save_corres
        self.save_w = save_w
        self.save_gt = save_gt
        self.checkpoint_path = EXP_path + '/EXP/{}/checkpoint'.format(case_name)
        self.best_checkpoint = os.path.join(self.checkpoint_path, "best.tar")
        self.latest_checkpoint = os.path.join(self.checkpoint_path, "latest.tar")
        self.case_name = case_name
        '''set testing dataset'''
        print('set testing dataset')
        self.test_data, self.test_set = self.set_dataset(args.test_set, args)
        self.test_loader = DataLoader(self.test_data, 1, shuffle=False)
        self.gt_pairs = self.test_set['GT_pairs']

        '''set testing model'''
        print('set testing model')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temperature = temperature

        args.flattenNet_config['input_dim'] = args.patch_size ** 2
        self.models = net(args)
        self.models.to(self.device)
        self.models.eval()
        self.models.requires_grad_(False)
        self.args = args

    def load_checkpoint_evl(self):
        path = self.best_checkpoint if os.path.exists(self.best_checkpoint) else self.latest_checkpoint
        if not os.path.exists(path):
            raise FileNotFoundError(f"No checkpoint found in {self.checkpoint_path}")

        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)

        self.models.load_state_dict(checkpoint['model_state_dict'])
        return path
    
    def get_max_file_number(self, directory):
        max_number = -1
        max_file = None
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                match = re.search(r'\d+', file)
                if match:
                    number = int(match.group())
                    if number > max_number:
                        max_number = number
                        max_file = file
        return max_file
    
    def calculate_area_opencv(self, points):
        # 将点转换为numpy数组
        contour = np.array(points)
        
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        
        return area
    
    def safe_get_concat_adj(self, adj, max_len):
        if adj.dim()==2 and adj.shape[0]==2:
            return adj  # already global graph

        return self.get_concat_adj(adj, max_len)
    def cosine_similarity(self, vec1, vec2):
        # Compute the dot product of two vectors
        dot_product = np.sum(vec1 * vec2, axis=1)
        # Calculate the L2 norm of each vector (i.e. the length of the vector)
        norm_vec1 = np.linalg.norm(vec1, axis=1)
        norm_vec2 = np.linalg.norm(vec2, axis=1)
        # Calculate cosine similarity
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    
    def safe_get_pad_mask(self, mask_para):
        if isinstance(mask_para, torch.Tensor):
            return mask_para.to(self.device)

        return self.get_pad_mask(mask_para)
    
    def calculate_ratio(self, mixed_feature, vec_c, vec_t):
        # Calculate the sum of two vectors
        vec_contour = self.cosine_similarity(mixed_feature, vec_c)
        vec_texture = self.cosine_similarity(mixed_feature, vec_t)
        sum_vec = vec_contour + vec_texture
        ratio = np.divide(vec_contour, sum_vec, out=np.zeros_like(vec_contour), where=sum_vec!=0)
        return ratio
    

    def test_start(self):
        self.load_checkpoint_evl()
        gt_pairs = self.gt_pairs
        device = self.device
        global c
        valid_nums4 = 0  
        valid_nums2 = 0
        valid_nums6 = 0
        c = 0 
        w_count = 0
        haus_list = [] 
        all_w_s = []
        all_w_t = []

        '''test start'''
        print('test start!')
        saved_test_data = {
            "pred_transformation":[],
            "GT_transformation":[],
        }
        with torch.no_grad():
            for batch, (mask_para, imgs, pcd, c_input, t_input, adjs, factors, att_mask) in enumerate(tqdm(self.test_loader)):
                max_point_nums = pcd[0].shape[1]
                adj_s = self.safe_get_concat_adj(adjs[0], max_point_nums)
                adj_t = self.safe_get_concat_adj(adjs[1], max_point_nums)

                source_input = {
                    'pcd': pcd[0].to(device), 'img': imgs[0].to(device), 'c_input': c_input[0].to(device),
                    'adj': adj_s.to(device), 'factor': factors[0].to(device), 't_input': t_input[0].to(device), "att_mask":att_mask[0].to(device)
                }

                target_input = {
                    'pcd': pcd[1].to(device), 'img': imgs[1].to(device), 'c_input': c_input[1].to(device),
                    'adj': adj_t.to(device), 'factor': factors[1].to(device), 't_input': t_input[1].to(device), "att_mask":att_mask[1].to(device)
                }

                pad_mask = self.safe_get_pad_mask(mask_para).to(device)  # mark the padded part in similarity matrix
                mask = mask_para[0].to(device)
                feature_s, concat_source, w_s = self.models(source_input)
                feature_t, concat_target, w_t = self.models(target_input)
                similarity_matrix = self.get_similarity_matrix(feature_s, feature_t, pad_mask)

                w1 = w_s.detach().cpu().numpy()
                w2 = w_t.detach().cpu().numpy()
                all_w_s.append(w1.reshape(-1))
                all_w_t.append(w2.reshape(-1))

                '''visualization part'''
                if hasattr(mask[0], "to_dense"):
                    gt_matrix = mask[0].to_dense().float().cpu().numpy()
                else:
                    gt_matrix = mask[0].float().cpu().numpy()
                similarity_matrix=similarity_matrix[0].detach().cpu().numpy()
                kernel = np.eye(3, dtype=np.uint8)
                kernel[1, 1] = 0
                kernel = np.rot90(kernel)
                similarity_matrix = cv2.erode(similarity_matrix, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
                kernel[1, 1] = 1
                similarity_matrix = cv2.dilate(similarity_matrix, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

                idx_s, idx_t = gt_pairs[batch]
                s_pcd_origin, t_pcd_origin = self.test_set['full_pcd_all'][idx_s], self.test_set['full_pcd_all'][idx_t]

                s_pcd, t_pcd = self.test_set['full_pcd_all'][idx_s], self.test_set['full_pcd_all'][idx_t]
                ind_s_origin, ind_t_origin = self.test_set['source_ind'][batch], self.test_set['target_ind'][batch]
                source_img, target_img = self.test_set['img_all'][idx_s], self.test_set['img_all'][idx_t]
                result_dir=EXP_path+'/EXP/{}/result'.format(self.case_name)
                os.makedirs(result_dir,exist_ok=True)
                img_save_path = EXP_path+'/EXP/{}/result/img'.format(self.case_name)
                corres_save_path = EXP_path+'/EXP/{}/result/corres'.format(self.case_name)
                os.makedirs(img_save_path, exist_ok=True)
                os.makedirs(corres_save_path, exist_ok=True)
                evl = visualization.Visualization(gt_matrix, similarity_matrix, s_pcd, t_pcd, source_img,
                                                target_img, ind_s_origin, ind_t_origin, s_pcd_origin, t_pcd_origin, conv_threshold=0.006) # 0.006改成0.0006-》RANSAC很慢，改成0.06试试-》效果不好

                transformation, pairs = evl.get_transformation()

                if self.save_w:
                    img_s = source_img.transpose(1, 0, 2)
                    img_s = np.ascontiguousarray(img_s)
                    evl.img_s = evl.weight_visualize(os.path.join(img_save_path, 'w_s{}.png'.format(c)),
                                                    img_s, s_pcd, w1[0])


                    img_t = target_img.transpose(1, 0, 2)
                    img_t = np.ascontiguousarray(img_t)
                    evl.img_t = evl.weight_visualize(os.path.join(img_save_path, 'w_t{}.png'.format(c)),
                                                    img_t, t_pcd, w2[0])

                if self.save_img:
                    evl.get_img(os.path.join(img_save_path, 'pred{}.png'.format(c)), transformation)

                # save ground truth result pairs
                if self.save_gt:
                    evl.get_gt_img(os.path.join(img_save_path, 'gt{}.png'.format(c)))

                # save result pairs with corresponding points connected.
                if self.save_corres:
                    evl.get_corresponding(os.path.join(corres_save_path, 'corres{}.png'.format(c)))

                # evaluation part
                if transformation is None:
                    haus_list.append(0.)
                    transformation = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,0]])


                intersection_s = evl.pcd_s_inter_gt_origin
                intersection_s_trans = affine_transform(intersection_s, np.delete(transformation[:2], 2, axis=-1))
                intersection_t = evl.pcd_t_inter_gt_origin

                haus_dist = hausdorff_distance(intersection_s_trans, intersection_t, distance='euclidean')
                haus_list.append(haus_dist) 

                GT_transformation = evl.GT_transformation

                saved_test_data["pred_transformation"].append(np.delete(transformation[:2], 2, axis=-1))
                saved_test_data["GT_transformation"].append(GT_transformation)

                ermes = e_rmse(intersection_s_trans, intersection_t) 
                if ermes < 2:
                    valid_nums2 += 1
                elif ermes < 4:
                    valid_nums4 += 1
                elif ermes < 6:
                    valid_nums6 += 1
                else:
                    pass
                c += 1

        all_w_s = np.concatenate(all_w_s, axis=0)
        all_w_t = np.concatenate(all_w_t, axis=0)
        print("==== Weight Statistics ====")
        print("[Source Weight]")
        print("mean:", all_w_s.mean())
        print("std :", all_w_s.std())
        print("min :", all_w_s.min())
        print("max :", all_w_s.max())

        print("[Target Weight]")
        print("mean:", all_w_t.mean())
        print("std :", all_w_t.std())
        print("min :", all_w_t.min())
        print("max :", all_w_t.max())
        w_s_sigmoid = 1 / (1 + np.exp(-all_w_s))
        w_t_sigmoid = 1 / (1 + np.exp(-all_w_t))
        print("==== Normalized Analysis ====")

        print("[Sigmoid]")
        print("mean:", w_s_sigmoid.mean(), w_t_sigmoid.mean())


        with open(EXP_path+'/EXP/{}/result/saved_test_exp_data.pkl'.format(self.case_name), 'wb') as file:
            pickle.dump(saved_test_data, file)

        registration_recall = (
        valid_nums2 / len(gt_pairs), valid_nums4 / len(gt_pairs), valid_nums6 / len(gt_pairs))
        with open(EXP_path+'/EXP/{}/result/registration recall.txt'.format(self.case_name), 'w') as f:
            f.write('{}'.format(registration_recall))

        with open(EXP_path+'/EXP/{}/result/haus_list.pkl'.format(self.case_name), 'wb') as f:
            pickle.dump(haus_list, f)


class STAGE_ONE(Train_model):
    def __init__(self, net, args, temperature, case_name, save_img=False, save_corres=False, save_w=False,
                 save_gt=False):
        self.save_img = save_img
        self.save_corres = save_corres
        self.save_w = save_w
        self.save_gt = save_gt
        checkpoint_path = os.path.join(EXP_path, 'EXP', case_name, 'checkpoint')
        self.best_checkpoint = os.path.join(checkpoint_path,"best.tar")
        self.latest_checkpoint=os.path.join(checkpoint_path,"latest.tar")
        self.checkpoint_path_evl = self.best_checkpoint
        self.case_name = case_name
        if os.path.exists(EXP_path+'/EXP2/{}'.format(case_name)) is False:
            os.makedirs(EXP_path+'/EXP2/{}'.format(case_name))
        feature_save_path = args.stage2_feature_path+"/{}".format(case_name)
        self.saved_train_feature_path = feature_save_path+'/train_feature_{}.pkl'.format(args.dataset_select)
        self.saved_val_feature_path = feature_save_path+'/val_feature_{}.pkl'.format(args.dataset_select)
        self.saved_test_feature_path = feature_save_path+'/test_feature_{}.pkl'.format(args.dataset_select)
        if os.path.exists(feature_save_path) is False:
            os.makedirs(feature_save_path)
        
        '''set train dataset'''
        print('set training dataset')
        self.train_data, self.train_GT = self.set_dataset(args.train_set, args)
        self.train_loader = DataLoader(self.train_data, 1, num_workers=0,shuffle=False)
        self.train_gt_pairs = self.train_GT['GT_pairs']
        self.train_pcd = self.train_GT['full_pcd_all']
        self.s_index_train = self.train_GT['source_ind']
        self.t_index_train = self.train_GT['target_ind']

        '''set val dataset'''
        print('set testing dataset')
        self.val_data, self.val_GT = self.set_dataset(args.valid_set, args)
        self.val_loader = DataLoader(self.val_data, 1, num_workers=0, shuffle=False)
        self.val_gt_pairs = self.val_GT['GT_pairs']
        self.test_data, self.test_GT = self.set_dataset(args.test_set, args)
        self.test_loader = DataLoader(self.test_data, 1, num_workers=0, shuffle=False)
        self.test_gt_pairs = self.test_GT['GT_pairs']
        self.test_pcd = self.test_GT['full_pcd_all']
        self.s_index_test = self.test_GT['source_ind']
        self.t_index_test = self.test_GT['target_ind']

        '''set testing model'''
        print('set stage1 model')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temperature = temperature

        args.flattenNet_config['input_dim'] = args.patch_size ** 2
        self.models = net(args).to(self.device)
        self.args = args

        self.val_pcd = self.val_GT['full_pcd_all']
        self.s_index_val = self.val_GT['source_ind']
        self.t_index_val = self.val_GT['target_ind']

    #     self.saved_feature = {
    #     'train_feature': [],
    #     'val_feature': [],
    #     'test_feature': [],
    # }
    #    

    def load_checkpoint_evl(self):

        if os.path.exists(self.best_checkpoint):
            path = self.best_checkpoint
        elif os.path.exists(self.latest_checkpoint):
            path = self.latest_checkpoint
        else:
            raise FileNotFoundError("No checkpoint found")

        checkpoint = torch.load(path, map_location=self.device)
        self.models.load_state_dict(checkpoint['model_state_dict'])
        
    def get_max_file_number(self, directory):
        max_number = -1
        max_file = None
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                match = re.search(r'\d+', file)
                if match:
                    number = int(match.group())
                    if number > max_number:
                        max_number = number
                        max_file = file
        return max_file

    def stage1_start(self):
        self.load_checkpoint_evl()
        self.models.eval()
        self.models.requires_grad_(False)
        device = self.device
        '''save train feature'''
        train_saved_feature = {
            "saved_feature": [],
            "GT_pairs": self.train_gt_pairs,
            "full_pcd": self.train_pcd,
            "source_ind": self.s_index_train,
            "target_ind": self.t_index_train
        }
        with torch.no_grad():
            for batch, (pcd, imgs, t_input, adj, factor, c_input) in enumerate(tqdm(self.train_loader)):
                max_point_nums = pcd.shape[1]
                adj = self.get_concat_adj(adj, max_point_nums)
                inputs = {
                    'pcd': pcd.to(device), 'img': imgs.to(device), 't_input': t_input.to(device),
                    'adj': adj.to(device), 'factor': factor.to(device), 'c_input': c_input.to(device)
                }

                matching_feature, feature, _ = self.models(inputs) # bs,2611,64
                feat = matching_feature if self.args.stage2_data_model == "merged" else feature
                train_saved_feature["saved_feature"].append(feat.squeeze(0).detach().cpu())
                # train_saved_feature["adj"].append(origin_adj[0])
        
        with open(self.saved_train_feature_path, 'wb') as file:
            pickle.dump(train_saved_feature, file)

        '''save val feature'''
        val_saved_feature = {
            "saved_feature": [],
            "GT_pairs": self.val_gt_pairs,
            "full_pcd": self.val_pcd,
            "source_ind": self.s_index_val,
            "target_ind": self.t_index_val
        }
        with torch.no_grad():
            for batch, (pcd, imgs, t_input, adj, factor, c_input) in enumerate(tqdm(self.val_loader)):
                max_point_nums = pcd.shape[1]
                adj = self.get_concat_adj(adj, max_point_nums)
                inputs = {
                    'pcd': pcd.to(device), 'img': imgs.to(device), 't_input': t_input.to(device),
                    'adj': adj.to(device), 'factor': factor.to(device), 'c_input': c_input.to(device)
                }

                matching_feature, feature, _ = self.models(inputs) # bs,2611,64
                feat = matching_feature if self.args.stage2_data_model == "merged" else feature
                val_saved_feature["saved_feature"].append(feat.squeeze(0).detach().cpu())
                # val_saved_feature["adj"].append(origin_adj[0])

        with open(self.saved_val_feature_path, 'wb') as file:
            pickle.dump(val_saved_feature, file)

        '''save test feature'''
        test_saved_feature = {
            "saved_feature": [],
            "GT_pairs": self.test_gt_pairs,
            "full_pcd": self.test_pcd,
            "source_ind": self.s_index_test,
            "target_ind": self.t_index_test
        }
        with torch.no_grad():
            for batch, (pcd, imgs, t_input, adj, factor, c_input) in enumerate(tqdm(self.test_loader)):
                max_point_nums = pcd.shape[1]
                adj = self.get_concat_adj(adj, max_point_nums)
                inputs = {
                    'pcd': pcd.to(device), 'img': imgs.to(device), 't_input': t_input.to(device),
                    'adj': adj.to(device), 'factor': factor.to(device), 'c_input': c_input.to(device)
                }

                matching_feature, feature, _ = self.models(inputs) # bs,2611,64
                feat = matching_feature if self.args.stage2_data_model == "merged" else feature
                test_saved_feature["saved_feature"].append(feat.squeeze(0).detach().cpu())
                # test_saved_feature["adj"].append(origin_adj[0])

        with open(self.saved_test_feature_path, 'wb') as file:
            pickle.dump(test_saved_feature, file)
        
        print("Stage 1 over")    

class STAGE_TWO(Train_model):
    def __init__(self, net, args, temperature, case_name, save_img=False, save_corres=False, save_w=False,
                 save_gt=False):
        print('set training dataset')
        base = os.path.join(args.stage2_feature_path, case_name)
        self.args = args
        self.saved_train_feature_path = self.find_feature(base, "train_feature_*.pkl")
        self.saved_val_feature_path = self.find_feature(base, "val_feature_*.pkl")
        self.saved_test_feature_path = self.find_feature(base, "test_feature_*.pkl")
        self.log_save_path = EXP_path+'/EXP2/{}/summary'.format(case_name)
        self.writer = SummaryWriter(self.log_save_path)
        self.case_name = case_name
        self.checkpoint_path = EXP_path+'/EXP2/{}/checkpoint'.format(case_name)
        self.max_point = args.max_length
        if not os.path.exists(self.log_save_path):
            os.makedirs(self.log_save_path, exist_ok=True)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path, exist_ok=True)

        # # DDP
        # torch.distributed.init_process_group(backend="nccl")
        # local_rank = torch.distributed.get_rank()
        # random_seed = 20
        # init_seeds(random_seed+torch.distributed.get_rank())
        # # local_rank = args.local_rank
        # self.device = torch.device("cuda", local_rank)
        # torch.cuda.set_device(local_rank)
        # models = net(args)

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # models = torch.nn.SyncBatchNorm.convert_sync_batchnorm(models)
        # models = models.to(device) 

        # self.models = DDP(models, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)



        # self.train_data, _ = self.set_dataset_searching(self.saved_train_feature_path, args)
        # self.train_sampler = DistributedSampler(self.train_data)
        # self.train_loader = DataLoader(self.train_data, args.batch_size, num_workers=0, sampler=self.train_sampler)
        # # self.train_loader = DataLoader(self.train_data, args.batch_size, num_workers=0, shuffle=True)

        # self.val_data, _ = self.set_dataset_searching(self.saved_val_feature_path, args)
        # self.val_loader = DataLoader(self.val_data, args.batch_size, num_workers=0, sampler=DistributedSampler(self.val_data))
        # # self.val_loader = DataLoader(self.val_data, args.batch_size, num_workers=0, shuffle=True)
        
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # ===== 单卡设备 =====
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ===== 初始化模型 =====
        models = net(args)
        models = models.to(self.device)

        self.models = models

        self.train_data, _ = self.set_dataset_searching(self.saved_train_feature_path, args)
        self.train_loader = DataLoader(self.train_data, args.matching_batch_size, shuffle=True, num_workers=0)

        self.val_data, _ = self.set_dataset_searching(self.saved_val_feature_path, args)
        self.val_loader = DataLoader(self.val_data, args.matching_batch_size, shuffle=False, num_workers=0)

        
        # self.models.to(self.device)
        self.contrast_temperature = args.contrast_temperature
        self.epoch = args.stage2_epoch

        self.optimizer = torch.optim.Adam(self.models.parameters(),
                                          lr=args.stage2_lr,
                                          weight_decay=args.stage2_weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.stage2_epoch)
        self.best_loss=float('inf')

    def find_feature(self, path, pattern):
        files = glob(os.path.join(path, pattern))
        print("Searching:", os.path.join(path, pattern))
        print("Found:", files)
        assert len(files) > 0, f"No file found for {pattern}"
        return sorted(files)[-1]

    def warmup(self, current_step: int):
        return 1 / (10 ** (float(self.args.warmup_steps - current_step)))
    
    def set_dataset_searching(self, data_path, args):
        with open(data_path, 'rb') as feature_file:
            stage1_features = pickle.load(feature_file)
        # gt_config['model_type'] = args.model_type
        # gt_config['channel'] = args.channel
        # gt_config['c_model'] = args.c_model
        # gt_config['patch_size'] = args.patch_size
        dataset = data_preprocess.MyDataSet_searching(stage1_features, args)
        return dataset, None
    def save_checkpoint(self, epoch, val_loss=None):
        state = {
            'epoch': epoch,
            'model_state_dict': self.models.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': getattr(self, 'best_loss', float('inf'))
        }

        # ===== latest =====
        torch.save(state, os.path.join(self.checkpoint_path, "latest.tar"))

        # ===== best =====
        if val_loss is not None and val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(state, os.path.join(self.checkpoint_path, "best.tar"))
            print(f"[Stage2] Save BEST model at epoch {epoch}, loss={val_loss:.4f}")
        
    def load_checkpoint(self):
        latest_path = os.path.join(self.checkpoint_path, "latest.tar")

        if not os.path.exists(latest_path):
            print(f'[Stage2] No checkpoint found at {self.checkpoint_path}')
            self.best_loss = float('inf')
            return 0

        print(f'[Stage2] Loaded checkpoint from: {latest_path}')
        checkpoint=torch.load(latest_path,map_location=self.device)

        self.models.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.best_loss = checkpoint.get('best_loss', float('inf'))

        return checkpoint['epoch'] + 1

    def get_max_file_number(self, directory):
        max_number = -1
        max_file = None
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                match = re.search(r'\d+', file)
                if match:
                    number = int(match.group())
                    if number > max_number:
                        max_number = number
                        max_file = file
        return max_file
    def calculate_train_val_top_recall(self, batch_feature_s, batch_feature_t):
        F_normalized_s = F.normalize(batch_feature_s, p=2, dim=1)
        F_normalized_t = F.normalize(batch_feature_t, p=2, dim=1)
        bs = batch_feature_s.shape[0]
        GT_pairs = []
        for i in range(0, bs):
            GT_pairs.append([i,i])
        cos_sim_matrix = torch.matmul(F_normalized_s, F_normalized_t.T)
        sort_matrix = torch.sort(cos_sim_matrix, dim=-1, descending=True)

        idx = sort_matrix[1]
        idx = idx.numpy() #（3279，3279）  gt_pair（2370，2）
        l = []
        for i in range(len(GT_pairs)):
            # if mins < len_all[i] <= maxs:
            l.append(np.argwhere(idx[GT_pairs[i][0]] == GT_pairs[i][1]))

        result = np.array(l).reshape(-1)
        top1 = (result < 1).sum() / len(l)
        top5 = (result < 5).sum() / len(l)
        top10 = (result < 10).sum() / len(l)
        top20 = (result < 20).sum() / len(l)

        return torch.tensor([top1]), torch.tensor([top5])
    
    def get_mask(self, logits, s_index, t_index):
        # s_index = GT_pairs[3]
        # t_index = GT_pairs[4]
        s_list = self.index_tensor(s_index)
        t_list =  self.index_tensor(t_index)

        mask = torch.eye(logits.shape[0], logits.shape[1], dtype=bool).to(logits.device)
        mask[s_list, t_list] = True

        return mask

    def index_tensor(self, tnsr):
        #一个batch后面如果重复出现同一个样本，使用第一次出现的位置的索引
        result = []
        for i in range(tnsr.shape[0]):
            if tnsr[i] in tnsr[:i]:
                # a = (tnsr == tnsr[i]).nonzero(as_tuple=True)[0][0]
                result.append((tnsr == tnsr[i]).nonzero(as_tuple=True)[0][0].item())
            else:
                result.append(i)
        return torch.tensor(result)
    
    def get_similarity_matrix(self, feature1, feature2, pad_mask):
        similarity_matrix = torch.bmm(feature1, feature2.permute(0, 2, 1)) / self.temperature
        similarity_matrix[pad_mask] -= 1e9  # give a very small value to the padded part for softmax operation
        s_i = torch.softmax(similarity_matrix, dim=1)  # row softmax
        s_j = torch.softmax(similarity_matrix, dim=-1)  # column softmax
        similarity_matrix = torch.multiply(s_i, s_j)

        return similarity_matrix
    
    def train_start(self):
        
        print('Stage 2 training start!!!')
        infor_loss_train=[]
        infor_loss_val = []

        start_epoch = self.load_checkpoint()
        InfoNCE_loss = InfoNCE(temperature=self.contrast_temperature)

        for i in range(start_epoch, self.epoch):

            infor_loss_train = []
            infor_loss_val = []

            self.models.train()

            for e, (stage1_features) in enumerate(tqdm(self.train_loader)):
                # self.train_sampler.set_epoch(e)
                all_data, mask_para = stage1_features
                stage1_features_s, stage1_features_t, index_s, index_t, pcd_s, pcd_t = all_data
                stage1_features_s, stage1_features_t, pcd_s, pcd_t = stage1_features_s.to(self.device), stage1_features_t.to(self.device), pcd_s.to(self.device), pcd_t.to(self.device)

                feature_s, _ = self.models(stage1_features_s, pcd_s)
                feature_t, _ = self.models(stage1_features_t, pcd_t)
                feature_s = F.normalize(feature_s, p=2, dim=1)
                feature_t = F.normalize(feature_t, p=2, dim=1)
                
                '''searching loss'''
                loss_s2t = InfoNCE_loss(feature_s, feature_t, gt_pairs=(index_s, index_t))
                loss_t2s = InfoNCE_loss(feature_t, feature_s, gt_pairs=(index_t, index_s))

                loss = (loss_s2t + loss_t2s) / 2
                reg = (feature_s.std() + feature_t.std())
                loss = loss - 0.01 * reg

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.models.parameters(), 5.0)
                self.optimizer.step()

                infor_loss_train.append(loss.item())
            self.scheduler.step()

            '''validation'''
            self.models.eval()
            with torch.no_grad():

                for _, (stage1_features) in enumerate(tqdm(self.val_loader)):

                    all_data, mask_para = stage1_features
                    stage1_features_s, stage1_features_t, index_s, index_t, pcd_s, pcd_t = all_data

                    stage1_features_s = stage1_features_s.to(self.device)
                    stage1_features_t = stage1_features_t.to(self.device)
                    pcd_s = pcd_s.to(self.device)
                    pcd_t = pcd_t.to(self.device)

                    feature_s, _ = self.models(stage1_features_s, pcd_s)
                    feature_t, _ = self.models(stage1_features_t, pcd_t)
                    feature_s = F.normalize(feature_s, p=2, dim=1)
                    feature_t = F.normalize(feature_t, p=2, dim=1)
                    val_loss = InfoNCE_loss(feature_s.detach(), feature_t.detach(), gt_pairs=(index_s, index_t))

                    infor_loss_val.append(val_loss.item())
            
            # ===== log =====
            train_loss=np.mean(infor_loss_train)
            val_loss=np.mean(infor_loss_val)

            self.writer.add_scalar('train_loss', train_loss, i)
            self.writer.add_scalar('val_loss', val_loss, i)

            # ===== save checkpoint =====
            self.save_checkpoint(i, val_loss)

            print(f"epoch={i}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")


class ST2_SearchModel(object):
    def __init__(self, net, args, temperature, case_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.case_name = case_name
        self.checkpoint_path = EXP_path+'/EXP2/{}/checkpoint'.format(case_name)
        base = os.path.join(args.stage2_feature_path, case_name)
        self.saved_test_feature_path = self.find_feature(base,"test_feature_*.pkl")
        self.test_data, self.Stage1_data = self.set_dataset_searching(self.saved_test_feature_path, args)
        self.test_loader = DataLoader(self.test_data, 1, num_workers=0, shuffle=False)
        self.models = net(args)
        self.models.to(self.device)
        self.models.eval()
        self.models.requires_grad_(False)
        self.max_point = args.max_length
        self.global_out_channels = args.global_out_channels


    def find_feature(self,path,pattern):
        files=glob(os.path.join(path,pattern))
        assert len(files)>0
        return sorted(files)[-1]

    def get_max_file_number(self, directory):
        max_number = -1
        max_file = None
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                match = re.search(r'\d+', file)
                if match:
                    number = int(match.group())
                    if number > max_number:
                        max_number = number
                        max_file = file
        return max_file
    
    def set_dataset_searching(self, data_path, args):
        with open(data_path, 'rb') as f:
            stage1_features = pickle.load(f)

        dataset = data_preprocess.MyDataSet_searching(stage1_features, args)
        return dataset, stage1_features

    def load_checkpoint_evl(self, checkpoint_path_evl):
        checkpoint = torch.load(checkpoint_path_evl,map_location=self.device)
        self.models.load_state_dict(checkpoint['model_state_dict'])
        return

    def feature_searching(self, result_matrix, gt_pair):
        """to get the topk searching result from score matrix"""
        sort_matrix = torch.sort(result_matrix, dim=-1, descending=True)
        idx = sort_matrix[1]
        idx=idx.cpu().numpy()
        l = []
        for i in range(len(gt_pair)):
            l.append(np.argwhere(idx[gt_pair[i][0]] == gt_pair[i][1]))

        result = np.array(l).reshape(-1)

        top1 = (result < 1).sum() / len(l)
        top5 = (result < 5).sum() / len(l)
        top10 = (result < 10).sum() / len(l)
        top20 = (result < 20).sum() / len(l)

        return top1, top5, top10, top20
    
    @staticmethod
    def get_concat_adj2(adj, max_len):
        device = adj.device
        temp_adj = torch.zeros((2, 0), dtype=torch.int).to(device)
        for i in range(len(adj)):
            b = torch.nonzero(adj[i]).transpose(0, 1)
            temp_adj = torch.hstack((temp_adj, b + i * max_len))

        return temp_adj
    
    
    def searching_start(self):
        best_checkpoint = "best.tar"
        print("best_checkpoint:{}".format(best_checkpoint))

        checkpoint_path_evl = EXP_path+'/EXP2/{}/checkpoint/{}'.format(self.case_name, best_checkpoint)
        self.load_checkpoint_evl(checkpoint_path_evl)

        self.stage2_result_path = EXP_path+'/EXP2/{}/result'.format(self.case_name)
        if os.path.exists(self.stage2_result_path) is False:
            os.mkdir(self.stage2_result_path)
        
        start_time = time.time()
        saved_test_weight = []
        print("searching start!")
        features_all = []
        self.models.eval()
        with torch.no_grad():

            for i in tqdm(range(len(self.test_data.stage1_feature))):
                feat=self.test_data.stage1_feature[i].unsqueeze(0).to(self.device)
                pcd=self.test_data.full_pcd_all[i].unsqueeze(0).to(self.device)
                F_s,w_s=self.models(feat,pcd)
                F_s=F.normalize(F_s,dim=1)
                features_all.append(F_s.cpu())
                saved_test_weight.append(w_s.cpu())

        features_all = torch.cat(features_all, dim=0)
        F_normalized = F.normalize(features_all, p=2, dim=1)

        cos_sim_matrix = torch.matmul(F_normalized, F_normalized.T)
        cos_sim_matrix.fill_diagonal_(-1)
        gt_pairs = np.array(self.test_data.GT_pairs)

        result = self.feature_searching(cos_sim_matrix, gt_pairs)
        print("Result:", result)
        end_time = time.time()
        print("Runtime:", end_time - start_time, "s")
        print("feature std:", features_all.std().item())
        print("feature mean:", features_all.mean().item())
        saved_matrix = {
            "matrix": cos_sim_matrix.data.cpu().numpy(),
            "GT_pairs": self.test_data.GT_pairs
        }
        saved_feature={
            "feature":features_all.numpy()
        }

        with open(self.stage2_result_path+'/sim_matrix_390_stage2_self_gate.pkl', 'wb') as f:
            pickle.dump(saved_matrix, f)
        with open(self.stage2_result_path+'/global_feature_390_stage2_self_gate.pkl', 'wb') as f:
            pickle.dump(saved_feature, f)
        searching_recall = result
        with open(self.stage2_result_path+'/searching recall.txt', 'w') as f:
            f.write('{}'.format(searching_recall))
        
        with open(self.stage2_result_path+'/saved_test_weight_390_stage2.pkl', 'wb') as file:
            pickle.dump(saved_test_weight, file)



    def searching_start_every_model(self):

        max_value = [0]

        for file in os.listdir(self.checkpoint_path):
            best_checkpoint = file
            self.feature_all_flatten = torch.zeros((0, self.global_out_channels))

            print("checkpoint:{}".format(best_checkpoint))

            checkpoint_path_evl = EXP_path+'/EXP2/{}/checkpoint/{}'.format(self.case_name, best_checkpoint).format(self.case_name, best_checkpoint)
            self.load_checkpoint_evl(checkpoint_path_evl)

            for batch, (stage1_features) in enumerate(tqdm(self.test_loader)):
                stage1_features, pcd = stage1_features
                stage1_features, pcd = stage1_features.to(self.device), pcd.to(self.device)


                F_global, _ = self.models(stage1_features, pcd)

                self.feature_all_flatten = torch.cat((self.feature_all_flatten, F_global.cpu()), dim=0)

            F_normalized = F.normalize(self.feature_all_flatten, p=2, dim=1)
            cos_sim_matrix = torch.matmul(F_normalized, F_normalized.T)
            cos_sim_matrix.fill_diagonal_(-1)
            result = self.feature_searching(cos_sim_matrix, self.Stage1_data['GT_pairs'])
            print(result)
            if result[0] > max_value[0]:
                max_value = result
        print("best test checkpoint:")
        print(result)


class Real_TestModel(Train_model):
    def __init__(self, net, args, temperature, case_name, save_img=True, save_corres=False, save_w=False,
                 save_gt=False):
        self.save_img = save_img
        self.save_corres = save_corres
        self.save_w = save_w
        self.save_gt = save_gt
        checkpoint_path = EXP_path+'/EXP/{}/checkpoint/'.format(case_name)
        best_checkpoint = self.get_max_file_number(checkpoint_path)
        self.checkpoint_path_evl = EXP_path+'/EXP/{}/checkpoint/{}'.format(case_name, best_checkpoint)
        self.case_name = case_name
        '''set testing dataset'''
        print('set testing dataset')
        self.test_data, self.test_set = self.set_real_dataset(args.real_test_set, args)
        self.test_loader = DataLoader(self.test_data, 1, shuffle=False)
        self.gt_pairs = self.test_set['GT_pairs']

        '''set testing model'''
        print('set testing model')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.temperature = temperature

        args.flattenNet_config['input_dim'] = args.patch_size ** 2
        self.models = net(args)
        self.models.to(self.device)
        self.models.eval()
        self.models.requires_grad_(False)
        self.args = args
    
    def set_real_dataset(self, data_path, args):
        with open(data_path, 'rb') as gt_file:
            gt_config = pickle.load(gt_file)

        gt_config['model_type'] = args.model_type
        gt_config['channel'] = args.channel
        gt_config['c_model'] = args.c_model
        gt_config['patch_size'] = args.patch_size
        dataset = data_preprocess.MyRealDataSet(gt_config, args)
        return dataset, gt_config
    
    def get_similarity_matrix_real_test(self, feature1, feature2, pad_mask):
        similarity_matrix = torch.bmm(feature1, feature2.permute(0, 2, 1)) / self.temperature
        similarity_matrix[pad_mask] -= 1e9  # give a very small value to the padded part for softmax operation
        s_i = torch.softmax(similarity_matrix, dim=1)  # row softmax
        s_j = torch.softmax(similarity_matrix, dim=-1)  # column softmax
        similarity_matrix = torch.multiply(s_i, s_j)

        return similarity_matrix
    

    def load_checkpoint_evl(self):
        checkpoint = torch.load(self.checkpoint_path_evl)
        self.models.load_state_dict(checkpoint['model_state_dict'])
        return
    
    def get_max_file_number(self, directory):
        max_number = -1
        max_file = None
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                match = re.search(r'\d+', file)
                if match:
                    number = int(match.group())
                    if number > max_number:
                        max_number = number
                        max_file = file
        return max_file
    
    def calculate_area_opencv(self, points):
        # 将点转换为numpy数组
        contour = np.array(points)
        
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        
        return area
    
    def cosine_similarity(self, vec1, vec2):
        # 计算两个向量的点积
        dot_product = np.sum(vec1 * vec2, axis=1)
        # 计算每个向量的L2范数（即向量的长度）
        norm_vec1 = np.linalg.norm(vec1, axis=1)
        norm_vec2 = np.linalg.norm(vec2, axis=1)
        # 计算余弦相似度
        cosine_similarity = dot_product / (norm_vec1 * norm_vec2)
        return cosine_similarity
    
    def calculate_ratio(self, mixed_feature, vec_c, vec_t):
        # 计算两个向量的和
        vec_contour = self.cosine_similarity(mixed_feature, vec_c)
        vec_texture = self.cosine_similarity(mixed_feature, vec_t)
        sum_vec = vec_contour + vec_texture
        # 计算第一个向量的每一个位置元素除以两个向量该位置的元素的和
        ratio = np.divide(vec_contour, sum_vec, out=np.zeros_like(vec_contour), where=sum_vec!=0)
        return ratio
    

    def test_start(self):
        self.load_checkpoint_evl()
        gt_pairs = self.gt_pairs
        device = self.device
        valid_nums4 = 0  # count the good registration nums
        valid_nums2 = 0
        valid_nums6 = 0
        c = 0  # count the fragment nums
        w_count = 0
        haus_list = []  # list of mean hausdroff distance of each gt pair

        '''test start'''
        print('test start!')
        saved_test_data = {
            "pred_real_transformation":[],
            # "GT_transformation":[],
        }
        saved_test_weight = []
        for batch, (mask_para, imgs, pcd, c_input, t_input, adjs, factors) in enumerate(tqdm(self.test_loader)):

            max_point_nums = pcd[0].shape[1]
            adj_s = self.get_concat_adj(adjs[0], max_point_nums)
            adj_t = self.get_concat_adj(adjs[1], max_point_nums)

            source_input = {
                'pcd': pcd[0].to(device), 'img': imgs[0].to(device), 'c_input': c_input[0].to(device),
                'adj': adj_s.to(device), 'factor': factors[0].to(device), 't_input': t_input[0].to(device)
            }

            target_input = {
                'pcd': pcd[1].to(device), 'img': imgs[1].to(device), 'c_input': c_input[1].to(device),
                'adj': adj_t.to(device), 'factor': factors[1].to(device), 't_input': t_input[1].to(device)
            }

            pad_mask = self.get_pad_mask(mask_para).to(device)  # mark the padded part in similarity matrix
            # mask = mask_para[0].to(device)
            feature_s, concat_source, w_s = self.models(source_input)
            feature_t, concat_target, w_t = self.models(target_input)
            similarity_matrix = self.get_similarity_matrix_real_test(feature_s, feature_t, pad_mask)


            '''visualization part'''
            similarity_matrix = similarity_matrix[0].cpu().numpy()
            kernel = np.eye(3, dtype=np.uint8)
            kernel[1, 1] = 0
            kernel = np.rot90(kernel)
            similarity_matrix = cv2.erode(similarity_matrix, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            kernel[1, 1] = 1
            similarity_matrix = cv2.dilate(similarity_matrix, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)

            idx_s, idx_t = gt_pairs[batch]
            s_pcd_origin, t_pcd_origin = self.test_set['full_pcd_all'][idx_s], self.test_set['full_pcd_all'][idx_t]
            # s_pcd, t_pcd = self.test_set['down_sample_pcd'][idx_s], self.test_set['down_sample_pcd'][idx_t]
            s_pcd, t_pcd = self.test_set['full_pcd_all'][idx_s], self.test_set['full_pcd_all'][idx_t]
            # ind_s_origin, ind_t_origin = self.test_set['source_ind'][batch], self.test_set['target_ind'][batch]
            source_img, target_img = self.test_set['img_all'][idx_s], self.test_set['img_all'][idx_t]
            img_save_path = EXP_path+'/EXP/{}/result/real_img'.format(self.case_name)
            corres_save_path = EXP_path+'/EXP/{}/result/corres'.format(self.case_name)
            os.makedirs(img_save_path, exist_ok=True)
            os.makedirs(corres_save_path, exist_ok=True)
            evl = visualization.Visualization_real_data(similarity_matrix, s_pcd, t_pcd, source_img,
                                              target_img, s_pcd_origin, t_pcd_origin, conv_threshold=0.006) 

            transformation, pairs = evl.get_transformation()


            # get weighted img
            # w_s = 1- w_s
            if self.save_w:
                img_s = source_img.transpose(1, 0, 2)
                img_s = np.ascontiguousarray(img_s)
                evl.img_s = evl.weight_visualize(os.path.join(img_save_path, 'w_s{}.png'.format(batch)),
                                                 img_s, s_pcd, w_s[0].detach().cpu().numpy())


                img_t = target_img.transpose(1, 0, 2)
                img_t = np.ascontiguousarray(img_t)
                evl.img_t = evl.weight_visualize(os.path.join(img_save_path, 'w_t{}.png'.format(batch)),
                                                 img_t, t_pcd, w_t[0].detach().cpu().numpy())

            # save predicted result pairs
            if self.save_img:
                evl.get_img(os.path.join(img_save_path, 'pred{}.png'.format(batch)), transformation)

        with open(EXP_path+'/EXP/{}/result/saved_real_test_data.pkl'.format(self.case_name), 'wb') as file:
            pickle.dump(saved_test_data, file)



class STAGE_ONE_REAL(Train_model):
    def __init__(self, net, args, temperature, case_name, save_img=False, save_corres=False, save_w=False,
                 save_gt=False):
        self.save_img = save_img
        self.save_corres = save_corres
        # self.save_w = save_w
        # self.save_gt = save_gt
        checkpoint_path = EXP_path+'/EXP/{}/checkpoint/'.format(case_name)
        best_checkpoint = self.get_max_file_number(checkpoint_path)
        self.checkpoint_path_evl = EXP_path+'/EXP/{}/checkpoint/{}'.format(case_name, best_checkpoint)
        self.case_name = case_name
        if os.path.exists(EXP_path+'/EXP2/{}'.format(case_name)) is False:
            os.makedirs(EXP_path+'/EXP2/{}'.format(case_name))
        feature_save_path = args.real_stage2_feature_path+"/{}".format(case_name)
        self.saved_train_feature_path = feature_save_path+'/real_img_feature_{}.pkl'.format(args.dataset_select)

        if os.path.exists(feature_save_path) is False:
            os.makedirs(feature_save_path)
        
        '''set train dataset'''
        print('set training dataset')
        self.train_data, self.train_GT = self.set_real_dataset(args.real_test_set, args)
        self.train_loader = DataLoader(self.train_data, 1, num_workers=0,shuffle=False)
        self.train_gt_pairs = self.train_GT['GT_pairs']
        self.train_pcd = self.train_GT['full_pcd_all']


        '''set testing model'''
        print('set stage1 model')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.temperature = temperature

        args.flattenNet_config['input_dim'] = args.patch_size ** 2
        self.models = net(args)
        self.models.to(self.device)
        self.models.eval()
        self.models.requires_grad_(False)
        self.args = args

    def set_real_dataset(self, data_path, args):
        with open(data_path, 'rb') as gt_file:
            gt_config = pickle.load(gt_file)

        gt_config['model_type'] = args.model_type
        gt_config['channel'] = args.channel
        gt_config['c_model'] = args.c_model
        gt_config['patch_size'] = args.patch_size
        dataset = data_preprocess.MyRealDataSet(gt_config, args)
        return dataset, gt_config


    def load_checkpoint_evl(self):
        checkpoint = torch.load(self.checkpoint_path_evl)
        self.models.load_state_dict(checkpoint['model_state_dict'])
        return
    
    def get_max_file_number(self, directory):
        max_number = -1
        max_file = None
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                match = re.search(r'\d+', file)
                if match:
                    number = int(match.group())
                    if number > max_number:
                        max_number = number
                        max_file = file
        return max_file

    def stage1_start(self):
        self.load_checkpoint_evl()
        device = self.device
        '''save train feature'''
        train_saved_feature = {
            "saved_feature": [],
            "GT_pairs":self.train_gt_pairs,
            "full_pcd":self.train_pcd,
        }
        for batch, (pcd, imgs, t_input, adj, factor, c_input) in enumerate(tqdm(self.train_loader)):
            max_point_nums = len(pcd[0])

            adj = self.get_concat_adj2(adj, max_point_nums)
            inputs = {
                'pcd': pcd.to(device), 'img': imgs.to(device), 't_input': t_input.to(device),
                'adj': adj.to(device), 'factor': factor.to(device), 'c_input': c_input.to(device)
            }

            matching_feature, feature, _ = self.models(inputs) # bs,2611,64
            train_saved_feature["saved_feature"].append(feature.detach().cpu())
            
        
        with open(self.saved_train_feature_path, 'wb') as file:
            pickle.dump(train_saved_feature, file)

        
        print("Stage 1 Real over")

class ST2_Real_SearchModel(object):
    def __init__(self, net, args, temperature, case_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.case_name = case_name
        self.checkpoint_path = EXP_path+'/EXP2/{}/checkpoint'.format(case_name)
        self.saved_test_feature_path = args.real_stage2_feature_path+'/{}/real_img_feature_{}.pkl'.format(case_name, args.dataset_select)
        self.test_data, self.Stage1_data = self.set_dataset_searching(self.saved_test_feature_path, args)
        self.test_loader = DataLoader(self.test_data, 1, num_workers=0, shuffle=False)
        self.models = net(args)
        self.models.to(self.device)
        self.models.eval()
        self.models.requires_grad_(False)
        self.feature_all_flatten = torch.zeros((0, args.global_out_channels))
        self.max_point = args.max_length


    def get_max_file_number(self, directory):
        max_number = -1
        max_file = None
        for file in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, file)):
                match = re.search(r'\d+', file)
                if match:
                    number = int(match.group())
                    if number > max_number:
                        max_number = number
                        max_file = file
        return max_file
    
    def set_dataset_searching(self, data_path, args):
        with open(data_path, 'rb') as feature_file:
            stage1_features = pickle.load(feature_file)
        dataset = data_preprocess.MyRealDataSet_searching(stage1_features, args)
        return dataset, stage1_features

    def load_checkpoint_evl(self, checkpoint_path_evl):
        checkpoint = torch.load(checkpoint_path_evl)
        self.models.load_state_dict(checkpoint['model_state_dict'])
        return

    def feature_searching(self, result_matrix, gt_pair):
        """to get the topk searching result from score matrix"""
        # result_matrix = result_matrix + result_matrix.T
        sort_matrix = torch.sort(result_matrix, dim=-1, descending=True)
        # sort_matrix = torch.sort(result_matrix, dim=-1, descending=False)
        idx = sort_matrix[1]
        idx = idx.numpy() #（3279，3279）  gt_pair（2370，2）
        l = []
        bad_list = []
        for i in range(len(gt_pair)):
            # if mins < len_all[i] <= maxs:
            l.append(np.argwhere(idx[gt_pair[i][0]] == gt_pair[i][1]))
            a = int(np.argwhere(idx[gt_pair[i][0]] == gt_pair[i][1]))
            if a > 20:
                bad_list.append(i)

        result = np.array(l).reshape(-1)

        top1 = (result < 1).sum() / len(l)
        top5 = (result < 5).sum() / len(l)
        top10 = (result < 10).sum() / len(l)
        top20 = (result < 20).sum() / len(l)

        return top1, top5, top10, top20
    
    @staticmethod
    def get_concat_adj2(adj, max_len):
        device = adj.device
        temp_adj = torch.zeros((2, 0), dtype=torch.int).to(device)
        for i in range(len(adj)):
            b = torch.nonzero(adj[i]).transpose(0, 1)
            # a = adj[i].coalesce().indices() #(2,8602)
            temp_adj = torch.hstack((temp_adj, b + i * max_len))

        return temp_adj
    
    
    def searching_start(self):
        best_checkpoint = self.get_max_file_number(self.checkpoint_path)
        print("best_checkpoint:{}".format(best_checkpoint))

        checkpoint_path_evl = EXP_path+'/EXP2/{}/checkpoint/{}'.format(self.case_name, best_checkpoint).format(self.case_name, best_checkpoint)
        self.load_checkpoint_evl(checkpoint_path_evl)

        for batch, (stage1_features) in enumerate(tqdm(self.test_loader)):
            stage1_features, pcd = stage1_features
            stage1_features, pcd = stage1_features.to(self.device), pcd.to(self.device)

            max_point_nums = self.max_point
            # adj = self.get_concat_adj2(adj, max_point_nums)
            
            F_global, _ = self.models(stage1_features, pcd)

            self.feature_all_flatten = torch.cat((self.feature_all_flatten, F_global.cpu()), dim=0)

        F_normalized = F.normalize(self.feature_all_flatten, p=2, dim=1)
        # 计算余弦相似度矩阵
        cos_sim_matrix = torch.matmul(F_normalized, F_normalized.T)
        cos_sim_matrix.fill_diagonal_(-1)
        result = self.feature_searching(cos_sim_matrix, self.Stage1_data['GT_pairs'])
        print(result)

        calute_NDCG(cos_sim_matrix, self.Stage1_data['GT_pairs'])


def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_deterministic:  
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:  
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    opt = config.args

    '''set 温度系数'''
    temp = math.sqrt(opt.feature_dim)

    net = pipeline.Vanilla
    ST2_net = pipeline.TransformerEncoderModel

    exp_name = 'exp1' 

    EXP_path = opt.exp_path
    if opt.model_type == 'matching_train': 
        trainer = Train_model(net, opt, temp, exp_name)
        trainer.train_start()
    elif opt.model_type == 'matching_test': 
        tester = TestModel(net, opt, temp, exp_name)
        tester.test_start()
    elif opt.model_type == 'save_stage1_feature': 
        ST1 = STAGE_ONE(net, opt, temp, exp_name)
        ST1.stage1_start()
    elif opt.model_type == 'searching_train':
        ST2 = STAGE_TWO(ST2_net, opt, temp, exp_name)
        ST2.train_start()
    elif opt.model_type == 'searching_test':
        ST2_searcher = ST2_SearchModel(ST2_net, opt, temp, exp_name)
        ST2_searcher.searching_start()

    elif opt.model_type == 'real_dataset_test':
        Real_tester = Real_TestModel(net, opt, temp, exp_name)
        Real_tester.test_start()
    elif opt.model_type == 'stage1_real':  # --------------------stage1:save feature---------------------
        ST1 = STAGE_ONE_REAL(net, opt, temp, exp_name)
        ST1.stage1_start()
    elif opt.model_type == 'stage2_real_searching':
        ST2_searcher = ST2_Real_SearchModel(ST2_net, opt, temp, exp_name)
        ST2_searcher.searching_start()
