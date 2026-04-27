import argparse
import os

flattenNet_config = {
    'input_dim': 9,
    'hidden': 32,
    'output_dim': 64,
    'dropout': 0.0,
    'k': 3
}

parser = argparse.ArgumentParser()

# =========================
# model config
# =========================
parser.add_argument('--patch_size', type=int, default=7)
parser.add_argument('--c_model', type=str, default='l')
parser.add_argument('--loss', type=str, default='focal')
parser.add_argument('--feature_dim', type=int, default=64)

# =========================
# GCN setting
# =========================
parser.add_argument('--k', default=16, type=int)
parser.add_argument('--block', default='res+')
parser.add_argument('--act', default='relu')
parser.add_argument('--norm', default='batch')
parser.add_argument('--bias', default=True)
parser.add_argument('--n_filters', default=64, type=int)
parser.add_argument('--n_blocks', default=14, type=int)
parser.add_argument('--in_channels', default=64, type=int)
parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--gat_head', default=1, type=int)

# =========================
# G-Unet setting
# =========================
parser.add_argument('-ks', nargs='+', type=float, default=[0.9, 0.8, 0.7])

# =========================
# dilated knn
# =========================
parser.add_argument('--epsilon', default=0.2, type=float)
parser.add_argument('--stochastic', default=True, type=bool)

# =========================
# training parameter
# =========================
parser.add_argument('--flattenNet_config', default=flattenNet_config)
parser.add_argument('--channel', type=int, default=3)
parser.add_argument('--epoch', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--matching_batch_size', type=int, default=32)
parser.add_argument('--load_checkpoint', type=bool, default=False)
parser.add_argument('--max_length', type=int, default=2900)

# =========================
# stage2 config
# =========================
parser.add_argument('--n_blocks_stage2', default=16, type=int)
parser.add_argument('--block_stage2', default='res+')
parser.add_argument('--n_filters_stage2', default=128, type=int)
parser.add_argument('--gat_head_stage2', default=1, type=int)
parser.add_argument('--matching_channels_stage2', default=128, type=int)
parser.add_argument('--stage2_matching_weight', default=0.05, type=float)
parser.add_argument('--stage2_lr', type=float, default=1e-3)
parser.add_argument('--stage2_epoch', type=int, default=128)
parser.add_argument('--warmup_steps', type=int, default=10)
parser.add_argument('--stage2_weight_decay', type=float, default=1e-3)

parser.add_argument('--global_out_channels', default=128, type=int)
parser.add_argument('--contrast_weight', default=1, type=float)
parser.add_argument('--contrast_temperature', default=0.07, type=float)
parser.add_argument('--local_rank', default=0, type=int)

parser.add_argument('--embed_dim', default=768, type=int)
parser.add_argument('--vit_length', default=196, type=int)
parser.add_argument('--tranct_length', default=1408, type=int)

# =========================
# 🚀 核心修复：数据路径统一管理
# =========================

# 数据集根目录
# DATA_ROOT = "../data/pkl"
DATA_ROOT = "../Fragments-dataset/Fragments-dataset/"

parser.add_argument('--data_root', type=str, default=DATA_ROOT)

# parser.add_argument('--train_set', type=str,
#                   default=os.path.join(DATA_ROOT, 'ori_train_set.pkl'))

# parser.add_argument('--valid_set', type=str,
#                   default=os.path.join(DATA_ROOT, 'ori_valid_set.pkl'))

# parser.add_argument('--test_set', type=str,
#                   default=os.path.join(DATA_ROOT, 'ori_test_set.pkl'))

# parser.add_argument('--search_set', type=str,
#                   default=os.path.join(DATA_ROOT, 'ori_test_set.pkl'))

parser.add_argument('--train_set', type=str,
    default=os.path.join(DATA_ROOT, 'train_set_with_downsample.pkl'))

parser.add_argument('--valid_set', type=str,
    default=os.path.join(DATA_ROOT, 'valid_set_with_downsample.pkl'))

parser.add_argument('--test_set', type=str,
    default=os.path.join(DATA_ROOT, 'test_set_with_downsample.pkl'))

parser.add_argument('--search_set', type=str,
    default=os.path.join(DATA_ROOT, 'test_set_with_downsample.pkl'))

parser.add_argument('--dataset_select', type=str, default='circle_sample_V5_2')

# =========================
# stage2 dataset
# =========================

stage2_data_model = "merged"

parser.add_argument('--stage2_data_model', type=str, default=stage2_data_model)

parser.add_argument('--stage2_feature_path', type=str,
                  default='./stage1_feature/{}'.format(stage2_data_model))

parser.add_argument('--in_channels_stage2', default=128, type=int)

# =========================
# exp config
# =========================

parser.add_argument('--exp_path', type=str, default="./")
parser.add_argument('--model_type', type=str, default='searching_test')

# =========================
# parse args
# =========================

args = parser.parse_args()

# override
args.in_channels = args.feature_dim