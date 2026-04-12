import argparse
import os

flattenNet_config = {
    'input_dim': 9,
    'hidden': 32,
    'output_dim': 64,
    'dropout': 0.0,
    'k': 3
}

args = argparse.ArgumentParser()

# =========================
# model config
# =========================
args.add_argument('--patch_size', type=int, default=7)
args.add_argument('--c_model', type=str, default='l')
args.add_argument('--loss', type=str, default='focal')
args.add_argument('--feature_dim', type=int, default=64)

# =========================
# GCN setting
# =========================
args.add_argument('--k', default=16, type=int)
args.add_argument('--block', default='res+')
args.add_argument('--act', default='relu')
args.add_argument('--norm', default='batch')
args.add_argument('--bias', default=True)
args.add_argument('--n_filters', default=64, type=int)
args.add_argument('--n_blocks', default=14, type=int)
args.add_argument('--in_channels', default=64, type=int)
args.add_argument('--dropout', default=0.2, type=float)
args.add_argument('--gat_head', default=1, type=int)

# =========================
# G-Unet setting
# =========================
args.add_argument('-ks', nargs='+', type=float, default=[0.9, 0.8, 0.7])

# =========================
# dilated knn
# =========================
args.add_argument('--epsilon', default=0.2, type=float)
args.add_argument('--stochastic', default=True, type=bool)

# =========================
# training parameter
# =========================
args.add_argument('--flattenNet_config', default=flattenNet_config)
args.add_argument('--channel', type=int, default=3)
args.add_argument('--epoch', type=int, default=128)
args.add_argument('--lr', type=float, default=1e-3)
args.add_argument('--weight_decay', type=float, default=5e-4)
args.add_argument('--matching_batch_size', type=int, default=25)
args.add_argument('--load_checkpoint', type=bool, default=False)
args.add_argument('--max_length', type=int, default=2900)

# =========================
# stage2 config
# =========================
args.add_argument('--n_blocks_stage2', default=12, type=int)
args.add_argument('--block_stage2', default='res+')
args.add_argument('--n_filters_stage2', default=128, type=int)
args.add_argument('--gat_head_stage2', default=1, type=int)
args.add_argument('--matching_channels_stage2', default=128, type=int)
args.add_argument('--stage2_matching_weight', default=0.05, type=float)
args.add_argument('--stage2_lr', type=float, default=1e-3)
args.add_argument('--stage2_epoch', type=int, default=128)
args.add_argument('--warmup_steps', type=int, default=10)
args.add_argument('--stage2_weight_decay', type=float, default=1e-3)

args.add_argument('--global_out_channels', default=128, type=int)
args.add_argument('--contrast_weight', default=1, type=float)
args.add_argument('--contrast_temperature', default=0.12, type=float)
args.add_argument('--local_rank', default=0, type=int)

args.add_argument('--embed_dim', default=768, type=int)
args.add_argument('--vit_length', default=196, type=int)
args.add_argument('--tranct_length', default=1408, type=int)

# =========================
# 🚀 核心修复：数据路径统一管理
# =========================

# 数据集根目录
DATA_ROOT = "./Fragments-dataset"

args.add_argument('--data_root', type=str, default=DATA_ROOT)

args.add_argument('--train_set', type=str,
                  default=os.path.join(DATA_ROOT, 'train_set_with_downsample.pkl'))

args.add_argument('--valid_set', type=str,
                  default=os.path.join(DATA_ROOT, 'valid_set_with_downsample.pkl'))

args.add_argument('--test_set', type=str,
                  default=os.path.join(DATA_ROOT, 'test_set_with_downsample.pkl'))

args.add_argument('--search_set', type=str,
                  default=os.path.join(DATA_ROOT, 'test_set_with_downsample.pkl'))

# =========================
# stage2 dataset
# =========================

stage2_data_model = "merged"

args.add_argument('--stage2_data_model', type=str, default=stage2_data_model)

args.add_argument('--stage2_feature_path', type=str,
                  default='./stage1_feature/{}'.format(stage2_data_model))

args.add_argument('--in_channels_stage2', default=128, type=int)

# =========================
# exp config
# =========================

args.add_argument('--exp_path', type=str, default="./")
args.add_argument('--model_type', type=str, default='searching_test')

# =========================
# parse args
# =========================

args = args.parse_args()

# override
args.in_channels = args.feature_dim