import numpy as np


def dcg_at_k(scores, k):
    """
    DCG@k
    scores: relevance list (0/1 或连续值)
    """
    scores = np.asarray(scores)[:k]
    return np.sum(scores / np.log2(np.arange(2, scores.shape[0] + 2)))


def ndcg_at_k(scores, k):
    """
    标准 NDCG@k（单个query版本）
    """
    scores = np.asarray(scores)

    ideal = np.sort(scores)[::-1]
    best = dcg_at_k(ideal, k)
    if best == 0:
        return 0.0
    return dcg_at_k(scores, k) / best

def calute_NDCG(sim_matrix, gt_pairs, k=10):
    """
    用于检索任务（PairingNet）

    sim_matrix: (N, N) 相似度矩阵
    gt_pairs: list[list[int]]
              每个 query 对应正确匹配的 index（通常是1个或少量）

    return: mean NDCG@k
    """

    sim_matrix = np.asarray(sim_matrix)
    N = sim_matrix.shape[0]

    ndcg_list = []

    for i in range(N):
        scores = sim_matrix[i]

        # top-k index
        topk_idx = np.argsort(scores)[::-1][:k]

        # DCG
        dcg = 0.0
        for rank, idx in enumerate(topk_idx):
            if isinstance(gt_pairs[i], (list, tuple, set)):
                hit = idx in gt_pairs[i]
            else:
                hit = idx == gt_pairs[i]

            if hit:
                dcg = 1.0 / np.log2(rank + 2)
                break

        # IDCG（理想情况：只有一个正确匹配）
        idcg = 1.0

        ndcg_list.append(dcg / idcg)

    return float(np.mean(ndcg_list))


# ================================
#  兼容旧代码接口
# ================================
def ndcg_test(sim_matrix, gt_pairs, k=10):
    return calute_NDCG(sim_matrix, gt_pairs, k)


# ================================
# 测试 demo（可删）
# ================================
if __name__ == "__main__":
    results = np.random.rand(20, 10)
    gt = [[i % 10] for i in range(20)]

    print("NDCG@10:", calute_NDCG(results, gt, k=10))