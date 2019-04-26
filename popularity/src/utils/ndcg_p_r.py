from numpy import *

def eval_NDCG_P_R(pred, label, k_list = array([10])): # k_list must in order
    k_list = sort(k_list)
    sorted_idx = argsort(-pred)
    num_positive = sum(label==1)
    DCG, IDCG = zeros(k_list.size,dtype=float32), zeros(k_list.size,dtype=float32)
    P = zeros(k_list.size, dtype=float32)
    R = zeros(k_list.size, dtype=float32)
    for rank in range(k_list[-1]):
        for i in range(k_list.size):
            if rank < k_list[i]:
                if label[sorted_idx[rank]] == 1:
                    DCG[i] += 1.0/log(rank+2)
                    P[i] += 1.0/k_list[i]
                    R[i] += 1.0/num_positive
                if rank < num_positive:
                    IDCG[i] += 1.0/log(rank+2)
    return DCG/IDCG, P, R