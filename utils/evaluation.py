import time
import numpy as np
import torch
torch.set_printoptions(precision=8)


def compute_hyperbolic_distances(vectors_u, vectors_v):
    """
    Compute  distances between input vectors.
    Modified based on gensim code.
    vectors_u: (batch_size, dim)
    vectors_v: (batch_size, dim)
    """
    euclidean_dists = np.linalg.norm(vectors_u - vectors_v, axis=1)  # (batch_size, )
    return euclidean_dists  # (batch_size, )


def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range

def compute_hyperbolic_similarity(embeds1, embeds2):
    x1, y1 = embeds1.shape  # <class 'numpy.ndarray'>
    x2, y2 = embeds2.shape
    assert y1 == y2
    dist_vec_list = list()
    for i in range(x1):
        embed1 = embeds1[i, ]  # <class 'numpy.ndarray'> (y1,)
        embed1 = np.reshape(embed1, (1, y1))  # (1, y1)
        embed1 = np.repeat(embed1, x2, axis=0)  # (x2, y1)
        dist_vec = compute_hyperbolic_distances(embed1, embeds2)
        dist_vec_list.append(dist_vec)
    dis_mat = np.row_stack(dist_vec_list)  # (x1, x2)
    return normalization(-dis_mat)

def cal_rank_hyperbolic(frags, sub_embed, embed, multi_types_list, top_k, greedy):
    onto_number = embed.shape[0]
    mr = 0
    mrr = 0
    hits = np.array([0 for _ in top_k])
    sim_mat = compute_hyperbolic_similarity(sub_embed, embed)
    results = set()
    test_num = sub_embed.shape[0]
    for i in range(len(frags)):
        ref = frags[i]
        rank = (-sim_mat[i, :]).argsort()
        aligned_e = rank[0]
        results.add((ref, aligned_e))
        multi_types = multi_types_list[ref]
        if greedy:
            rank_index = onto_number
            for item in multi_types:
                temp_rank_index = np.where(rank == item)[0][0]
                rank_index = min(temp_rank_index, rank_index)
            mr += (rank_index + 1)
            mrr += 1 / (rank_index + 1)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    hits[j] += 1
        else:
            for item in multi_types:
                rank_index = np.where(rank == item)[0][0]
                mr += (rank_index + 1)
                mrr += 1 / (rank_index + 1)
                for j in range(len(top_k)):
                    if rank_index < top_k[j]:
                        hits[j] += 1
            test_num += (len(multi_types) - 1)
    return mr, mrr, hits, results, test_num

def eval_type_hyperbolic(embed1, embed2, ent_type, top_k,  greedy=True):
    
    ref_num = len(embed1)
    hits = np.array([0 for _ in top_k])
    mr = 0
    mrr = 0
    total_test_num = 0
    total_alignment = set()

    frags = np.array(range(ref_num))
    results=cal_rank_hyperbolic(frags, embed1, embed2, ent_type, top_k, greedy)

    mr, mrr, hits, total_alignment, total_test_num = results


    if greedy:
        assert total_test_num == ref_num
    else:
        print("multi types:", total_test_num - ref_num)

    hits = hits / total_test_num
    for i in range(len(hits)):
        hits[i] = round(hits[i], 4)
    mr /= total_test_num
    mrr /= total_test_num

    eval_performance=dict()
    eval_performance['mrr']=mrr
    eval_performance['hits1']=hits[0]
    eval_performance['hits3']=hits[1]
    eval_performance['hits5']=hits[2]
    eval_performance['hits10']=hits[3]

    return eval_performance

def batch_evaluation( batch_results, all_facts, gt_dict,ret_ranks,device):
    """
    Perform batch evaluation.
    """
    for i, result in enumerate(batch_results):
        target = all_facts[3][i]
        pos = all_facts[2][i]
        key = " ".join([
            str(all_facts[0][i][x].item()) for x in range(len(all_facts[0][i]))
            if x != pos
        ])

        # filtered setting
        rm_idx = torch.tensor(gt_dict[pos.item()][key]).to(device)
        rm_idx=torch.where(rm_idx!=target,rm_idx,0)
        result.index_fill_(0,rm_idx,-np.Inf)

        sortidx = torch.argsort(result,dim=-1,descending=True)

        if all_facts[4][i] == 1:
            ret_ranks['entity']=torch.cat([ret_ranks['entity'],(torch.where(sortidx == target)[0] + 1)],dim=0)
        elif all_facts[4][i] == -1:
            ret_ranks['relation']=torch.cat([ret_ranks['relation'],(torch.where(sortidx == target)[0]+ 1)],dim=0)
        else:
            raise ValueError("Invalid `feature.mask_type`.")

        if torch.sum(all_facts[1][i]) == 3:
            if pos == 1:
                ret_ranks['2-r']=torch.cat([ret_ranks['2-r'],(torch.where(sortidx == target)[0] + 1)],dim=0)
            elif pos == 0 or pos == 2:
                ret_ranks['2-ht']=torch.cat([ret_ranks['2-ht'],(torch.where(sortidx == target)[0] + 1)],dim=0)
            else:
                raise ValueError("Invalid `feature.mask_position`.")
        elif torch.sum(all_facts[1][i]) > 3:
            if pos == 1:
                ret_ranks['n-r']=torch.cat([ret_ranks['n-r'],(torch.where(sortidx == target)[0]+ 1)],dim=0)
            elif pos == 0 or pos == 2:
                ret_ranks['n-ht']=torch.cat([ret_ranks['n-ht'],(torch.where(sortidx == target)[0]+ 1)],dim=0)
            elif pos > 2 and all_facts[4][i] == -1:
                ret_ranks['n-a']=torch.cat([ret_ranks['n-a'],(torch.where(sortidx == target)[0]+ 1)],dim=0)
            elif pos > 2 and all_facts[4][i] == 1:
                ret_ranks['n-v']=torch.cat([ret_ranks['n-v'],(torch.where(sortidx == target)[0]+ 1)],dim=0)
            else:
                raise ValueError("Invalid `feature.mask_position`.")
        else:
            raise ValueError("Invalid `feature.arity`.")
    return ret_ranks

def compute_metrics(ret_ranks):
    """
    Combine the ranks from batches into final metrics.
    """

    all_ent_ranks = ret_ranks['entity']
    all_rel_ranks = ret_ranks['relation']
    _2_r_ranks = ret_ranks['2-r']
    _2_ht_ranks = ret_ranks['2-ht']
    _n_r_ranks = ret_ranks['n-r']
    _n_ht_ranks = ret_ranks['n-ht']
    _n_a_ranks = ret_ranks['n-a']
    _n_v_ranks = ret_ranks['n-v']
    all_r_ranks = torch.cat([ret_ranks['2-r'],ret_ranks['n-r']],dim=0)
    all_ht_ranks =  torch.cat([ret_ranks['2-ht'],ret_ranks['n-ht']],dim=0)

    mrr_ent = torch.mean(1.0 / all_ent_ranks).item()
    hits1_ent = torch.mean(torch.where(all_ent_ranks <= 1.0,1.0,0.0)).item()
    hits3_ent = torch.mean(torch.where(all_ent_ranks <= 3.0,1.0,0.0)).item()
    hits5_ent = torch.mean(torch.where(all_ent_ranks <= 5.0,1.0,0.0)).item()
    hits10_ent = torch.mean(torch.where(all_ent_ranks <= 10.0,1.0,0.0)).item()

    mrr_rel = torch.mean(1.0 / all_rel_ranks).item()
    hits1_rel = torch.mean(torch.where(all_rel_ranks <= 1.0,1.0,0.0)).item()
    hits3_rel = torch.mean(torch.where(all_rel_ranks <= 3.0,1.0,0.0)).item()
    hits5_rel = torch.mean(torch.where(all_rel_ranks <= 5.0,1.0,0.0)).item()
    hits10_rel = torch.mean(torch.where(all_rel_ranks <= 10.0,1.0,0.0)).item()

    mrr_2r = torch.mean(1.0 / _2_r_ranks).item()
    hits1_2r = torch.mean(torch.where(_2_r_ranks <= 1.0,1.0,0.0)).item()
    hits3_2r = torch.mean(torch.where(_2_r_ranks <= 3.0,1.0,0.0)).item()
    hits5_2r = torch.mean(torch.where(_2_r_ranks <= 5.0,1.0,0.0)).item()
    hits10_2r = torch.mean(torch.where(_2_r_ranks <= 10.0,1.0,0.0)).item()

    mrr_2ht = torch.mean(1.0 / _2_ht_ranks).item()
    hits1_2ht = torch.mean(torch.where(_2_ht_ranks <= 1.0,1.0,0.0)).item()
    hits3_2ht = torch.mean(torch.where(_2_ht_ranks <= 3.0,1.0,0.0)).item()
    hits5_2ht = torch.mean(torch.where(_2_ht_ranks <= 5.0,1.0,0.0)).item()
    hits10_2ht = torch.mean(torch.where(_2_ht_ranks <= 10.0,1.0,0.0)).item()

    mrr_nr = torch.mean(1.0 / _n_r_ranks).item()
    hits1_nr = torch.mean(torch.where(_n_r_ranks <= 1.0,1.0,0.0)).item()
    hits3_nr = torch.mean(torch.where(_n_r_ranks <= 3.0,1.0,0.0)).item()
    hits5_nr = torch.mean(torch.where(_n_r_ranks <= 5.0,1.0,0.0)).item()
    hits10_nr = torch.mean(torch.where(_n_r_ranks <= 10.0,1.0,0.0)).item()

    mrr_nht = torch.mean(1.0 / _n_ht_ranks).item()
    hits1_nht = torch.mean(torch.where(_n_ht_ranks <= 1.0,1.0,0.0)).item()
    hits3_nht = torch.mean(torch.where(_n_ht_ranks <= 3.0,1.0,0.0)).item()
    hits5_nht = torch.mean(torch.where(_n_ht_ranks <= 5.0,1.0,0.0)).item()
    hits10_nht = torch.mean(torch.where(_n_ht_ranks <= 10.0,1.0,0.0)).item()

    mrr_na = torch.mean(1.0 / _n_a_ranks).item()
    hits1_na = torch.mean(torch.where(_n_a_ranks <= 1.0,1.0,0.0)).item()
    hits3_na = torch.mean(torch.where(_n_a_ranks <= 3.0,1.0,0.0)).item()
    hits5_na = torch.mean(torch.where(_n_a_ranks <= 5.0,1.0,0.0)).item()
    hits10_na = torch.mean(torch.where(_n_a_ranks <= 10.0,1.0,0.0)).item()

    mrr_nv = torch.mean(1.0 / _n_v_ranks).item()
    hits1_nv = torch.mean(torch.where(_n_v_ranks <= 1.0,1.0,0.0)).item()
    hits3_nv = torch.mean(torch.where(_n_v_ranks <= 3.0,1.0,0.0)).item()
    hits5_nv = torch.mean(torch.where(_n_v_ranks <= 5.0,1.0,0.0)).item()
    hits10_nv = torch.mean(torch.where(_n_v_ranks <= 10.0,1.0,0.0)).item()

    mrr_r = torch.mean(1.0 / all_r_ranks).item()
    hits1_r = torch.mean(torch.where(all_r_ranks <= 1.0,1.0,0.0)).item()
    hits3_r = torch.mean(torch.where(all_r_ranks <= 3.0,1.0,0.0)).item()
    hits5_r = torch.mean(torch.where(all_r_ranks <= 5.0,1.0,0.0)).item()
    hits10_r = torch.mean(torch.where(all_r_ranks <= 10.0,1.0,0.0)).item()

    mrr_ht = torch.mean(1.0 / all_ht_ranks).item()
    hits1_ht = torch.mean(torch.where(all_ht_ranks <= 1.0,1.0,0.0)).item()
    hits3_ht = torch.mean(torch.where(all_ht_ranks <= 3.0,1.0,0.0)).item()
    hits5_ht = torch.mean(torch.where(all_ht_ranks <= 5.0,1.0,0.0)).item()
    hits10_ht = torch.mean(torch.where(all_ht_ranks <= 10.0,1.0,0.0)).item()

    eval_result = {
        'entity': {
            'mrr': mrr_ent,
            'hits1': hits1_ent,
            'hits3': hits3_ent,
            'hits5': hits5_ent,
            'hits10': hits10_ent
        },
        'relation': {
            'mrr': mrr_rel,
            'hits1': hits1_rel,
            'hits3': hits3_rel,
            'hits5': hits5_rel,
            'hits10': hits10_rel
        },
        'ht': {
            'mrr': mrr_ht,
            'hits1': hits1_ht,
            'hits3': hits3_ht,
            'hits5': hits5_ht,
            'hits10': hits10_ht
        },
        '2-ht': {
            'mrr': mrr_2ht,
            'hits1': hits1_2ht,
            'hits3': hits3_2ht,
            'hits5': hits5_2ht,
            'hits10': hits10_2ht
        },
        'n-ht': {
            'mrr': mrr_nht,
            'hits1': hits1_nht,
            'hits3': hits3_nht,
            'hits5': hits5_nht,
            'hits10': hits10_nht
        },
        'r': {
            'mrr': mrr_r,
            'hits1': hits1_r,
            'hits3': hits3_r,
            'hits5': hits5_r,
            'hits10': hits10_r
        },
        '2-r': {
            'mrr': mrr_2r,
            'hits1': hits1_2r,
            'hits3': hits3_2r,
            'hits5': hits5_2r,
            'hits10': hits10_2r
        },
        'n-r': {
            'mrr': mrr_nr,
            'hits1': hits1_nr,
            'hits3': hits3_nr,
            'hits5': hits5_nr,
            'hits10': hits10_nr
        },
        'n-a': {
            'mrr': mrr_na,
            'hits1': hits1_na,
            'hits3': hits3_na,
            'hits5': hits5_na,
            'hits10': hits10_na
        },
        'n-v': {
            'mrr': mrr_nv,
            'hits1': hits1_nv,
            'hits3': hits3_nv,
            'hits5': hits5_nv,
            'hits10': hits10_nv
        },
    }

    return eval_result