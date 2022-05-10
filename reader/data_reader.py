
from __future__ import print_function
from __future__ import division
from abc import abstractclassmethod

import logging
import collections


import json
import copy

from torch import _add_batch_dim

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

#读取三元组
def read_facts(file):
    facts_list = list()
    max_n=0
    with open(file, 'r', encoding='utf8') as f:
        for line in f:
            fact=list()
            obj = json.loads(line)
            if obj['N']>max_n:
                max_n=obj['N']
            fact.append(obj['subject'])
            fact.append(obj['relation'])
            fact.append(obj['object'])
            if obj['N']>2:
                for key in obj.keys():
                    if key!='N' and key!='subject' and key!='relation' and key!='object':
                        for item in obj[key]:
                            fact.append(key)
                            fact.append(item)
            facts_list.append(fact)
        f.close()
    return facts_list,max_n

def read_dict(ent_file,rel_file):
    dict_id=dict()
    dict_id['PAD']=0
    dict_id['MASK']=1
    dict_num=2
    rel_num=0
    with open(rel_file, 'r', encoding='utf8') as f:
        for line in f:
            line=line.strip('\n')
            dict_id[line]=dict_num
            dict_num+=1
            rel_num+=1
        f.close()    
    with open(ent_file, 'r', encoding='utf8') as f:
        for line in f:
            line=line.strip('\n')
            dict_id[line]=dict_num
            dict_num+=1
        f.close()
    return dict_id,dict_num,rel_num

def facts_to_id(facts,max_n,node_dict):
    id_facts=list()
    id_masks=list()
    mask_labels=list()
    mask_pos=list()
    mask_types=list()
    for fact in facts:
        id_fact=list()
        id_mask=list()
        for i,item in enumerate(fact):
            id_fact.append(node_dict[item])
            id_mask.append(1.0)
        
        for j,mask_label in enumerate(id_fact):  
            x=copy.copy(id_fact)        
            x[j]=1
            y=copy.copy(id_mask)
            if j%2==0:
                mask_type=1
            else:
                mask_type=-1
            while len(x)<(2*max_n-1):
                x.append(0) 
                y.append(0.0)                         
            id_facts.append(x)
            id_masks.append(y)
            mask_pos.append(j)
            mask_labels.append(mask_label)    
            mask_types.append(mask_type) 
    return [id_facts,id_masks,mask_pos,mask_labels,mask_types]

def get_truth(all_facts,max_n,node_dict):
    max_aux=max_n-2
    max_seq_length = 2 * max_aux + 3
    gt_dict = collections.defaultdict(lambda: collections.defaultdict(list))
    all_fact_ids=list()
    for fact in all_facts:
        id_fact=list()
        for i,item in enumerate(fact):
            id_fact.append(node_dict[item]) 
        all_fact_id=copy.copy(id_fact)
        all_fact_ids.append(all_fact_id)
        while len(id_fact)<(2*max_n-1):
            id_fact.append(0)
        for pos in range(max_seq_length):
            if id_fact[pos]==0:
                continue
            key = " ".join([
                str(id_fact[x]) for x in range(max_seq_length) if x != pos
            ])
            gt_dict[pos][key].append(id_fact[pos])          

    return gt_dict,all_fact_ids

def get_input(all_file, train_file, valid_file, test_file, entities_file, relation_file):
    all_facts,max_n= read_facts(all_file)
    train_facts,_= read_facts(train_file)
    valid_facts,_= read_facts(valid_file)
    test_facts,_= read_facts(test_file)
    node_dict, node_num, rel_num=read_dict(entities_file,relation_file)
    all_facts,all_fact_ids= get_truth(all_facts,max_n,node_dict)
    train_facts= facts_to_id(train_facts,max_n,node_dict)
    valid_facts= facts_to_id(valid_facts,max_n,node_dict)
    test_facts= facts_to_id(test_facts,max_n,node_dict)
    input_info=dict()
    input_info['all_facts']=all_facts
    input_info['all_fact_ids']=all_fact_ids
    input_info['train_facts']=train_facts
    input_info['valid_facts']=valid_facts
    input_info['test_facts']=test_facts
    input_info['node_dict']=node_dict
    input_info['node_num']=node_num
    input_info['rel_num']=rel_num
    input_info['max_n']=max_n
    return input_info

def pos_get_input(all_file, train_file, valid_file, test_file, entities_file, relation_file):
    all_facts,max_n= read_facts(all_file)
    train_facts,_= read_facts(train_file)
    valid_facts,_= read_facts(valid_file)
    test_facts,_= read_facts(test_file)
    node_dict, node_num, rel_num=read_dict(entities_file,relation_file)
    all_facts,all_fact_ids= get_truth(all_facts,max_n,node_dict)
    train_facts= pos_facts_to_id(train_facts,max_n,node_dict)
    valid_facts= pos_facts_to_id(valid_facts,max_n,node_dict)
    test_facts= pos_facts_to_id(test_facts,max_n,node_dict)
    input_info=dict()
    input_info['all_facts']=all_facts
    input_info['all_fact_ids']=all_fact_ids
    input_info['train_facts']=train_facts
    input_info['valid_facts']=valid_facts
    input_info['test_facts']=test_facts
    input_info['node_dict']=node_dict
    input_info['node_num']=node_num
    input_info['rel_num']=rel_num
    input_info['max_n']=max_n
    return input_info

def pos_facts_to_id(facts,max_n,node_dict):
    id_facts=list()
    id_masks=list()
    mask_labels=list()
    mask_pos=list()
    mask_types=list()
    for fact in facts:
        id_fact=list()
        id_mask=list()
        for i,item in enumerate(fact):
            id_fact.append(node_dict[item])
            id_mask.append(1.0)
        
        for j,mask_label in enumerate(id_fact): 
            if j!=2:
                continue 
            x=copy.copy(id_fact)        
            x[j]=1
            y=copy.copy(id_mask)
            if j%2==0:
                mask_type=1
            else:
                mask_type=-1
            while len(x)<(2*max_n-1):
                x.append(0) 
                y.append(0.0)                         
            id_facts.append(x)
            id_masks.append(y)
            mask_pos.append(j)
            mask_labels.append(mask_label)    
            mask_types.append(mask_type) 
    return [id_facts,id_masks,mask_pos,mask_labels,mask_types]

def truth_to_id(all_facts, ins_ent_ids, onto_ent_ids):
    typing=dict()
    for fact in all_facts:
        if fact[0] not in typing.keys():
            typing[fact[0]]=list()
        if onto_ent_ids[fact[2]] not in typing[fact[0]]:
            typing[fact[0]].append(onto_ent_ids[fact[2]])
    typing_id=dict()
    for key in typing.keys():
        typing_id[str(ins_ent_ids[key])]=typing[key]
    return typing_id

def cross_to_id(facts, ins_ent_ids, onto_ent_ids):

    h=list()
    t=list()
    for item in facts:
        h.append(ins_ent_ids[item[0]])
        t.append(onto_ent_ids[item[2]])
    return [h,t]
    
def get_cross_input(all_file, train_file, valid_file, test_file, ins_ent_ids,  onto_ent_ids):

    all_facts,_= read_facts(all_file)   
    train_facts,_= read_facts(train_file)
    valid_facts,_= read_facts(valid_file)
    test_facts,_= read_facts(test_file)

    all_facts = truth_to_id(all_facts, ins_ent_ids,  onto_ent_ids)
    train_facts = cross_to_id(train_facts, ins_ent_ids,  onto_ent_ids)
    valid_facts = cross_to_id(valid_facts, ins_ent_ids,  onto_ent_ids)
    test_facts = cross_to_id(test_facts, ins_ent_ids,  onto_ent_ids)

    cross_info=dict()
    cross_info['all_cross']=all_facts
    cross_info['train_cross']=train_facts
    cross_info['valid_cross']=valid_facts
    cross_info['test_cross']=test_facts

    return cross_info

def pos_read_input(folder):
    ins_info = pos_get_input(folder + "ins/ins_all.json",folder +"ins/ins_train.json", 
                            folder + "ins/ins_valid.json",folder + "ins/ins_test.json",
                            folder + "ins/ins_entities.txt",folder + "ins/ins_relations.txt")
    logger.info("Number of ins_all fact_ids: "+str(len(ins_info['all_fact_ids'])))
    logger.info("Number of ins_train facts: "+str(len(ins_info['train_facts'][0])))
    logger.info("Number of ins_valid facts: "+str(len(ins_info['valid_facts'][0])))
    logger.info("Number of ins_test facts: "+str(len(ins_info['test_facts'][0])))
    logger.info("Number of ins nodes: "+str(ins_info['node_num']))
    logger.info("Number of ins relations: "+str(ins_info['rel_num']))
    logger.info("Number of ins max_n: "+str(ins_info['max_n']))
    logger.info("Number of ins max_seq_length: "+str(2*ins_info['max_n']-1))

    onto_info = get_input(folder + "onto/onto_all.json",folder +"onto/onto_train.json", 
                            folder + "onto/onto_valid.json",folder + "onto/onto_test.json",
                            folder + "onto/onto_entities.txt",folder + "onto/onto_relations.txt")
    logger.info("Number of onto_all fact_ids: "+str(len(onto_info['all_fact_ids'])))
    logger.info("Number of onto_train facts: "+str(len(onto_info['train_facts'][0])))
    logger.info("Number of onto_valid facts: "+str(len(onto_info['valid_facts'][0])))
    logger.info("Number of onto_test facts: "+str(len(onto_info['test_facts'][0])))
    logger.info("Number of onto nodes: "+str(onto_info['node_num']))
    logger.info("Number of onto relations: "+str(onto_info['rel_num']))
    logger.info("Number of onto max_n: "+str(onto_info['max_n']))
    logger.info("Number of onto max_seq_length: "+str(2*onto_info['max_n']-1))


    cross_info = get_cross_input(folder + "cross/cross_all.json", folder + "cross/cross_train.json",
                                    folder + "cross/cross_valid.json", folder + "cross/cross_test.json",
                                    ins_info['node_dict'], onto_info['node_dict'])
    logger.info("Number of cross_train facts: "+str(len(cross_info['train_cross'][0])))
    logger.info("Number of cross_valid facts: "+str(len(cross_info['valid_cross'][0])))
    logger.info("Number of cross_test facts: "+str(len(cross_info['test_cross'][0])))
    
    return ins_info, onto_info, cross_info

def read_input(folder):
    ins_info = get_input(folder + "ins/ins_all.json",folder +"ins/ins_train.json", 
                            folder + "ins/ins_valid.json",folder + "ins/ins_test.json",
                            folder + "ins/ins_entities.txt",folder + "ins/ins_relations.txt")
    logger.info("Number of ins_all fact_ids: "+str(len(ins_info['all_fact_ids'])))
    logger.info("Number of ins_train facts: "+str(len(ins_info['train_facts'][0])))
    logger.info("Number of ins_valid facts: "+str(len(ins_info['valid_facts'][0])))
    logger.info("Number of ins_test facts: "+str(len(ins_info['test_facts'][0])))
    logger.info("Number of ins nodes: "+str(ins_info['node_num']))
    logger.info("Number of ins relations: "+str(ins_info['rel_num']))
    logger.info("Number of ins max_n: "+str(ins_info['max_n']))
    logger.info("Number of ins max_seq_length: "+str(2*ins_info['max_n']-1))

    onto_info = get_input(folder + "onto/onto_all.json",folder +"onto/onto_train.json", 
                            folder + "onto/onto_valid.json",folder + "onto/onto_test.json",
                            folder + "onto/onto_entities.txt",folder + "onto/onto_relations.txt")
    logger.info("Number of onto_all fact_ids: "+str(len(onto_info['all_fact_ids'])))
    logger.info("Number of onto_train facts: "+str(len(onto_info['train_facts'][0])))
    logger.info("Number of onto_valid facts: "+str(len(onto_info['valid_facts'][0])))
    logger.info("Number of onto_test facts: "+str(len(onto_info['test_facts'][0])))
    logger.info("Number of onto nodes: "+str(onto_info['node_num']))
    logger.info("Number of onto relations: "+str(onto_info['rel_num']))
    logger.info("Number of onto max_n: "+str(onto_info['max_n']))
    logger.info("Number of onto max_seq_length: "+str(2*onto_info['max_n']-1))


    cross_info = get_cross_input(folder + "cross/cross_all.json", folder + "cross/cross_train.json",
                                    folder + "cross/cross_valid.json", folder + "cross/cross_test.json",
                                    ins_info['node_dict'], onto_info['node_dict'])
    logger.info("Number of cross_train facts: "+str(len(cross_info['train_cross'][0])))
    logger.info("Number of cross_valid facts: "+str(len(cross_info['valid_cross'][0])))
    logger.info("Number of cross_test facts: "+str(len(cross_info['test_cross'][0])))
    
    return ins_info, onto_info, cross_info








