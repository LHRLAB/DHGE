import argparse
import ast
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
from utils.args import ArgumentGroup,print_arguments
import logging
from reader.data_reader import read_input
from reader.data_loader import prepare_EC_info,prepare_adj_info,get_edge_labels
from model.DHGE import DHGE
import time
import math
import random
import torch
import torch.utils.data.dataset as Dataset
import torch.utils.data.dataloader as DataLoader
import copy
from itertools import cycle
from utils.evaluation import batch_evaluation,compute_metrics
torch.set_printoptions(precision=8)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.info(logger.getEffectiveLevel())

parser = argparse.ArgumentParser(description='HyperKE4TI')
DHGE_g=ArgumentGroup(parser, "model", "model and checkpoint configuration.")
DHGE_g.add_arg('input', type=str, default='dataset/JW44K-6K/JW44K-6K-',help="")  # db
DHGE_g.add_arg('output', type=str, default='/home/luohaoran/HyperKA-pytorch/output/results/',help="")

DHGE_g.add_arg('dim', type=int, default=256,help="")
DHGE_g.add_arg('onto_dim', type=int, default=256,help="")
DHGE_g.add_arg('ins_layer_num', type=int, default=3,help="")
DHGE_g.add_arg('onto_layer_num', type=int, default=3,help="")
DHGE_g.add_arg('neg_typing_margin', type=float, default=0.1,help="")
DHGE_g.add_arg('neg_triple_margin', type=float, default=0.2,help="")

DHGE_g.add_arg('nums_neg', type=int, default=30,help="")
DHGE_g.add_arg('mapping_neg_nums', type=int, default=30,help="")

DHGE_g.add_arg('learning_rate', type=float, default=5e-4,help="")
DHGE_g.add_arg('batch_size', type=int, default=1024,help="")
DHGE_g.add_arg('epochs', type=int, default=160,help="")

DHGE_g.add_arg('combine', type=ast.literal_eval, default=True,help="")
DHGE_g.add_arg('ent_top_k', type=list, default=[1, 3, 5, 10],help="")
DHGE_g.add_arg("use_cuda", bool,   True,  "If set, use GPU for training.")

DHGE_g.add_arg('ins_intermediate_size', type=int, default=512,help="")
DHGE_g.add_arg('onto_intermediate_size', type=int, default=512,help="")
DHGE_g.add_arg('num_hidden_layers', type=int, default=12,help="")
DHGE_g.add_arg('num_attention_heads', type=int, default=4,help="")
DHGE_g.add_arg('hidden_dropout_prob', type=float, default=0.1,help="")
DHGE_g.add_arg('attention_dropout_prob', type=float, default=0.1,help="")
DHGE_g.add_arg('num_edges', type=int, default=5,help="")

args = parser.parse_args(args=[])

class EDataset(Dataset.Dataset):
    def __init__(self, triples1):
        self.triples1=triples1 
    def __len__(self):
        return len(self.triples1[0])
    def __getitem__(self,index):        
        return  self.triples1[0][index],self.triples1[1][index],self.triples1[2][index],self.triples1[3][index],self.triples1[4][index]

class CDataset(Dataset.Dataset):
    def __init__(self, triples2):
        self.triples2=triples2
    def __len__(self):
        return len(self.triples2[0])
    def __getitem__(self,index):        
        return self.triples2[0][index],self.triples2[1][index],self.triples2[2][index],self.triples2[3][index],self.triples2[4][index]

class SDataset(Dataset.Dataset):
    def __init__(self, cross_info,ins_info,onto_info,nums_neg,device):
        self.device=device
        self.nums_neg=nums_neg
        self.seed_sup_ent1 = cross_info['train_cross'][0]
        self.seed_sup_ent2 = cross_info['train_cross'][1]  
        self.ins_neg_sample=  list(range(2+ins_info['rel_num'],ins_info['node_num']))
        self.onto_neg_sample=  list(range(2+onto_info['rel_num'],onto_info['node_num']))
        self.seed_links = list()
        self.seed_link_set=set()
        for i in range(len(self.seed_sup_ent1)):
            seed_link=(self.seed_sup_ent1[i], self.seed_sup_ent2[i])
            self.seed_links.append(seed_link)
            self.seed_link_set.add(seed_link)
        self.links=list()
        self.typing_negs=list()
        for i in range(len(self.seed_links)):             
            typing_neg_links = list()         
            typing_sample=copy.copy(self.onto_neg_sample)
            typing_sample.remove(self.seed_links[i][1])
            typing_neg_ent2 = random.sample(typing_sample, self.nums_neg)
            typing_neg_links.extend([(self.seed_links[i][0], typing_neg_ent2[k]) for k in range(self.nums_neg)])
            typing_neg_links = torch.tensor(typing_neg_links).to(device)
            seed_link=torch.tensor(list(self.seed_links[i])).to(device)
            self.links.append(seed_link)
            self.typing_negs.append(typing_neg_links)
    def __len__(self):
        return len(self.links)
    def __getitem__(self,index):
        return self.links[index],self.typing_negs[index]

def main(args):
    config = vars(args)
    if args.use_cuda:
        device = torch.device("cuda")
        config["device"]="cuda"
    else:
        device = torch.device("cpu")
        config["device"]="cpu"
    print("data folder:", args.input)

    ins_info, onto_info, cross_info = read_input(args.input)

    instance_info, ontology_info = prepare_EC_info (ins_info, onto_info, device)
    ins_edge_labels = get_edge_labels(ins_info['max_n']).to(device)
    onto_edge_labels = get_edge_labels(onto_info['max_n']).to(device)
    ins_adj_info, onto_adj_info= prepare_adj_info(ins_info, onto_info, device)

    model = DHGE(instance_info, ontology_info, ins_adj_info, onto_adj_info, config).to(device)

    # E_train_dataloader
    ins_train_facts=list()
    for ins_train_fact in ins_info['train_facts']:
        ins_train_fact=torch.tensor(ins_train_fact).to(device)
        ins_train_facts.append(ins_train_fact)
    train_data_E_reader=EDataset(ins_train_facts)
    train_E_pyreader=DataLoader.DataLoader(train_data_E_reader,batch_size=args.batch_size,shuffle=True,drop_last=False)
    # C_train_dataloader
    onto_train_facts=list()
    for onto_train_fact in onto_info['train_facts']:
        onto_train_fact=torch.tensor(onto_train_fact).to(device)
        onto_train_facts.append(onto_train_fact)
    train_data_C_reader=CDataset(onto_train_facts)
    train_C_pyreader=DataLoader.DataLoader(train_data_C_reader,batch_size=args.batch_size,shuffle=True,drop_last=False)  
    # S_train_dataloader
    train_data_S_reader=SDataset(cross_info,ins_info,onto_info,config['nums_neg'],device)
    train_S_batch_size=max(math.ceil(args.batch_size/ len(ins_info['train_facts'][0]) *train_data_S_reader.__len__())-1,2)
    train_S_pyreader=DataLoader.DataLoader(train_data_S_reader,batch_size=train_S_batch_size,shuffle=True,drop_last=False) 
    # train_information
    logging.info("train_ins_batch_size: "+str(args.batch_size))
    logging.info("train_onto_batch_size: "+str(args.batch_size))
    logging.info("train_cross_batch_size: "+str(train_S_batch_size))
    steps = math.ceil(len(ins_info['train_facts']) / args.batch_size)
    logging.info("train_steps_per_epoch: "+str(steps))

    # E_valid_dataloader
    ins_valid_facts=list()
    for ins_valid_fact in ins_info['valid_facts']:
        ins_valid_fact=torch.tensor(ins_valid_fact).to(device)
        ins_valid_facts.append(ins_valid_fact)
    valid_data_E_reader=EDataset(ins_valid_facts)
    valid_E_pyreader=DataLoader.DataLoader(  
        valid_data_E_reader,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)
    # C_valid_dataloader
    onto_valid_facts=list()
    for onto_valid_fact in onto_info['valid_facts']:
        onto_valid_fact=torch.tensor(onto_valid_fact).to(device)
        onto_valid_facts.append(onto_valid_fact)
    valid_data_C_reader=CDataset(onto_valid_facts)
    valid_C_pyreader=DataLoader.DataLoader(  
        valid_data_C_reader,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)
    # S_valid_dataloader
    ref_ent1 = cross_info['valid_cross'][0]
    all_ref = [cross_info['all_cross'][str(k)] for k in ref_ent1]
    ref_ent1=torch.tensor(ref_ent1).to('cuda') 

    # E_valid_dataloader
    ins_test_facts=list()
    for ins_test_fact in ins_info['test_facts']:
        ins_test_fact=torch.tensor(ins_test_fact).to(device)
        ins_test_facts.append(ins_test_fact)
    test_data_E_reader=EDataset(ins_test_facts)
    test_E_pyreader=DataLoader.DataLoader(  
        test_data_E_reader,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)
    # C_valid_dataloader
    onto_test_facts=list()
    for onto_test_fact in onto_info['test_facts']:
        onto_test_fact=torch.tensor(onto_test_fact).to(device)
        onto_test_facts.append(onto_test_fact)
    test_data_C_reader=CDataset(onto_test_facts)
    test_C_pyreader=DataLoader.DataLoader(  
        test_data_C_reader,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)
    # S_valid_dataloader
    ref_ent1_test = cross_info['test_cross'][0]
    all_ref_test = [cross_info['all_cross'][str(k)] for k in ref_ent1_test]
    ref_ent1_test=torch.tensor(ref_ent1_test).to('cuda')
  
    # ECS_optimizers
    ins_optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    onto_optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'])
    cross_optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate']/5.0)

    # Start Training
    iterations = 1
    for iteration in range(1, args.epochs // iterations + 1):
        logger.info("iteration "+str(iteration))
        t1 = time.time()
        model.train()
        for i in range(iterations):          
            ins_epoch_loss = 0
            onto_epoch_loss =0
            mapping_loss = 0
            start = time.time()
            for j,data in enumerate(zip(train_E_pyreader,cycle(train_C_pyreader),cycle(train_S_pyreader))):
                ins_pos,onto_pos,pos_list=data
                # E_training
                ins_optimizer.zero_grad()
                ins_loss,_= model.forward_E(ins_pos,ins_edge_labels) 
                ins_loss.backward()
                ins_optimizer.step()
                ins_epoch_loss += ins_loss
                # C_training
                onto_optimizer.zero_grad()
                onto_loss,_= model.forward_C(onto_pos,onto_edge_labels) 
                onto_loss.backward()
                onto_optimizer.step()
                onto_epoch_loss += onto_loss 
                # S_training
                cross_optimizer.zero_grad()
                loss2 = model.forward_S(pos_list)
                loss2.backward()
                cross_optimizer.step()
                mapping_loss += loss2
                # print_ECS_loss_per_step
                if j%100==0:
                    logger.info(str(j)+' , ins_loss: '+str(ins_loss.item())+' , onto_loss: '+str(onto_loss.item())+' , loss2: '+str(loss2.item()))
            # print_ECS_loss_per_epoch
            ins_epoch_loss /= steps
            onto_epoch_loss /= steps
            mapping_loss /= steps
            end = time.time()
            t2=round(end - start, 2)
            logger.info("ins_epoch_loss = {:.3f}, onto_epoch_loss = {:.3f}, typing_loss = {:.3f}, time = {:.3f} s".format(ins_epoch_loss, onto_epoch_loss, mapping_loss, t2))
        # Start validation and testing
        model.eval()
        with torch.no_grad():
            # EC_validation
            h1EC = predict(
                model=model,
                ins_test_pyreader=valid_E_pyreader,
                onto_test_pyreader=valid_C_pyreader,
                ins_all_facts=ins_info['all_facts'],
                onto_all_facts=onto_info['all_facts'],
                ins_edge_labels=ins_edge_labels,
                onto_edge_labels=onto_edge_labels,
                device=device)            
            # S_validation
            h1 = model.test(ref_ent1,all_ref)
            # EC_testing
            h2EC = predict(
                model=model,
                ins_test_pyreader=test_E_pyreader,
                onto_test_pyreader=test_C_pyreader,
                ins_all_facts=ins_info['all_facts'],
                onto_all_facts=onto_info['all_facts'],
                ins_edge_labels=ins_edge_labels,
                onto_edge_labels=onto_edge_labels,
                device=device)  
            # S_testing
            h2 = model.test(ref_ent1_test,all_ref_test)

    logger.info("stop")

def predict(model, ins_test_pyreader,  onto_test_pyreader, 
                ins_all_facts,onto_all_facts,
                ins_edge_labels,onto_edge_labels,device):
    start=time.time()

    step = 0
    ins_ret_ranks=dict()
    ins_ret_ranks['entity']=torch.empty(0).to(device)
    ins_ret_ranks['relation']=torch.empty(0).to(device)
    ins_ret_ranks['2-r']=torch.empty(0).to(device)
    ins_ret_ranks['2-ht']=torch.empty(0).to(device)
    ins_ret_ranks['n-r']=torch.empty(0).to(device)
    ins_ret_ranks['n-ht']=torch.empty(0).to(device)
    ins_ret_ranks['n-a']=torch.empty(0).to(device)
    ins_ret_ranks['n-v']=torch.empty(0).to(device)

    onto_ret_ranks=dict()
    onto_ret_ranks['entity']=torch.empty(0).to(device)
    onto_ret_ranks['relation']=torch.empty(0).to(device)
    onto_ret_ranks['2-r']=torch.empty(0).to(device)
    onto_ret_ranks['2-ht']=torch.empty(0).to(device)
    onto_ret_ranks['n-r']=torch.empty(0).to(device)
    onto_ret_ranks['n-ht']=torch.empty(0).to(device)
    onto_ret_ranks['n-a']=torch.empty(0).to(device)
    onto_ret_ranks['n-v']=torch.empty(0).to(device)

    #while steps < max_train_steps:
    for i, data in enumerate(zip(ins_test_pyreader,cycle(onto_test_pyreader))):
        ins_pos,onto_pos=data
        _,ins_np_fc_out= model.forward_E(ins_pos,ins_edge_labels) 
        _,onto_np_fc_out= model.forward_C(onto_pos,onto_edge_labels) 

        ins_ret_ranks=batch_evaluation(ins_np_fc_out, ins_pos, ins_all_facts,ins_ret_ranks,device)
        onto_ret_ranks=batch_evaluation(onto_np_fc_out, onto_pos, onto_all_facts,onto_ret_ranks,device)
        #print("Processing prediction steps: %d | ins_examples: %d | onto_examples: %d " % (step, ins_global_idx,onto_global_idx))
        step += 1     

    ins_eval_performance = compute_metrics(ins_ret_ranks)

    ins_all_entity = "ENTITY\t\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        ins_eval_performance['entity']['mrr'],
        ins_eval_performance['entity']['hits1'],
        ins_eval_performance['entity']['hits3'],
        ins_eval_performance['entity']['hits5'],
        ins_eval_performance['entity']['hits10'])

    ins_all_relation = "RELATION\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        ins_eval_performance['relation']['mrr'],
        ins_eval_performance['relation']['hits1'],
        ins_eval_performance['relation']['hits3'],
        ins_eval_performance['relation']['hits5'],
        ins_eval_performance['relation']['hits10'])

    ins_all_ht = "HEAD/TAIL\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        ins_eval_performance['ht']['mrr'],
        ins_eval_performance['ht']['hits1'],
        ins_eval_performance['ht']['hits3'],
        ins_eval_performance['ht']['hits5'],
        ins_eval_performance['ht']['hits10'])

    ins_all_r = "PRIMARY_R\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        ins_eval_performance['r']['mrr'],
        ins_eval_performance['r']['hits1'],
        ins_eval_performance['r']['hits3'],
        ins_eval_performance['r']['hits5'],
        ins_eval_performance['r']['hits10'])

    logger.info("\n-------- E Evaluation Performance --------\n%s\n%s\n%s\n%s\n%s" % (
        "\t".join(["TASK\t", "MRR", "Hits@1", "Hits@3", "Hits@5", "Hits@10"]),
        ins_all_ht, ins_all_r, ins_all_entity, ins_all_relation))

    onto_eval_performance = compute_metrics(onto_ret_ranks)

    onto_all_entity = "ENTITY\t\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        onto_eval_performance['entity']['mrr'],
        onto_eval_performance['entity']['hits1'],
        onto_eval_performance['entity']['hits3'],
        onto_eval_performance['entity']['hits5'],
        onto_eval_performance['entity']['hits10'])

    onto_all_relation = "RELATION\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        onto_eval_performance['relation']['mrr'],
        onto_eval_performance['relation']['hits1'],
        onto_eval_performance['relation']['hits3'],
        onto_eval_performance['relation']['hits5'],
        onto_eval_performance['relation']['hits10'])

    onto_all_ht = "HEAD/TAIL\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        onto_eval_performance['ht']['mrr'],
        onto_eval_performance['ht']['hits1'],
        onto_eval_performance['ht']['hits3'],
        onto_eval_performance['ht']['hits5'],
        onto_eval_performance['ht']['hits10'])

    onto_all_r = "PRIMARY_R\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
        onto_eval_performance['r']['mrr'],
        onto_eval_performance['r']['hits1'],
        onto_eval_performance['r']['hits3'],
        onto_eval_performance['r']['hits5'],
        onto_eval_performance['r']['hits10'])

    logger.info("\n-------- C Evaluation Performance --------\n%s\n%s\n%s\n%s\n%s" % (
        "\t".join(["TASK\t", "MRR", "Hits@1", "Hits@3", "Hits@5", "Hits@10"]),
        onto_all_ht, onto_all_r, onto_all_entity, onto_all_relation))
    
    end=time.time()
    logger.info("INS and ONTO time: "+str(round(end - start, 3))+'s')

    return (ins_eval_performance['entity']['hits1']+onto_eval_performance['entity']['hits1'])/2.0

if __name__ == '__main__':
    print_arguments(args)
    main(args)