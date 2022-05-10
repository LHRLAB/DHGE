from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
from model.HGNN_encoder import HGNNLayer
import time
from utils.evaluation import eval_type_hyperbolic
import torch
import torch.nn
from model.gran_model import GRANModel
from model.graph_encoder import truncated_normal
torch.set_printoptions(precision=16)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.info(logger.getEffectiveLevel())

class DHGE(torch.nn.Module):
    def __init__(self, ins_info, onto_info, ins_adj_info, onto_adj_info, config):
        super(DHGE,self).__init__()
        #CONFIG SETTING
        self.config = config
        self.ins_node_num = ins_info["node_num"]       
        self.onto_node_num = onto_info["node_num"]
        #INIT EMBEDDING    
        self.ins_node_embeddings = torch.nn.Embedding(self.ins_node_num, self.config['dim'])
        self.ins_node_embeddings.weight.data=truncated_normal(self.ins_node_embeddings.weight.data,std=0.02)
        self.onto_node_embeddings = torch.nn.Embedding(self.onto_node_num, self.config["onto_dim"])
        self.onto_node_embeddings.weight.data=truncated_normal(self.onto_node_embeddings.weight.data,std=0.02)

        ##GRAN_LAYER
        self.ins_config=dict()
        self.ins_config['num_hidden_layers']=self.config['num_hidden_layers']
        self.ins_config['num_attention_heads']=self.config['num_attention_heads']
        self.ins_config['hidden_size']=self.config['dim']
        self.ins_config['intermediate_size']=self.config['ins_intermediate_size']
        self.ins_config['hidden_dropout_prob']=self.config['hidden_dropout_prob']
        self.ins_config['attention_dropout_prob']=self.config['attention_dropout_prob']
        self.ins_config['vocab_size']=self.ins_node_num
        self.ins_config['num_relations']=ins_info["rel_num"]
        self.ins_config['num_edges']=self.config['num_edges']
        self.ins_config['max_arity']=ins_info['max_n']
        self.ins_config['device']=self.config['device']
        self.ins_granlayer=GRANModel(self.ins_config,self.ins_node_embeddings).to(self.config['device'])

        self.onto_config=dict()
        self.onto_config['num_hidden_layers']=self.config['num_hidden_layers']
        self.onto_config['num_attention_heads']=self.config['num_attention_heads']
        self.onto_config['hidden_size']=self.config['onto_dim']
        self.onto_config['intermediate_size']=self.config['onto_intermediate_size']
        self.onto_config['hidden_dropout_prob']=self.config['hidden_dropout_prob']
        self.onto_config['attention_dropout_prob']=self.config['attention_dropout_prob']
        self.onto_config['vocab_size']=self.onto_node_num
        self.onto_config['num_relations']=onto_info["rel_num"]
        self.onto_config['num_edges']=self.config['num_edges']
        self.onto_config['max_arity']=onto_info['max_n']
        self.onto_config['device']=self.config['device']
        self.onto_granlayer=GRANModel(self.onto_config,self.onto_node_embeddings).to(self.config['device'])

        ##HGNN_LAYER
        self.ins_adj_mat=torch.sparse_coo_tensor(ins_adj_info['indices'],ins_adj_info['values'],torch.Size(ins_adj_info['size']), dtype=torch.float32)
        self.onto_adj_mat=torch.sparse_coo_tensor(onto_adj_info['indices'],onto_adj_info['values'],torch.Size(onto_adj_info['size']), dtype=torch.float32)
        self.activation = torch.nn.Tanh()
        self.ins_output = list()
        self.onto_output = list()
        self.ins_layer_num = config["ins_layer_num"]
        self.onto_layer_num = config["onto_layer_num"]
        for i in range(self.ins_layer_num):
            activation = self.activation
            if i == self.ins_layer_num - 1:
                activation = None
            setattr(self,"ins_gcn_layer{}".format(i),HGNNLayer(self.ins_adj_mat, self.config['dim'], self.config['dim'], act=activation))

        for i in range(self.onto_layer_num):
            activation = self.activation
            if i == self.onto_layer_num - 1:
                activation = None
            setattr(self,"onto_gcn_layer{}".format(i),HGNNLayer(self.onto_adj_mat, self.config['onto_dim'], self.config['onto_dim'], act=activation))
        
        ##JointEmbedding_LAYER
        self.typing_mapping_matrix = torch.nn.init.orthogonal_(torch.nn.parameter.Parameter(torch.zeros([self.config['dim'], self.config['onto_dim']])),gain=0.02)


    def forward_E(self,ins_pos,ins_edge_labels):
        ins_input_ids, ins_input_mask, ins_mask_pos, ins_mask_label, ins_mask_type = ins_pos
        self.ins_triple_loss, self.ins_fc_out=self.ins_granlayer(ins_input_ids, ins_input_mask, ins_edge_labels ,ins_mask_pos, ins_mask_label, ins_mask_type)

        return self.ins_triple_loss , self.ins_fc_out

    def forward_C(self,onto_pos,onto_edge_labels):
        onto_input_ids, onto_input_mask, onto_mask_pos, onto_mask_label, onto_mask_type = onto_pos
        self.onto_triple_loss, self.onto_fc_out=self.onto_granlayer(onto_input_ids, onto_input_mask, onto_edge_labels ,onto_mask_pos, onto_mask_label, onto_mask_type)

        return self.onto_triple_loss,self.onto_fc_out

    def forward_S(self,pos_list):
        links,typing_negs=pos_list
        cross_pos_left,cross_pos_right=links.split(1,dim=-1)
        cross_pos_left=cross_pos_left.reshape([-1])
        cross_pos_right=cross_pos_right.reshape([-1])
        typing_neg_left,typing_neg_right=typing_negs.reshape([-1,typing_negs.size(-1)]).split(1,dim=-1)
        typing_neg_left=typing_neg_left.reshape([-1])
        typing_neg_right=typing_neg_right.reshape([-1])

        self._graph_convolution()
        ins_embeddings = self.ins_output[-1]
        onto_embeddings = self.onto_output[-1]
        if self.config['combine']:
            ins_embeddings = torch.add(ins_embeddings, self.ins_output[0])
            onto_embeddings = torch.add(onto_embeddings, self.onto_output[0])

        cross_pos_left=cross_pos_left[:,None].expand(-1,self.config['dim'])
        cross_left = torch.gather(input=ins_embeddings, dim=0, index=cross_pos_left)
        cross_pos_right=cross_pos_right[:,None].expand(-1,self.config['onto_dim'])
        cross_right = torch.gather(input=onto_embeddings, dim=0, index=cross_pos_right)
        
        mapped_sup_embeds1 = torch.matmul(cross_left, self.typing_mapping_matrix)
        typing_sup_distance = torch.norm(mapped_sup_embeds1-cross_right,p=2,dim=-1,keepdim=False)       

        # *****************add neg sample***********************************************
        typing_neg_left=typing_neg_left[:,None].expand(-1,self.config['dim'])
        typing_neg_embeds1 = torch.gather(input=ins_embeddings, dim=0, index=typing_neg_left)
        typing_neg_right=typing_neg_right[:,None].expand(-1,self.config['onto_dim'])
        typing_neg_embeds2 = torch.gather(input=onto_embeddings, dim=0, index=typing_neg_right)
        mapped_neg_embeds1 = torch.matmul(typing_neg_embeds1, self.typing_mapping_matrix)
        typing_neg_distance = torch.norm(mapped_neg_embeds1-typing_neg_embeds2,p=2,dim=-1,keepdim=False)

        # *****************add neg sample***********************************************
        typing_pos_loss = torch.sum(torch.nn.ReLU()(typing_sup_distance))
        typing_neg_loss = torch.sum(torch.nn.ReLU()(self.config['neg_typing_margin'] - typing_neg_distance))
        self.mapping_loss = typing_pos_loss + typing_neg_loss
        return self.mapping_loss

    def _graph_convolution(self):
        self.ins_output = list()  # reset
        self.onto_output = list()
        # ************************* instance gnn ***************************
        # In this case, we assume that the initialized embeddings are in the hyperbolic space.
        ins_output_embeddings=self.ins_node_embeddings.weight
        self.ins_output.append(ins_output_embeddings)
        for i in range(self.ins_layer_num):
            ins_output_embeddings = getattr(self,"ins_gcn_layer{}".format(i))(ins_output_embeddings)
            ins_output_embeddings = torch.add(ins_output_embeddings, self.ins_output[-1])
            self.ins_output.append(ins_output_embeddings)
        # ************************* ontology gnn ***************************
        # In this case, we assume that the initialized embeddings are in the hyperbolic space.
        onto_output_embeddings=self.onto_node_embeddings.weight
        self.onto_output.append(onto_output_embeddings)
        for i in range(self.onto_layer_num):
            onto_output_embeddings = getattr(self,"onto_gcn_layer{}".format(i))(onto_output_embeddings)
            onto_output_embeddings = torch.add(onto_output_embeddings, self.onto_output[-1])
            self.onto_output.append(onto_output_embeddings)

    def _generate_triple_loss(self, phs, prs, pts, nhs, nrs, nts):
        pos_score = torch.norm(torch.add(phs, prs)-pts,p=2,dim=-1,keepdim=False)
        neg_score = torch.norm(torch.add(nhs, nrs)-nts,p=2,dim=-1,keepdim=False)
        pos_loss = torch.sum(torch.nn.ReLU()(pos_score))
        neg_loss = torch.sum(torch.nn.ReLU()(self.config['neg_triple_margin'] - neg_score))
        return pos_loss + neg_loss

    def test(self,ref_ent1,all_ref):
        start = time.time()
        ins_embeddings = self.ins_output[-1]
        onto_embeddings = self.onto_output[-1]
        if self.config['combine']:
            ins_embeddings = torch.add(ins_embeddings, self.ins_output[0])
            onto_embeddings = torch.add(onto_embeddings, self.onto_output[0])

        ref_ent1=ref_ent1[:,None].expand(-1,self.config['dim'])
        ref_ins_embed = torch.gather(input=ins_embeddings,dim=0,index=ref_ent1)
        ref_ins_embed = ref_ins_embed
        ref_ins_embed = torch.matmul(ref_ins_embed, self.typing_mapping_matrix)
        ref_ins_embed = ref_ins_embed.cpu().detach().numpy()
        onto_embed = onto_embeddings.cpu().detach().numpy()
        eval_performance = eval_type_hyperbolic(ref_ins_embed, onto_embed, all_ref,
                                     self.config['ent_top_k'], greedy=True)        

        result = "ENT_TYPING\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f" % (
            eval_performance['mrr'],
            eval_performance['hits1'],
            eval_performance['hits3'],
            eval_performance['hits5'],
            eval_performance['hits10'])       

        logger.info("\n-------- S Evaluation Performance --------\n%s\n%s" % (
            "\t".join(["TASK\t", "MRR", "Hits@1", "Hits@3", "Hits@5", "Hits@10"]),
            result))

        end=time.time()
        logger.info("CROSS time: "+str(round(end - start, 3))+'s')
        return eval_performance['hits1']