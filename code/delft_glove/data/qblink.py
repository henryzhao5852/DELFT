from nltk import word_tokenize
import dgl
from dgl import DGLGraph
import dgl.function as fn
import torch
from .utils import text_tokenize
from .base import DELFTDataset
import unicodedata

def normalize(text):
    return unicodedata.normalize('NFD', text).replace(' ', '_').lower()

'''
Data vectorization into DGL graph
'''

def vectorize_qblink(ex, model, istrain, max_seq_length=64):
    q_id = ex['id']
    text = ex['text']
    positive_entity = ex['pos_et']
    negative_entities = ex['neg_ets']
    ### Maximum 5 sentences per edge
    num_edges = 5

    g = DGLGraph()

    question_node_list = list()
    candidate_node_list = list()
    first_sent_tokens = list()
    first_sent_masks = list()
    question_tokens = list()
    question_masks = list()

    input_ids, input_mask = text_tokenize(text, model.word_dict, max_seq_length)
    question_tokens.append(input_ids)
    question_masks.append(input_mask)
    node_sub_questions = list()
    

    for sup_q in ex['q_et']:
        
        sub_question = sup_q['text']
        input_ids, input_mask = text_tokenize(word_tokenize(sub_question), model.word_dict, max_seq_length)
        question_tokens.append(input_ids)
        question_masks.append(input_mask)

        for et in sup_q['entity']:
            topic = et['et']
            node_first_sent = et['first_sent']
            if topic is None:
                continue

            question_node_list.append(topic)
            question_idx = len(question_tokens) - 1
            node_sub_questions.append(question_idx)
            input_ids, input_mask = text_tokenize(node_first_sent, model.word_dict, max_seq_length)
            
            first_sent_tokens.append(input_ids)
            first_sent_masks.append(input_mask)


    candidate_node_list.append(normalize(positive_entity['et']))
    input_ids, input_mask = text_tokenize(positive_entity['first_sent'], model.word_dict, max_seq_length)
    first_sent_tokens.append(input_ids)
    first_sent_masks.append(input_mask)
    node_sub_questions.append(0)
    for neg_et in negative_entities:
        input_ids, input_mask = text_tokenize(neg_et['first_sent'], model.word_dict, max_seq_length)
        
        candidate_node_list.append(normalize(neg_et['et']))

        first_sent_tokens.append(input_ids)
        first_sent_masks.append(input_mask)
        node_sub_questions.append(0)

    num_nodes = len(question_node_list) + len(candidate_node_list)
    g.add_nodes(num_nodes)

    num_questions = len(question_tokens)

    ### combine question and first sentence
    all_tokens = question_tokens + first_sent_tokens
    all_masks = question_masks + first_sent_masks

    all_tensor = torch.LongTensor(all_tokens)
    all_masks_tensor = torch.LongTensor(all_masks)

    #### add node features
    g.ndata['first_sent'] = all_tensor[num_questions:].cpu()
    g.ndata['first_sent_mask'] = all_masks_tensor[num_questions:].eq(0)
  

    for i in range(len(question_node_list)):
        sub_q_num = node_sub_questions[i]
        
        g.nodes[i].data['question'] = all_tensor[sub_q_num].unsqueeze(0)
        g.nodes[i].data['question_mask'] = all_masks_tensor[sub_q_num].unsqueeze(0).eq(0)
        g.nodes[i].data['label'] = torch.tensor(-1).unsqueeze(0)
        
    g.nodes[len(question_node_list)].data['question'] = all_tensor[0].unsqueeze(0)
    g.nodes[len(question_node_list)].data['question_mask'] = all_masks_tensor[0].unsqueeze(0).eq(0)
    g.nodes[len(question_node_list)].data['label'] = torch.tensor(1).unsqueeze(0)
    #### for candidates, we only use the full question sentence
    for i in range(len(question_node_list) + 1, len(question_node_list) + len(candidate_node_list)):
        g.nodes[i].data['question'] = all_tensor[0].unsqueeze(0)
        g.nodes[i].data['question_mask'] = all_masks_tensor[0].unsqueeze(0).eq(0)
        g.nodes[i].data['label'] = torch.tensor(0).unsqueeze(0)
        

    #### add postive edges
        
    for k_entity in positive_entity['evidence']:
        normalized_k_entity = normalize(k_entity)
        if normalized_k_entity in question_node_list:
            s_id = question_node_list.index(normalized_k_entity)
            g.add_edge(question_node_list.index(normalized_k_entity), len(question_node_list))
            evidence_tokens = list()
            evidence_masks = list()
            all_evidences = positive_entity['evidence'][k_entity]
            
            for evi_text in all_evidences[:num_edges]:
                input_ids, input_mask = text_tokenize(evi_text, model.word_dict, max_seq_length)
                
                evidence_tokens.append(input_ids)
                evidence_masks.append(input_mask)

            evidence_tensor = torch.LongTensor(evidence_tokens)
            evidence_masks_tensor = torch.LongTensor(evidence_masks)

            edge_features = torch.LongTensor(1, num_edges, max_seq_length).zero_()
            edge_feature_masks = torch.LongTensor(1, num_edges, max_seq_length).zero_()
            egde_sent_mask = torch.ByteTensor(1, num_edges).fill_(1)
            edge_features[0, :len(evidence_tokens), :].copy_(evidence_tensor)
            edge_feature_masks[0, :len(evidence_tokens), :].copy_(evidence_masks_tensor)
            egde_sent_mask[0, :len(evidence_tokens)].fill_(0)
     
            g.edges[s_id, len(question_node_list)].data['evidence'] = edge_features
            g.edges[s_id, len(question_node_list)].data['evidence_mask'] = edge_feature_masks.eq(0)
            g.edges[s_id, len(question_node_list)].data['evidence_sent_mask'] = egde_sent_mask


        
    for neg_et in negative_entities:
        #### 
        for k_entity in neg_et['evidence']:
            normalized_k_entity = normalize(k_entity)
            if normalized_k_entity in question_node_list:
                s_id = question_node_list.index(normalized_k_entity)
                t_id = len(question_node_list) + candidate_node_list.index(normalize(neg_et['et']))
                g.add_edge(s_id, t_id)
                    
                evidence_tokens = list()
                evidence_masks = list()
                all_evidences = neg_et['evidence'][normalized_k_entity]
                for evi_text in all_evidences[:num_edges]:
                    input_ids, input_mask = text_tokenize(evi_text, model.word_dict, max_seq_length)
                    evidence_tokens.append(input_ids)
                    evidence_masks.append(input_mask)
                evidence_tensor = torch.LongTensor(evidence_tokens)
                evidence_masks_tensor = torch.LongTensor(evidence_masks)

                edge_features = torch.LongTensor(1, num_edges, max_seq_length).zero_()
                edge_feature_masks = torch.LongTensor(1, num_edges, max_seq_length).zero_()
                egde_sent_mask = torch.ByteTensor(1, num_edges).fill_(1)
                edge_features[0, :len(evidence_tokens), :].copy_(evidence_tensor)
                edge_feature_masks[0, :len(evidence_tokens), :].copy_(evidence_masks_tensor)
                egde_sent_mask[0, :len(evidence_tokens)].fill_(0)
     
                g.edges[s_id, t_id].data['evidence'] = edge_features
                g.edges[s_id, t_id].data['evidence_mask'] = edge_feature_masks.eq(0)
                g.edges[s_id, t_id].data['evidence_sent_mask'] = egde_sent_mask

    return g


class QBLINKDataset(DELFTDataset):
    def __init__(self, examples, model, istrain):
        super(QBLINKDataset, self).__init__(examples, model, istrain)


    def __getitem__(self, index):
        return vectorize_qblink(self.examples[index], self.model, self.istrain)    

