from nltk import word_tokenize
import dgl
from dgl import DGLGraph
import dgl.function as fn
import torch
from .utils import text_tokenize
from .base import DELFTDataset
import unicodedata
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers import AdamW, WarmupLinearSchedule
from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_transformers.modeling_bert import BertModel, BertEncoder, BertPreTrainedModel



def normalize(text):
    return unicodedata.normalize('NFD', text).replace(' ', '_').lower()


class GNNBertEncoder(BertEncoder):
    def __init__(self, config):
        super(GNNBertEncoder, self).__init__(config)
        self.config = config

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)[0]
            pooled_output = hidden_states[:, 0]
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers



class GNNBertModel(BertModel):
    def __init__(self, config):
        super(GNNBertModel, self).__init__(config)

        self.encoder = GNNBertEncoder(config)
    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0



        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(embedding_output,
                                       extended_attention_mask)
        ## We average the last three layers output as BERT embeddings 
        sequence_output = encoder_outputs[-3:]
        sequence_output = torch.mean(torch.stack(sequence_output), dim=0)
        pooled_output = self.pooler(encoder_outputs[-1])
        outputs = sequence_output, pooled_output  # add hidden_states and attentions if they are here

        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)



bert_model = GNNBertModel.from_pretrained('bert-base-uncased', cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(-1)).to('cuda')

'''
Data vectorization into DGL graph
'''

def vectorize_qanta(ex, tokenizer, device, istrain, max_seq_length=64):
    bert_model.eval()
    t_id = ex['id']
    text = ex['text']
    positive_entity = ex['pos_et']
    negative_entities = ex['neg_ets']
    ## In QANTA setting, we limit the maximum sentences as three ( for efficient training and evaluation)
    num_edges = 3
    g = DGLGraph()
    question_node_list = list()
    candidate_node_list = list()
    first_sent_tokens = list()
    first_sent_masks = list()
    question_tokens = list()
    question_masks = list()

    input_ids, input_mask = text_tokenize(text, tokenizer, max_seq_length)
    question_tokens.append(input_ids)
    question_masks.append(input_mask)
    node_sub_questions = list()


    for sup_q in ex['q_et']:
        sub_question = sup_q['text']
        input_ids, input_mask = text_tokenize(sub_question, tokenizer, max_seq_length)
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
            input_ids, input_mask = text_tokenize(node_first_sent, tokenizer, max_seq_length)
            
            first_sent_tokens.append(input_ids)
            first_sent_masks.append(input_mask)




   

    candidate_node_list.append(normalize(positive_entity['et']))
    input_ids, input_mask = text_tokenize(positive_entity['first_sent'], tokenizer, max_seq_length)
    first_sent_tokens.append(input_ids)
    first_sent_masks.append(input_mask)
    node_sub_questions.append(0)

    for neg_et in negative_entities:
        candidate_node_list.append(normalize(neg_et['et']))
        input_ids, input_mask = text_tokenize(neg_et['first_sent'], tokenizer, max_seq_length)
        
        first_sent_tokens.append(input_ids)
        first_sent_masks.append(input_mask)
        node_sub_questions.append(0)

    num_nodes = len(question_node_list) + len(candidate_node_list)
    g.add_nodes(num_nodes)

    num_questions = len(question_tokens)

    ### combine question and first sentence
    all_tokens = question_tokens + first_sent_tokens
    all_masks = question_masks + first_sent_masks

    all_tensor = torch.LongTensor(all_tokens).to(device)
    all_masks_tensor = torch.LongTensor(all_masks).to(device)
    all_encodings = list()
    num_exs = 50
    for iii in range(int(all_tensor.size(0) / num_exs)):
        encoding, _ = bert_model(all_tensor[iii * num_exs: (iii + 1) * num_exs], None, all_masks_tensor[iii * num_exs: (iii + 1) * num_exs])
        encoding = encoding.detach().cpu()
        all_encodings.append(encoding)
    if all_tensor.size(0) % num_exs > 0:
        encoding, _ = bert_model(all_tensor[int(all_tensor.size(0) / num_exs)  * num_exs: ], None, all_masks_tensor[int(all_tensor.size(0) / num_exs)  * num_exs: ])
        encoding = encoding.detach().cpu()
        all_encodings.append(encoding)
    all_encodings = torch.cat(all_encodings, dim=0)
    
    all_masks_tensor = all_masks_tensor.cpu()



    g.ndata['first_sent'] = all_encodings[num_questions:].cpu()
    g.ndata['first_sent_mask'] = all_masks_tensor[num_questions:].cpu().eq(0)

  


    for i in range(len(question_node_list)):
        sub_q_num = node_sub_questions[i]
        
        g.nodes[i].data['question'] = all_encodings[sub_q_num].unsqueeze(0)
        g.nodes[i].data['question_mask'] =all_masks_tensor[sub_q_num].unsqueeze(0).eq(0)
        g.nodes[i].data['label'] = torch.tensor(-1).unsqueeze(0)

        
    g.nodes[len(question_node_list)].data['question'] = all_encodings[0].unsqueeze(0)
    g.nodes[len(question_node_list)].data['question_mask'] = all_masks_tensor[0].unsqueeze(0).eq(0)
    g.nodes[len(question_node_list)].data['label'] = torch.tensor(1).unsqueeze(0)
    #### for candidates, we only use the full question sentence
    for i in range(len(question_node_list) + 1, len(question_node_list) + len(candidate_node_list)):
        g.nodes[i].data['question'] = all_encodings[0].unsqueeze(0)
        g.nodes[i].data['question_mask'] = all_masks_tensor[0].unsqueeze(0).eq(0)
        g.nodes[i].data['label'] = torch.tensor(0).unsqueeze(0)
    
    




    for k_entity in positive_entity['evidence']:
        normalized_k_entity = normalize(k_entity)
        if normalized_k_entity in question_node_list:
            s_id = question_node_list.index(normalized_k_entity)
            g.add_edge(question_node_list.index(normalized_k_entity), len(question_node_list))
            evidence_tokens = list()
            evidence_masks = list()
            evidence_ids = list()
            all_evidences = positive_entity['evidence'][k_entity]
            
            for evi_text in all_evidences[:num_edges]:
                input_ids, input_mask = text_tokenize(evi_text, tokenizer, max_seq_length)
                
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
            g.edges[s_id, len(question_node_list)].data['evidence_mask'] = edge_feature_masks
            g.edges[s_id, len(question_node_list)].data['evidence_sent_mask'] = egde_sent_mask


        
    for neg_et in negative_entities:
        for k_entity in neg_et['evidence']:
            normalized_k_entity = normalize(k_entity)
            if normalized_k_entity in question_node_list:
                s_id = question_node_list.index(normalized_k_entity)
                t_id = len(question_node_list) + candidate_node_list.index(normalize(neg_et['et']))
                g.add_edge(s_id, t_id)
                    
                evidence_tokens = list()
                evidence_masks = list()
                evidence_ids = list()
                all_evidences = neg_et['evidence'][normalized_k_entity]
                for evi_text in all_evidences[:num_edges]:
                    input_ids, input_mask = text_tokenize(evi_text, tokenizer, max_seq_length)
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
                g.edges[s_id, t_id].data['evidence_mask'] = edge_feature_masks
                g.edges[s_id, t_id].data['evidence_sent_mask'] = egde_sent_mask
    
    ### Batch the sentences and get BERT embeddings
    if 'evidence' in g.edata:
        evi = g.edata['evidence'].to(device)
        evi_mask = g.edata['evidence_mask'].to(device)
        batch_size, sent_max_len, word_max_len = evi.size(0), evi.size(1), evi.size(2)
        evi = evi.view(batch_size * sent_max_len, word_max_len)
        evi_mask = evi_mask.view(batch_size * sent_max_len, word_max_len)
        all_encodings = list()
        num_exs = 50
        for iii in range(int(evi.size(0) / num_exs)):
            encoding, _ = bert_model(evi[iii * num_exs: (iii + 1) * num_exs], None, evi_mask[iii * num_exs: (iii + 1) * num_exs])
            encoding = encoding.detach().cpu()
            all_encodings.append(encoding)
        if evi.size(0) % num_exs > 0:
            encoding, _ = bert_model(evi[int(evi.size(0) / num_exs)  * num_exs: ], None, evi_mask[int(evi.size(0) / num_exs)  * num_exs: ])
            encoding = encoding.detach().cpu()
            all_encodings.append(encoding)
        
        g.edata['evidence'] = torch.cat(all_encodings, dim=0).view(batch_size, sent_max_len, word_max_len, -1)
        g.edata['evidence_mask'] = g.edata['evidence_mask'].eq(0)


  

    return g 






class QANTADataset(DELFTDataset):
    def __init__(self, examples, args, tokenizer, istrain):
        super(QANTADataset, self).__init__(examples, args, tokenizer, istrain)


    def __getitem__(self, index):
        return vectorize_qanta(self.examples[index], self.tokenizer, self.args.device, self.istrain)    

