import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
import dgl.function as fn
import numpy as np


'''
Each GNN layer
'''

class RGCNLayer(nn.Module):
    def __init__(self, args, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.args = args
        self.is_input_layer = is_input_layer
        self.evidence_input_encoder = Encoder_rnn(args)
        self.evidence_hidden_encoder = Encoder_rnn(args)
        self.evidence_word_attn = BilinearSeqAttn(self.args.hidden_size * 2, self.args.hidden_size * 2)
        self.evidence_sent_attn = BilinearSeqAttn(self.args.hidden_size * 2, self.args.hidden_size * 2)
        self.evidence_score = nn.Bilinear(self.args.hidden_size * 2, self.args.hidden_size * 2, 1)

        self.evidence_rep_linear = torch.nn.Sequential(
            nn.Linear(self.args.hidden_size * 2, self.args.hidden_size * 2),
            nn.Dropout(0.5),
            nn.ReLU())

        self.reduce_linear = torch.nn.Sequential(
            nn.Linear(self.args.hidden_size * 4, self.args.hidden_size * 2),
            nn.Dropout(0.5),
            nn.ReLU())

        self.hq_linear = torch.nn.Sequential(
            nn.Linear(self.args.hidden_size * 2, self.args.hidden_size * 2),
            nn.Dropout(0.5),
            nn.ReLU())
        self.h_linear = torch.nn.Sequential(
            nn.Linear(self.args.hidden_size * 2, self.args.hidden_size * 2),
            nn.Dropout(0.5),
            nn.ReLU())
        self.evi_linear = torch.nn.Sequential(
            nn.Linear(self.args.hidden_size * 2, self.args.hidden_size * 2),
            nn.Dropout(0.5),
            nn.ReLU())
    

    def propagate(self, g):
        if self.is_input_layer:
            '''
            Get edge representation
            '''
            def msg_func(edges):
                evi = edges.data['evidence']
                evi_mask = edges.data['evidence_mask']
                batch_size, sent_max_len, word_max_len = evi.size(0), evi.size(1), evi.size(2)
                evi = evi.view(batch_size * sent_max_len, word_max_len, -1)
                evi_mask = evi_mask.view(batch_size * sent_max_len, word_max_len)


                evi_len = evi_mask.data.eq(0).sum(1).cpu().numpy().tolist()
                evi_idx = [i for i in range(len(evi_len)) if evi_len[i] > 0]
                evi_mask = evi_mask[evi_idx]
                evi_len = [i for i in evi_len if i > 0]
                evi = evi[evi_idx]

                evidence_rep = self.evidence_input_encoder(evi, evi_len)

                entity_rep = edges.src['hq'].repeat(1, sent_max_len).view(batch_size * sent_max_len, -1)
                entity_rep = entity_rep[evi_idx]
                _, evidence_rep = self.evidence_word_attn(evidence_rep, entity_rep, evi_mask[:, :max(evi_len)])
                ### Have to duplicate it here
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                evidence_word_rep = torch.FloatTensor(batch_size * sent_max_len, 2 * self.args.hidden_size).zero_().to(device)
                evidence_word_rep[evi_idx] = evidence_rep
                evidence_word_rep = evidence_word_rep.view(batch_size, sent_max_len, 2 * self.args.hidden_size)

                edges.data['evi_rep'] = evidence_word_rep
                sent_len = edges.data['evidence_sent_mask'].eq(0).sum(1).unsqueeze(1).float()
                evidence_sent_rep = torch.div(evidence_word_rep.sum(1), sent_len)
                evi_sent_score = self.evidence_score(evidence_sent_rep, edges.src['hq'])
                
            
                return {'evi_sent_rep': evidence_sent_rep, 'score': evi_sent_score}
        else:
            def msg_func(edges):

                evidence_word_rep = self.evidence_rep_linear(edges.data['evi_rep'])
                sent_len = edges.data['evidence_sent_mask'].eq(0).sum(1).unsqueeze(1).float()
                evidence_sent_rep = torch.div(evidence_word_rep.sum(1), sent_len)

                
                edges.data['evi_rep'] = evidence_word_rep

                evi_sent_score = self.evidence_score(evidence_sent_rep, edges.src['hq'])

                return {'evi_sent_rep': evidence_sent_rep, 'score': evi_sent_score}
        
        def softmax_feat(edges):
            return {'normalized_score': F.softmax(edges.data['score'], dim=1)}

        def reduce_func(nodes):

            sc = nodes.mailbox['sc']
            v = nodes.mailbox['msg']
            c = nodes.mailbox['c']
            rep = self.reduce_linear(torch.cat((v, c), 2))

            return {'new_evi_rep': (sc * rep).sum(1)}
        

        def edge_aggv(edges):
            return {'msg': edges.data['evi_sent_rep'], 'sc': edges.data['normalized_score'], 'c': edges.src['h']}
        
        
        def apply_node_func(nodes):
            h_rep = self.h_linear(nodes.data['h'])
            hq_rep = self.hq_linear(nodes.data['hq'])
            if 'new_evi_rep' in nodes.data:
                
                evi_rep = self.evi_linear(nodes.data['new_evi_rep'])
                h = h_rep + hq_rep + evi_rep
            else:
                h = h_rep + hq_rep
            

            return {'h': h, 'hq': hq_rep}
        
        if 'evidence' in g.edata:
            g.apply_edges(msg_func)
            #### Get incoming edge score for each candidate node
            g.group_apply_edges(func=softmax_feat, group_by='src')
        
        
        g.update_all(edge_aggv, reduce_func, apply_node_func)

        
    def forward(self, g):
        self.propagate(g)




class Encoder_rnn(nn.Module):
    '''
    Encoder layer (GRU)
    '''
    def __init__(self, args):
        super(Encoder_rnn, self).__init__()
        self.args = args

        self.rnn = nn.GRU(input_size = self.args.input_size,
                          hidden_size = self.args.hidden_size,
                          num_layers = self.args.num_layers,
                          batch_first = True,
                          dropout = self.args.dropout,
                          bidirectional = True)

    

    def forward(self, alias, alias_len):  

        alias_len = np.array(alias_len)
        sorted_idx= np.argsort(-alias_len)
        alias = alias[sorted_idx]
        alias_len = alias_len[sorted_idx]
        unsorted_idx = np.argsort(sorted_idx)

        packed_emb = torch.nn.utils.rnn.pack_padded_sequence(alias, alias_len, batch_first=True)
        output, hn = self.rnn(packed_emb)
        unpacked, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(output)
        unpacked = unpacked.transpose(0, 1)
        unpacked = unpacked[torch.LongTensor(unsorted_idx)]
        return unpacked

'''
Adopted from DrQA repo
'''

class LinearAttn(nn.Module):
    def __init__(self, input_size):
        super(LinearAttn, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x, x_mask):
        """
        self-attention Layer

        Args:
            x: batch * len * hdim
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha: batch * len
        """
        x_flat = x.view(x.size(0) * x.size(1), x.size(2))
        ##### change to batch * len * hdim
        scores = self.linear(x_flat).view(x.size(0), x.size(1))
      
        scores.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(scores, dim=1)

        #x = x.transpose(0, 1)
        output_avg = alpha.unsqueeze(1).bmm(x).squeeze(1)

        return output_avg

'''
Adopted from DrQA repo
'''
             
class BilinearSeqAttn(nn.Module):
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.
    Optionally don't normalize output weights.
    """

    def __init__(self, x_size, y_size, identity=False, normalize=True):
        super(BilinearSeqAttn, self).__init__()
        self.normalize = normalize

        # If identity is true, we just use a dot product without transformation.
        if not identity:
            self.linear = nn.Linear(y_size, x_size)
        else:
            self.linear = None

    def forward(self, x, y, x_mask):
        """
        Args:
            x: batch * len * hdim1
            y: batch * hdim2
            x_mask: batch * len (1 for padding, 0 for true)
        Output:
            alpha = batch * len
        """
        Wy = self.linear(y) if self.linear is not None else y
        xWy = x.bmm(Wy.unsqueeze(2)).squeeze(2)
        xWy.data.masked_fill_(x_mask.data, -float('inf'))
        alpha = F.softmax(xWy, dim=-1)

        matched_seq = alpha.unsqueeze(1).bmm(x).squeeze(1)

        return alpha, matched_seq



class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.question_rnn = Encoder_rnn(args)
        self.question_attn = LinearAttn(self.args.hidden_size * 2)
        self.first_sent_rnn = Encoder_rnn(args)
        self.first_sent_attn = LinearAttn(self.args.hidden_size * 2)
        self.evidence_encoder = Encoder_rnn(args)
        self.num_hidden_layers = 1
        self.device = args.device
        self.build_model()
        self.dropout = nn.Dropout(0.5)
        self.final_layer = torch.nn.Sequential(
            nn.Linear(self.args.hidden_size * 2, self.args.hidden_size),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(self.args.hidden_size, 1)    
        )
    
    
 
        
        

    

    def build_model(self):
        self.layers = nn.ModuleList()
        # input to hidden
        i2h = self.build_input_layer().to(self.device)
        self.layers.append(i2h)
        # hidden to hidden
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer().to(self.device)
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer().to(self.device)
        self.layers.append(h2o)
    
    
    def build_input_layer(self):
        return RGCNLayer(self.args, is_input_layer=True)
    
    def build_hidden_layer(self):
        return RGCNLayer(self.args)

    def build_output_layer(self):
        return RGCNLayer(self.args)
    
    
    def forward(self, batch):
        #### first layer initialize the features 
        g = batch.graph

        g.ndata['question'] = g.ndata['question'].to(self.device)
        g.ndata['question_mask'] = g.ndata['question_mask'].to(self.device)
        g.ndata['first_sent_mask'] = g.ndata['first_sent_mask'].to(self.device)
        g.ndata['first_sent'] = g.ndata['first_sent'].to(self.device)
        g.ndata['label'] = g.ndata['label'].to(self.device)
        
        if 'evidence' in g.edata:
            g.edata['evidence'] = g.edata['evidence'].to(self.device)
            g.edata['evidence_mask'] = g.edata['evidence_mask'].to(self.device)
            g.edata['evidence_sent_mask'] = g.edata['evidence_sent_mask'].to(self.device)
        
        
        q_len = g.ndata['question_mask'].data.eq(0).sum(1).cpu().numpy().tolist()
        question_rep = self.question_rnn(g.ndata['question'], q_len)
        max_length = max(q_len)
        xq_mask = g.ndata['question_mask'][:, :max_length]
        question_rep = self.question_attn(question_rep, xq_mask)
        g.ndata['hq'] = question_rep

        sent_len = g.ndata['first_sent_mask'].data.eq(0).sum(1).cpu().numpy().tolist()
        sent_rep = self.first_sent_rnn(g.ndata['first_sent'], sent_len)
        
        max_sent_length = max(sent_len)
        xsent_mask = g.ndata['first_sent_mask'][:, :max_sent_length]
        sent_rep = self.question_attn(sent_rep, xsent_mask)
        g.ndata['h'] = sent_rep
        
        
        
        



        
        for layer in self.layers:
            layer(g)
        all_hiddens = g.ndata.pop('h')
        logits = self.final_layer(all_hiddens).squeeze()
        return logits
