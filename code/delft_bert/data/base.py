from torch.utils.data import Dataset
import dgl
import collections

SSTBatch = collections.namedtuple('SSTBatch', ['graph', 'label'])


def batcher(device):
    def batcher_dev(batch):
        batch_graphs = dgl.batch(batch)
        return SSTBatch(graph=batch_graphs,
                        label=batch_graphs.ndata['label'].to(device)
                        )
    return batcher_dev




class DELFTDataset(Dataset):
    def __init__(self, examples, args, tokenizer, istrain):
        self.examples = examples
        self.args = args
        self.istrain = istrain
        self.tokenizer = tokenizer
        
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return vectorize(self.examples[index], self.tokenizer, self.args.device, self.istrain)    
