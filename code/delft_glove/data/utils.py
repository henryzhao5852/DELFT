import unicodedata
import json
import logging


logger = logging.getLogger()



def load_data(data_file):  
    data = list()
    with open(data_file) as f:
        for line in f.readlines():
            ex = json.loads(line)
            data.append(ex)

    return data

'''
Adopted from DrQA repo
'''

class Dictionary(object):
    NULL = '<NULL>'
    UNK = '<UNK>'
    START = 2

    @staticmethod
    def normalize(token):
        return unicodedata.normalize('NFD', token)

    def __init__(self):
        self.tok2ind = {self.NULL: 0, self.UNK: 1}
        self.ind2tok = {0: self.NULL, 1: self.UNK}

    def __len__(self):
        return len(self.tok2ind)

    def __iter__(self):
        return iter(self.tok2ind)

    def __contains__(self, key):
        if type(key) == int:
            return key in self.ind2tok
        elif type(key) == str:
            return self.normalize(key) in self.tok2ind

    def __getitem__(self, key):
        if type(key) == int:
            return self.ind2tok.get(key, self.UNK)
        if type(key) == str:
            return self.tok2ind.get(self.normalize(key),
                                    self.tok2ind.get(self.UNK))

    def __setitem__(self, key, item):
        if type(key) == int and type(item) == str:
            self.ind2tok[key] = item
        elif type(key) == str and type(item) == int:
            self.tok2ind[key] = item
        else:
            raise RuntimeError('Invalid (key, item) types.')

    def add(self, token):
        token = self.normalize(token)
        if token not in self.tok2ind:
            index = len(self.tok2ind)
            self.tok2ind[token] = index
            self.ind2tok[index] = token

    def tokens(self):
        """Get dictionary tokens.
        Return all the words indexed by this dictionary, except for special
        tokens.
        """
        tokens = [k for k in self.tok2ind.keys()
                  if k not in {'<NULL>', '<UNK>'}]
        return tokens


    
def index_embedding_words(embedding_file):
    """Put all the words in embedding_file into a set."""
    words = set()
    with open(embedding_file) as f:
        for line in f:
            w = Dictionary.normalize(line.rstrip().split(' ')[0])
            words.add(w)
    return words


def load_words(args, examples):
    """Iterate and index all the words in examples (documents + questions)."""
    def _insert(iterable):
        for w in iterable:
            w = Dictionary.normalize(w.lower())
            if valid_words and w not in valid_words:
                continue
            words.add(w)
    
    if args.embedding_file:
        logger.info('Restricting to words in %s' % args.embedding_file)
        valid_words = index_embedding_words(args.embedding_file)

        logger.info('Num words in set = %d' % len(valid_words))
    else:
        valid_words = None
    
    words = set()
    word_dict = Dictionary()
    for ex in examples:
        text = ex['text']
        positive_entity = ex['pos_et']
        negative_entities = ex['neg_ets']
        _insert(text)
        pos_first_sent = positive_entity['first_sent']
        _insert(pos_first_sent)
        for q_et, evidences in positive_entity['evidence'].items():
            for evi in evidences:
                
                _insert(evi)
        
        for neg_et in negative_entities:
            neg_first_sent = neg_et['first_sent']
            _insert(neg_first_sent)
            for q_et, evidences in neg_et['evidence'].items():
                for evi in evidences:
                    _insert(evi)
    
    for w in words:
        word_dict.add(w)
    return word_dict            
        



def text_tokenize(text, word_dict, max_seq_length):

    input_ids = [word_dict[w.lower()] for w in text[-max_seq_length:]]
    if len(input_ids) == 0:
        input_ids.append(1)
    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
    
    return input_ids, input_mask
