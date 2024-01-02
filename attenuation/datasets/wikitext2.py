import torch
from torch import Tensor
from torch.utils.data import dataset
from torchtext.datasets import WikiText2
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer


DATASET_NAME = "WikiText2"

def load():
    ## CONSTRUCT VOCAB
    tokenizer = get_tokenizer('basic_english')
    sample_iter = WikiText2(split='train')
    vocab = build_vocab_from_iterator(map(tokenizer, sample_iter), specials=['<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    ## CONVERT DATA FROM STRINGS TO TENSORS
    train_iter, val_iter, test_iter = WikiText2()
    train_data = process_iter(train_iter, vocab, tokenizer)
    val_data = process_iter(val_iter, vocab, tokenizer)
    test_data = process_iter(test_iter, vocab, tokenizer)

    return train_data, val_data, test_data, len(vocab)

def process_iter(
        raw_text_iter: dataset.IterableDataset, 
        vocab, 
        tokenizer, 
        **kwargs
    ) -> Tensor:

    ## CONVERT RAW TEXT ITERABLE INTO A FLAT TENSOR
    data = [
        torch.tensor(vocab(tokenizer(item)), dtype=torch.long) 
        for item in raw_text_iter
    ]
    return torch.cat(tuple(filter(lambda t: t.numel() > 0, data)))




    # train_iter, val_iter, test_iter = EnWik9()
    # train_iter, test_iter = IMDB(split=('train','test'))
    # enwik9 = EnWik9()
    # train_iter, val_iter, test_iter = IWSLT2017(root='.data', split=('train', 'valid', 'test'), language_pair=('it', 'en'))
