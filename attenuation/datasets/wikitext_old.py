import math
import torch
import datasets
import torchtext
from tqdm import tqdm
from torchtext.data.utils import get_tokenizer

from attenuation.model.gpt2 import GPT2


## BUILD DATALOADER
def get_data(dataset, vocab, batch_size):
    data = []                                                   
    for example in dataset:
        if example['tokens']:                                      
            tokens = example['tokens'].append('<eos>')             
            tokens = [vocab[token] for token in example['tokens']] 
            data.extend(tokens)                                    
    data = torch.LongTensor(data)                                 
    num_batches = data.shape[0] // batch_size 
    data = data[:num_batches * batch_size]                       
    data = data.view(batch_size, num_batches)          
    return data


def get_batch(data, seq_len, num_batches, idx):
    src = data[:, idx:idx+seq_len]                   
    target = data[:, idx+1:idx+seq_len+1]             
    return src, target


def train(model, data, optimizer, criterion, batch_size, seq_len, clip):
    
    epoch_loss = 0
    model.train()
    device = model.device
    # drop all batches that are not a multiple of seq_len
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)
    
    for idx in tqdm(range(0, num_batches - 1, seq_len), desc='Training: ',leave=False):  # The last batch can't be a src
        optimizer.zero_grad()
        hidden = model.detach_hidden(hidden)

        src, target = get_batch(data, seq_len, num_batches, idx)
        src, target = src.to(device), target.to(device)
        batch_size = src.shape[0]
        prediction, hidden = model(src, hidden)               

        prediction = prediction.reshape(batch_size * seq_len, -1)   
        target = target.reshape(-1)
        loss = criterion(prediction, target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches


def evaluate(model, data, criterion, batch_size, seq_len, device):

    epoch_loss = 0
    model.eval()
    num_batches = data.shape[-1]
    data = data[:, :num_batches - (num_batches -1) % seq_len]
    num_batches = data.shape[-1]

    hidden = model.init_hidden(batch_size, device)

    with torch.no_grad():
        for idx in range(0, num_batches - 1, seq_len):
            hidden = model.detach_hidden(hidden)
            src, target = get_batch(data, seq_len, num_batches, idx)
            src, target = src.to(device), target.to(device)
            batch_size= src.shape[0]

            prediction, hidden = model(src, hidden)
            prediction = prediction.reshape(batch_size * seq_len, -1)
            target = target.reshape(-1)

            loss = criterion(prediction, target)
            epoch_loss += loss.item() * seq_len
    return epoch_loss / num_batches


def generate(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']:
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:
                break

            indices.append(prediction)

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens



if __name__ == "__main__":

    ## GET DATASET
    # export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
    # dataset = datasets.load_dataset('wikitext', 'wikitext-103')
    # dataset = datasets.load_dataset(path="wikitext", name="wikitext-103-v1", split="train")
    # dataset = datasets.load_dataset('/home/brothag1/code/wikitext/wikitext.py', 'wikitext-2-raw-v1')
    # dataset = datasets.load_dataset('/home/brothag1/code/wikitext/wikitext.py', 'wikitext-103-raw-v1')
    dataset = datasets.load_dataset('wikitext', 'wikitext-2-raw-v1')
    print(dataset)
    print(dataset['train'][88]['text'])

    ## GET TOKENIZER
    tokenizer = get_tokenizer('basic_english')
    tokenize_data = lambda example, tokenizer: {'tokens': tokenizer(example['text'])}  
    tokenized_dataset = dataset.map(tokenize_data, remove_columns=['text'], 
    fn_kwargs={'tokenizer': tokenizer})
    print(tokenized_dataset['train'][88]['tokens'])

    ## GET VOCAB
    vocab = torchtext.vocab.build_vocab_from_iterator(tokenized_dataset['train']['tokens'], 
    min_freq=3) 
    vocab.insert_token('<unk>', 0)           
    vocab.insert_token('<eos>', 1)            
    vocab.set_default_index(vocab['<unk>'])   
    print(len(vocab))                         
    print(vocab.get_itos()[:10])  

    ## GET SETS
    batch_size = 128
    train_data = get_data(tokenized_dataset['train'], vocab, batch_size)
    valid_data = get_data(tokenized_dataset['validation'], vocab, batch_size)
    test_data = get_data(tokenized_dataset['test'], vocab, batch_size)
    # We have 16214 batches, each of 128 words

    ## INITIALIZE MODEL
    lr = 2.5e-4
    model = GPT2(num_vocab=len(vocab))
    device = model.device()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'The model has {num_params:,} trainable parameters')


    ## TRAIN MODEL
    n_epochs = 50
    seq_len = 50
    clip = 0.25
    saved = False

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=0)

    if saved:
        model.load_state_dict(torch.load('best-val-gpt2_lm.pt',  map_location=device))
        test_loss = evaluate(model, test_data, criterion, batch_size, seq_len, device)
        print(f'Test Perplexity: {math.exp(test_loss):.3f}')
    else:
        best_valid_loss = float('inf')

        for epoch in range(n_epochs):
            train_loss = train(model, train_data, optimizer, criterion, 
                        batch_size, seq_len, clip, device)
            valid_loss = evaluate(model, valid_data, criterion, batch_size, 
                        seq_len, device)
            
            lr_scheduler.step(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'best-val-gpt2_lm.pt')

            print(f'\tTrain Perplexity: {math.exp(train_loss):.3f}')
            print(f'\tValid Perplexity: {math.exp(valid_loss):.3f}')

    ## INFERENCE
    prompt = 'Think about'
    max_seq_len = 30
    seed = 0

    temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]
    for temperature in temperatures:
        generation = generate(prompt, max_seq_len, temperature, model, tokenizer, 
                            vocab, device, seed)
        print(str(temperature)+'\n'+' '.join(generation)+'\n')