import functools
import datasets
import torchtext
import torch
import torch.nn as nn
import torch.optim as optim

from models.NBoW import NBoW
from models.CNN import CNN
from models.LSTM import LSTM
import argparse

# Parse the argument to determine the model architecture we are using and our training epochs
parser = argparse.ArgumentParser()
parser.add_argument("--model_arch", type=str, default='cnn',
                    help='specify the model architecture we are using for sentiment analysis')
parser.add_argument("--n_epochs", type=int, default=10,
                    help='specify the number of epochs to train the model')
args = parser.parse_args()

# Load training and testing data and tokenize them
train_data, test_data = datasets.load_dataset('imdb', split=['train', 'test'])
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

def tokenize_data(example, tokenizer):
    tokens = {'tokens': tokenizer(example['text'])}
    return tokens

train_data = train_data.map(tokenize_data, fn_kwargs={'tokenizer': tokenizer})
test_data = test_data.map(tokenize_data, fn_kwargs={'tokenizer': tokenizer})


# Split the training and validation data 
train_valid_data = train_data.train_test_split(test_size=0.25)
train_data = train_valid_data['train']
valid_data = train_valid_data['test']

min_freq = 3
special_tokens = ['<unk>', '<pad>']

# Build the vocabulary
vocab = torchtext.vocab.build_vocab_from_iterator(train_data['tokens'],
                                min_freq=min_freq,
                                specials=special_tokens)

unk_index = vocab['<unk>']
pad_index = vocab['<pad>']

vocab.set_default_index(unk_index)

def numericalize_data(example, vocab):
    ids = {'ids': [vocab[token] for token in example['tokens']]}
    return ids

train_data = train_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
valid_data = valid_data.map(numericalize_data, fn_kwargs={'vocab': vocab})
test_data = test_data.map(numericalize_data, fn_kwargs={'vocab': vocab})

train_data.set_format(type='torch', columns=['ids', 'label'])
valid_data.set_format(type='torch', columns=['ids', 'label'])
test_data.set_format(type='torch', columns=['ids', 'label'])


vocab_size = len(vocab)
embedding_dim = 256
output_dim = 2


# Determine the model architecture according to the parameter
if args.model_arch == "cnn":
    vocab_size = len(vocab)
    embedding_dim = 300
    n_filters = 100
    filter_sizes = [3,5,7]
    output_dim = len(train_data.unique('label'))
    dropout_rate = 0.25
    model = CNN(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout_rate, pad_index)
elif args.model_arch == "nbow":
    model = NBoW(vocab_size, embedding_dim, output_dim, pad_index)
else:
    raise NotImplementedError


# Initialize the embedding mapping, the optimizer, and the loss function
vectors = torchtext.vocab.GloVe()
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = model.to(device)
criterion = criterion.to(device)

def collate(batch, pad_index):
    batch_ids = [i['ids'] for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_labels = [i['label'] for i in batch]
    batch_labels = torch.stack(batch_labels)
    batch = {'ids': batch_ids,
             'labels': batch_labels}
    return batch

batch_size = 512

collate = functools.partial(collate, pad_index=pad_index)

# Specify dataloaders
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, collate_fn=collate)
valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, collate_fn=collate)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, collate_fn=collate)

# Training function
def train(dataloader, model, criterion, optimizer, device):

    model.train()
    epoch_loss = 0
    epoch_accuracy = 0

    for batch in dataloader:
        tokens = batch['ids'].to(device)
        labels = batch['labels'].to(device)
        if args.model_arch == "lstm":
            predictions = model(tokens, batch['length'].to(device))
        else:
            predictions = model(tokens)
        loss = criterion(predictions, labels)
        accuracy = get_accuracy(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_accuracy += accuracy.item()

    return epoch_loss / len(dataloader), epoch_accuracy / len(dataloader)

# Evaluation function
def evaluate(dataloader, model, criterion, device):
    
    model.eval()
    epoch_loss = 0
    epoch_accuracy = 0

    with torch.no_grad():
        for batch in dataloader:
            tokens = batch['ids'].to(device)
            labels = batch['labels'].to(device)
            predictions = model(tokens)
            loss = criterion(predictions, labels)
            accuracy = get_accuracy(predictions, labels)
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()

    return epoch_loss / len(dataloader), epoch_accuracy / len(dataloader)


# The function that calculates the accuracy
def get_accuracy(predictions, labels):
    batch_size = predictions.shape[0]
    predicted_classes = predictions.argmax(1, keepdim=True)
    correct_predictions = predicted_classes.eq(labels.view_as(predicted_classes)).sum()
    accuracy = correct_predictions.float() / batch_size
    return accuracy

n_epochs = args.n_epochs

for epoch in range(n_epochs):

    train_loss, train_acc = train(train_dataloader, model, criterion, optimizer, device)
    valid_loss, valid_acc = evaluate(valid_dataloader, model, criterion, device)

    print(f'epoch: {epoch+1}')
    print(f'train_loss: {train_loss:.3f}, train_acc: {train_acc:.3f}')
    print(f'valid_loss: {valid_loss:.3f}, valid_acc: {valid_acc:.3f}')
