import json
from nltk_utils import tokenize, stem, bag_of_words
from model import NeuralNet

import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

with open('intents.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
xy = []

# Loop through each sentence in our intents patterns
for intent in intents['intents']:   #key: intents, value: list of intents
    tag = intent['tag']             #key: tag, value: intent
    tags.append(tag)
    # Loop through each pattern in the patterns
    for pattern in intent['patterns']:
        # Tokenize each word in the sentence
        w = tokenize(pattern)
        # Add to our words list (not append, because we don't want a list of lists)
        all_words.extend(w)
        # Add to xy pair
        # pattern and tag for each pattern
        xy.append((w, tag))

# Stem and lower each word and remove duplicates
ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]

# Sort all words and remove duplicates
all_words = sorted(set(all_words))
# Sort tags and remove duplicates
tags = sorted(set(tags))

# Create training data
X_train = [] # bag of words for each pattern
y_train = [] # label for each tag

for (pattern_sentence, tag) in xy:
    # X: bag of words for each pattern
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    # y: PyTorch CrossEntropyLoss
    label = tags.index(tag)
    y_train.append(label) # CrossEntropyLoss

# Convert to numpy arrays
X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # Dataset[idx]
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    # len(Dataset)
    def __len__(self):
        return self.n_samples
    
# Hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()

# Data loader which takes the dataset, shuffles it, and creates batches
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #verify if GPU is available
model = NeuralNet(input_size, hidden_size, output_size).to(device) #push it to device if it's available

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

         # forward
        outputs = model(words)
        loss = criterion(outputs, labels)

        # backward and optimizer step
        optimizer.zero_grad() #empty the gradients first
        loss.backward() #calculate the gradients / the backpropagation
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'final loss: {loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

# Serialize it
FILE = "data.pth" # for pytorch

torch.save(data, FILE)

print(f'training complete. file savec to {FILE}')