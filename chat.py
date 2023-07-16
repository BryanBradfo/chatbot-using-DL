import random
import json
import torch 

from model import NeuralNet
from nltk_utils import tokenize, stem, bag_of_words

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open("intents.json", "r") as f:
    intents = json.load(f)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval() # set model to evaluation mode

bot_name = "Inspiration Bot"
print("Puis-je vous aider? (tapez 'quitte' si vous voulez arrêter)")

while True:
    sentence = input('Moi: ')
    if sentence == "quitte" :
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0]) # reshape to fit the model (one sample)
    X = torch.from_numpy(X)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()] #predicted.item : class label

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: Je n'ai pas bien compris, pouvez-vous reformuler ?")