import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import csv
from torch.autograd import Variable
from sklearn.metrics import f1_score
import numpy as np
torch.manual_seed(1)

training_data = []
with open('training.csv','rb') as f:
    reader = csv.reader(f)
    for row in reader:
        training_data.append((row[0].split(),row[1]))


word_to_idx = {}
word_to_idx['<PAD>'] = len(word_to_idx)
label_to_idx = {}
for context, label in training_data:
    for word in context:
        if word not in word_to_idx:
            word_to_idx[word] = len(word_to_idx)
    if label not in label_to_idx:
        label_to_idx[label] = len(label_to_idx)
word_to_idx[''] = len(word_to_idx)

class CNNClassifier(nn.Module):
    def __init__(self, n_word, n_dim, n_class, kernel_num, kernel_sizes):
        super(CNNClassifier, self).__init__()
        self.word_embedding = nn.Embedding(n_word, n_dim)
        self.convs1 = [nn.Conv2d(1, kernel_num, (K, n_dim)) for K in kernel_sizes]
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(kernel_sizes)*kernel_num, n_class)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = self.word_embedding(x)
        x = x.unsqueeze(1)
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x,1)
        x = self.dropout(x)
        logit = self.fc1(x)
        return logit

model = CNNClassifier(len(word_to_idx), 200, len(label_to_idx),200, [3,4,5])
print model
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def make_sequence(x, dic):
    idx = [dic[i] for i in x]
    idx = Variable(torch.LongTensor(idx))
    return idx

for epoch in range(100):
    print('*' * 10)
    print('epoch {}'.format(epoch+1))
    running_loss = 0
    training_data1 = training_data[:800]
    test_data = training_data[800:]
    outs = ''
    labels = ''
    batch_size = 10
    for i in range(len(training_data1)//20):
        labels = []
        idx = 0
        for data in training_data1[i*20:(i+1)*20]:
            word, label = data
            if len(word) == 0:
                word = ['']
            word =  word + ['<PAD>']*4
            if idx == 0:
                word_list = make_sequence(word, word_to_idx)
                word_list = word_list.unsqueeze(0)
                out = model(word_list, word)
            else:
                word_list = make_sequence(word, word_to_idx)
                word_list = word_list.unsqueeze(0)
                out = torch.cat([out,model(word_list, word)],0)
            labels.append(label)
            idx += 1
        label = make_sequence(labels, label_to_idx)
        loss = criterion(out, label)
        running_loss += loss.data[0]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print 'batch: {:.1f}  loss: {:.4f}\r'.format(i,loss.data[0])
    print('Loss: {}'.format(running_loss / len(data)))


    corrects, avg_loss = 0, 0
    y_true = np.array([])
    y_pred = np.array([])
    for data in test_data:
        word, label = data
        if len(word) == 0:
            word = ['']
        word =  word + ['<PAD>']*4
        word_list = make_sequence(word, word_to_idx)
        word_list = word_list.unsqueeze(0)
        label = Variable(torch.LongTensor([label_to_idx[label]]))
        out = model(word_list, word)
        loss = criterion(out, label)
        avg_loss += loss.data[0]
        y_true = np.concatenate((y_true,label.data.numpy()),axis=0)
        y_pred = np.concatenate((y_pred,torch.max(out,1)[1].view(label.size()).data.numpy()),axis=0)
        corrects += (torch.max(out, 1)[1].view(label.size()).data == label.data).sum()
    size = len(test_data)
    avg_loss = loss.data[0]/size
    accuracy = corrects*100.0/size
    f1 = f1_score(y_true, y_pred, average='macro',labels=[0,2,3])
    print ('Macro F1: {}'.format(f1))
    print ('\nEvaluation - loss: {:.6f}  acc: {:.4f}%({}/{}) \n'.format(avg_loss,
                                                                       accuracy,
                                                                       corrects,
                                                                        size))


