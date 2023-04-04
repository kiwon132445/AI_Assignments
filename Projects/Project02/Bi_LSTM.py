import pandas as pd
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
START_TAG = "<START>"
STOP_TAG = "<STOP>"

class Bi_LSTM(nn.Module):
    def __init__(self, rating_categories, hidden_dim, embedding_dim, num_layers=1):
        super(Bi_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.rating_categories =rating_categories
        self.num_layers=num_layers
        self.dropout = nn.Dropout(0.1)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True, num_layers=self.num_layers)

        # Maps the output of the LSTM into rating space.
        self.hidden2ratings = [nn.Linear(
             hidden_dim, 1) for i in range(len(rating_categories))]

        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return (torch.randn(self.num_layers*2, 1, self.hidden_dim // 2),
                torch.randn(self.num_layers*2, 1, self.hidden_dim // 2))
    
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = sentence.view(1, 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(1, self.hidden_dim)
        return lstm_out
    
    def _get_linear_features(self, lstm_feats):
        lin_feats = []
        for i in range(len(self.hidden2ratings)):
            output = self.hidden2ratings[i](lstm_feats)
            lin_feats.append(output)
            
        lin_feats = torch.cat(lin_feats)
        return lin_feats
    
    def forward(self, input_vectors):
        output = self._get_lstm_features(input_vectors)
        outputs = self._get_linear_features(output)
        outputs = self.dropout(outputs)
        return outputs
    
class Bi_LSTM_Manager:
    # https://spacy.io/models/en#en_core_web_lg
    # embedding dimensionality is 300
    EMBEDDING_DIM = 300
    model = None
    
    rating_categories = {'stars':0, 'useful':1, 'funny':2, 'cool':3}
#     rating_range = [0, 1, 2, 3, 4, 5] #, START_TAG: 6, STOP_TAG: 7}
    
    def __init__(self, hidden_layers=512):
        self.HIDDEN_DIM = hidden_layers
        self.model = Bi_LSTM(self.rating_categories, self.HIDDEN_DIM, self.EMBEDDING_DIM)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # tensor manager
    def teseract(self, x_dataset, y_dataset, index):
        sentence = self.tensor_vector(x_dataset['vectors'].iloc[index])
        ratings = self.tensor_ratings(y_dataset, index)
        return sentence, ratings
    
    # tensors the text vectors
    def tensor_vector(self, vector):
        return torch.tensor(vector, dtype=torch.float).to(device)
    
    # tensors the rating values
    def tensor_ratings(self, dataset, index):
        ratings = []
        ratings.append(torch.tensor([dataset['stars'].iloc[index]], dtype=torch.float))
        ratings.append(torch.tensor([dataset['useful'].iloc[index]], dtype=torch.float))
        ratings.append(torch.tensor([dataset['funny'].iloc[index]], dtype=torch.float))
        ratings.append(torch.tensor([dataset['cool'].iloc[index]], dtype=torch.float))
#         ratings.append(self.fit_values(dataset['stars'].iloc[index]))
        newRatings = torch.cat(ratings).unsqueeze(-1)
        return newRatings.to(device)
    
#     def fit_values(self, value):
#         if value is None or value < 5:
#             return torch.tensor([0], dtype=torch.long)
#         elif value > 5:
#             return torch.tensor([5], dtype=torch.long)
#         else:
#             return torch.tensor([value], dtype=torch.long)
        
    def train_model(self, x_dataset, y_dataset, loss_function=0, optimizer=0, epoch=1):
        max_dataset_size = max(len(x_dataset), len(y_dataset))
        plot_every = 1 if max_dataset_size < 100 else int(max_dataset_size*0.01)
        learning_rate = 0.0005
        total_loss = 0
        all_losses = []
        if loss_function == 0:
            criterion = nn.MSELoss()
        else:
            criterion = nn.L1Loss()

        i_tqdm = tqdm(range(1, len(x_dataset)*epoch + 1))
        for ep in range(epoch):
            start = time.time()
            all_losses.append([])
            for row_idx in range(max(len(x_dataset), len(y_dataset))):
                self.optimizer.zero_grad()
                self.model.zero_grad()

                sentence_vector_tensor, rating_tensors = self.teseract(x_dataset, y_dataset, row_idx)
                output = self.model(sentence_vector_tensor)

                loss = 0
#                 for i in range(len(output)):
#                     l = criterion(output[i], rating_tensors[i])
#                     loss += l

                loss = criterion(output, rating_tensors.float())
                #print(loss)
                loss.backward()
                if optimizer == 0:
                    self.optimizer.step()
                else:
                    for p in self.model.parameters():
                        p.data.add_(p.grad.data, alpha=-learning_rate)

                output, loss = output, loss.item() #/ len(self.rating_categories)
                total_loss += loss

                i_tqdm.update(1)
                i_tqdm.set_description('Loss: %.4f' % (loss))
                if row_idx % plot_every == 0:
                    all_losses[ep].append(total_loss / plot_every)
                    total_loss = 0

            end = time.time()
            print("Epoch: %d" % (ep+1), " Complete\nEnd time:%d" % (end-start), "seconds")

        i_tqdm.close()
        return all_losses
    
    def predict(self, x_dataset):
        start = time.time()
        y_pred = pd.DataFrame()
        stars = []
        useful = []
        funny = []
        cool = []

        i_tqdm = tqdm(range(1, len(x_dataset) + 1))
        i_tqdm.set_description('BI_LSTM Prediction')

        for row_idx in range(len(x_dataset)):
            self.model.zero_grad()
            sentence_vector_tensor = self.tensor_vector(x_dataset['vectors'].iloc[row_idx])
            pred = self.model(sentence_vector_tensor).detach().numpy()

            stars.append(round(pred[0][0]))
            useful.append(round(pred[1][0]))
            funny.append(round(pred[2][0]))
            cool.append(round(pred[3][0]))

            i_tqdm.update(1)

        y_pred['stars'] = stars
        y_pred['useful'] = useful
        y_pred['funny'] = funny
        y_pred['cool'] = cool

        end = time.time()
        print("End time:%d" % (time.time()-start), "seconds")
        i_tqdm.close()
        return y_pred
            
    def f1_score(self, y, y_pred, average='macro'):
        f1_scores = {}
        for key in self.rating_categories.keys():
            f1_scores[key] = f1_score(y[key], y_pred[key], average=average)
        return f1_scores
    
    def classification_report(self, y, y_pred):
        for key in self.rating_categories.keys():
            print("Classification Report: ", key)
            print(classification_report(y[key], y_pred[key], zero_division=1))
        
    def plot_loss_graph(self, losses):
        plt.figure()
        plt.plot(losses)