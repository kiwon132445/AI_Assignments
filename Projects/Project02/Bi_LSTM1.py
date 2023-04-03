import spacy
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score 
from tqdm import tqdm

START_TAG = "<START>"
STOP_TAG = "<STOP>"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class Bi_LSTM(nn.Module):
    def __init__(self, rating_to_idx, embedding_dim, hidden_dim):
        super(Bi_LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.rating_to_idx = rating_to_idx
        self.rating_size = len(rating_to_idx)
        
#         nlp = spacy.load("en_core_web_lg")
#         embed_weights = torch.FloatTensor(nlp.vocab.vectors.data)
#         self.embed_layer = nn.Embedding.from_pretrained(embed_weights)
        
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, bidirectional=True)

        # Maps the output of the LSTM into rating space.
        self.hidden2rating = nn.Linear(hidden_dim, self.rating_size)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        # multiply by 4 to include other ratings
        self.transitions = nn.Parameter(
            torch.randn(self.rating_size*4, self.rating_size*4))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[rating_to_idx[START_TAG], :] = -10000
        self.transitions.data[:, rating_to_idx[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()
    
    # caution, might need fix
    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))
    
    #how do i handle multiple desired outputs?
    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.rating_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.rating_to_idx[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_rating in range(self.rating_size):
                # broadcast the emission score: it is the same regardless of
                # the previous rating
                emit_score = feat[next_rating].view(
                    1, -1).expand(1, self.rating_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_rating from i
                trans_score = self.transitions[next_rating].view(1, -1)
                # The ith entry of next_rating_var is the value for the
                # edge (i -> next_rating) before we do log-sum-exp
                next_rating_var = forward_var + trans_score + emit_score
                # The forward variable for this rating is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_rating_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.rating_to_idx[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha
    
    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2rating(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, ratings):
        # Gives the score of a provided rating sequence
        score = torch.zeros(1)
        rating = torch.cat([torch.tensor([self.rating_to_idx[START_TAG]], dtype=torch.long), ratings])
        for i, feat in enumerate(feats):
            score = score + \
                self.transitions[rating[i + 1], rating[i]] + feat[rating[i + 1]]
        score = score + self.transitions[self.rating_to_idx[STOP_TAG], rating[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.rating_size), -10000.)
        init_vvars[0][self.rating_to_idx[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_rating in range(self.rating_size):
                # next_rating_var[i] holds the viterbi variable for rating i at the
                # previous step, plus the score of transitioning
                # from rating i to rating.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_rating_var = forward_var + self.transitions[next_rating]
                best_rating_id = argmax(next_rating_var)
                bptrs_t.append(best_rating_id)
                viterbivars_t.append(next_rating_var[0][best_rating_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.rating_to_idx[STOP_TAG]]
        best_rating_id = argmax(terminal_var)
        path_score = terminal_var[0][best_rating_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_rating_id]
        for bptrs_t in reversed(backpointers):
            best_rating_id = bptrs_t[best_tag_id]
            best_path.append(best_rating_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.rating_to_idx[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path
    
    #Create 4 loss functions like such for stars, useful, funny, cool
    def neg_log_likelihood(self, sentence, ratings):
        feats = self._get_lstm_features(sentence)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, ratings)
        return forward_score - gold_score

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, rating_seq = self._viterbi_decode(lstm_feats)
        return score, rating_seq
    
class Bi_LSTM_Manager:
    # https://spacy.io/models/en#en_core_web_lg
    # embedding dimensionality is 300
    # HIDDEN_DIM := number samples/ alpha=(2<->10) * (input_layer_number=1 + output_layer_number=1) - embedding_layer?
    
    EMBEDDING_DIM = 300
    HIDDEN_DIM = 128 #adjustable
    EPOCH = 5
    
    rating_to_idx = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, START_TAG: 6, STOP_TAG: 7}

    def __init__(self):
        self.model = Bi_LSTM(self.rating_to_idx, self.EMBEDDING_DIM, self.HIDDEN_DIM)
        
    def tensor_vector(self, vector):
        return torch.tensor(vector, dtype=tensor.float)
    
    def train_model(self, vectored_dataset):
        #model = BiLSTM_CRF(len(word_to_ix), rating_to_idx, EMBEDDING_DIM, HIDDEN_DIM)
        optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

        # Make sure prepare_sequence from earlier in the LSTM section is loaded
        for epoch in range(self.EPOCH):  # again, normally you would NOT do 300 epochs, it is toy data
            i_tqdm = tqdm(range(1, len(vectored_dataset) + 1))
            i_tqdm.set_description('Loss: %' % (epoch))
            for sentence, ratings in training_data:
                # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is,
                # turn them into Tensors of word indices.
                sentence_in = prepare_sequence(sentence, word_to_ix)
                targets = torch.tensor([rating_to_idx[t] for t in ratings], dtype=torch.long)

                # Step 3. Run our forward pass.
                loss = model.neg_log_likelihood(sentence_in, targets)

                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling optimizer.step()
                loss.backward()
                optimizer.step()
                #Training Progress for this epoch
                i_tqdm.update(1)
            i_tqdm.close()
            
    def predict(self, x_dataset):
        return 0
    
    def score(self, y_pred, y):
        return 0