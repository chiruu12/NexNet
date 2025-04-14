import torch.nn as nn

class NER(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size, window_size, out_size):
        super(NER, self).__init__()

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.layer1 = nn.Linear(embed_size * (window_size * 2 +1), hidden_size)
        self.layer2 = nn.Linear(hidden_size, out_size) # predict the probability of each tag
        self.relu   = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        embeds = self.embed(inputs) # (batch_size, 5, emb_size)
        embeds = embeds.view(-1, embeds.size(1) * embeds.size(2)) # (batch_size, 5 * emb_size)
        h0 = self.dropout(self.relu(self.layer1(embeds)))
        out = self.layer2(h0)
        return out