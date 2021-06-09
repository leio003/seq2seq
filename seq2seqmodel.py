import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, emb):
        super(Encoder, self).__init__()
        if (emb != None):
            self.embedding = nn.Embedding.from_pretrained(emb)
            # requires_grad指定是否在训练过程中对词向量的权重进行微调
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, hidden_size, dropout, emb):
        super(Decoder, self).__init__()
        if(emb != None):
            self.embedding = nn.Embedding.from_pretrained(emb)
            # requires_grad指定是否在训练过程中对词向量的权重进行微调
            self.embedding.weight.requires_grad = True
        else:
            self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, y, hidden):
        embedding = self.embedding(y)
        output, hidden = self.gru(embedding, hidden)
        output = self.out(output)

        return output

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, y):
        encoder_out, hid = self.encoder(x)
        output = self.decoder(y=y, hidden=hid)
        return output

    def translate(self, x, y, max_length=50):
        encoder_out, hid = self.encoder(x)
        preds = []
        batch_size = x.shape[0]
        for i in range(max_length):
            output = self.decoder(y=y, hidden=hid)
            y = output.max(2)[1].view(batch_size, 1)
            preds.append(y)

        return torch.cat(preds, 1)
