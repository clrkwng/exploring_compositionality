import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, emb_dims, no_of_cont, lin_layer_sizes, output_size, emb_dropout, lin_layer_dropouts):
        super(FeedForward, self).__init__()

        self.emb_layers = nn.ModuleList([nn.Embedding(x,y) for x, y in emb_dims])
        n_of_embs = sum([y for x, y in emb_dims])
        self.n_of_embs = n_of_embs
        self.no_of_cont = no_of_cont

        first_lin_layer = nn.Linear(self.n_of_embs + self.no_of_cont, lin_layer_sizes[0])
        self.lin_layers = nn.ModuleList([first_lin_layer] + [nn.Linear(lin_layer_sizes[i], lin_layer_sizes[i+1]) for i in range(len(lin_layer_sizes) - 1)])

        # Comment this out for Xavier Initialization.
        for lin_layer in self.lin_layers:
            nn.init.kaiming_normal_(lin_layer.weight.data)

        self.output_layer = nn.Linear(lin_layer_sizes[-1], output_size)
        nn.init.kaiming_normal_(self.output_layer.weight.data)

        self.first_bn_layer = nn.BatchNorm1d(self.no_of_cont)
        self.bn_layers = nn.ModuleList([nn.BatchNorm1d(size) for size in lin_layer_sizes])

        self.emb_dropout_layer = nn.Dropout(emb_dropout)
        self.dropout_layers = nn.ModuleList([nn.Dropout(p) for p in lin_layer_dropouts])

    def forward(self, cont_data, cat_data):
        if self.n_of_embs != 0:
            x = [emb_layer(cat_data[:, i].long()) for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, 1)
            x = self.emb_dropout_layer(x)

        if self.no_of_cont != 0:
            normalized_cont_data = self.first_bn_layer(cont_data)

        if self.n_of_embs != 0:
            x = torch.cat([x, normalized_cont_data], 1)
        else:
            x = normalized_cont_data
        
        for lin_layer, dropout_layer, bn_layer in zip(self.lin_layers, self.dropout_layers, self.bn_layers):
            x = F.relu(lin_layer(x))
            x = bn_layer(x)
            x = dropout_layer(x)

        x = self.output_layer(x)
        return x

        