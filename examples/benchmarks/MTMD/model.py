import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from utils import cal_cos_similarity

class MLP(nn.Module):

    def __init__(self, d_feat, hidden_size=512, num_layers=3, dropout=0.0):
        super().__init__()

        self.mlp = nn.Sequential()

        for i in range(num_layers):
            if i > 0:
                self.mlp.add_module('drop_%d'%i, nn.Dropout(dropout))
            self.mlp.add_module('fc_%d'%i, nn.Linear(
                360 if i == 0 else hidden_size, hidden_size))
            self.mlp.add_module('relu_%d'%i, nn.ReLU())

        self.mlp.add_module('fc_out', nn.Linear(hidden_size, 1))

    def forward(self, x):
        # feature
        # [N, F]
        return self.mlp(x).squeeze()

class HIST(nn.Module):
    def __init__(self, d_feat=6, hidden_size=64, num_layers=2, dropout=0.0, base_model="GRU", K =3):
        super().__init__()

        self.d_feat = d_feat
        self.hidden_size = hidden_size

        self.rnn = nn.GRU(
                input_size=d_feat,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout,
            )

        self.fc_ps = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps.weight)
        self.fc_hs = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs.weight)

        self.fc_ps_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps_fore.weight)
        self.fc_hs_fore = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs_fore.weight)

        self.fc_ps_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_ps_back.weight)
        self.fc_hs_back = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_hs_back.weight)
        self.fc_indi = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.xavier_uniform_(self.fc_indi.weight)

        self.leaky_relu = nn.LeakyReLU()
        self.softmax_s2t = torch.nn.Softmax(dim = 0)
        self.softmax_t2s = torch.nn.Softmax(dim = 1)
        
        self.fc_out_ps = nn.Linear(hidden_size, 1)
        self.fc_out_hs = nn.Linear(hidden_size, 1)
        self.fc_out_indi = nn.Linear(hidden_size, 1)
        self.fc_out = nn.Linear(hidden_size, 1)
        self.K = K
        
    def cal_cos_similarity(self, x, y): # the 2nd dimension of x and y are the same
        xy = x.mm(torch.t(y))
        x_norm = torch.sqrt(torch.sum(x*x, dim =1)).reshape(-1, 1)
        y_norm = torch.sqrt(torch.sum(y*y, dim =1)).reshape(-1, 1)
        cos_similarity = xy/x_norm.mm(torch.t(y_norm))
        cos_similarity[cos_similarity != cos_similarity] = 0
        return cos_similarity

    def forward(self, x, concept_matrix, market_value):
        device = torch.device(torch.get_device(x))
        x_hidden = x.reshape(len(x), self.d_feat, -1) # [N, F, T]      
        x_hidden = x_hidden.permute(0, 2, 1) # [N, T, F]
        x_hidden, _ = self.rnn(x_hidden)
        x_hidden = x_hidden[:, -1, :]

        # Predefined Concept Module
       
        market_value_matrix = market_value.reshape(market_value.shape[0], 1).repeat(1, concept_matrix.shape[1])
        stock_to_concept = concept_matrix * market_value_matrix
        
        stock_to_concept_sum = torch.sum(stock_to_concept, 0).reshape(1, -1).repeat(stock_to_concept.shape[0], 1)
        stock_to_concept_sum = stock_to_concept_sum.mul(concept_matrix)

        stock_to_concept_sum = stock_to_concept_sum + (torch.ones(stock_to_concept.shape[0], stock_to_concept.shape[1]).to(device))
        stock_to_concept = stock_to_concept / stock_to_concept_sum
        hidden = torch.t(stock_to_concept).mm(x_hidden)
        
        hidden = hidden[hidden.sum(1)!=0]
        stock_to_concept = x_hidden.mm(torch.t(hidden))
        # stock_to_concept = cal_cos_similarity(x_hidden, hidden)
        stock_to_concept = self.softmax_s2t(stock_to_concept)
        hidden = torch.t(stock_to_concept).mm(x_hidden)
        
        concept_to_stock = cal_cos_similarity(x_hidden, hidden) 
        concept_to_stock = self.softmax_t2s(concept_to_stock)

        p_shared_info = concept_to_stock.mm(hidden)
        p_shared_info = self.fc_ps(p_shared_info)

        p_shared_back = self.fc_ps_back(p_shared_info)
        output_ps = self.fc_ps_fore(p_shared_info)
        output_ps = self.leaky_relu(output_ps)

        pred_ps = self.fc_out_ps(output_ps).squeeze()
        
        # Hidden Concept Module
        h_shared_info = x_hidden - p_shared_back
        hidden = h_shared_info
        h_stock_to_concept = cal_cos_similarity(h_shared_info, hidden)

        dim = h_stock_to_concept.shape[0]
        diag = h_stock_to_concept.diagonal(0)
        h_stock_to_concept = h_stock_to_concept * (torch.ones(dim, dim) - torch.eye(dim)).to(device)
        # row = torch.linspace(0,dim-1,dim).to(device).long()
        # column = h_stock_to_concept.argmax(1)
        row = torch.linspace(0, dim-1, dim).reshape([-1, 1]).repeat(1, self.K).reshape(1, -1).long().to(device)
        column = torch.topk(h_stock_to_concept, self.K, dim = 1)[1].reshape(1, -1)
        mask = torch.zeros([h_stock_to_concept.shape[0], h_stock_to_concept.shape[1]], device = h_stock_to_concept.device)
        mask[row, column] = 1
        h_stock_to_concept = h_stock_to_concept * mask
        h_stock_to_concept = h_stock_to_concept + torch.diag_embed((h_stock_to_concept.sum(0)!=0).float()*diag)
        hidden = torch.t(h_shared_info).mm(h_stock_to_concept).t()
        hidden = hidden[hidden.sum(1)!=0]

        h_concept_to_stock = cal_cos_similarity(h_shared_info, hidden)
        h_concept_to_stock = self.softmax_t2s(h_concept_to_stock)
        h_shared_info = h_concept_to_stock.mm(hidden)
        h_shared_info = self.fc_hs(h_shared_info)

        h_shared_back = self.fc_hs_back(h_shared_info)
        output_hs = self.fc_hs_fore(h_shared_info)
        output_hs = self.leaky_relu(output_hs)
        pred_hs = self.fc_out_hs(output_hs).squeeze()

        # Individual Information Module
        individual_info  = x_hidden - p_shared_back - h_shared_back
        output_indi = individual_info
        output_indi = self.fc_indi(output_indi)
        output_indi = self.leaky_relu(output_indi)
        pred_indi = self.fc_out_indi(output_indi).squeeze()
        # Stock Trend Prediction
        all_info = output_ps + output_hs + output_indi
        pred_all = self.fc_out(all_info).squeeze()
        return pred_all
