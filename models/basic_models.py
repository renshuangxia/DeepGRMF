import torch
import torch.nn as nn

class Basic_MF_Model(nn.Module):
    """
     Base Model for Matrix Factorization
    """

    def __init__(self, n_cl, n_drug,n_factors=64):
        super(Basic_MF_Model, self).__init__()

        self.n_cl = n_cl
        self.n_drug = n_drug

        self.cl_factors = nn.Embedding(n_cl, n_factors, sparse=False)
        self.drug_factors = nn.Embedding(n_drug, n_factors, sparse=False)

        self.cl_biases = nn.Embedding(n_cl, 1, sparse=False)
        self.drug_biases = nn.Embedding(n_drug, 1, sparse=False)
        
        self.cl_full_indicies = torch.LongTensor([i for i in range(self.n_cl)])
        self.drug_full_indicies = torch.LongTensor([i for i in range(self.n_drug)])  
       
    def forward(self, cl_indicies, drug_indicies):
        cl_embedding = self.cl_factors(cl_indicies) + self.cl_biases(cl_indicies)
        drug_embedding = self.drug_factors(drug_indicies) + self.drug_biases(drug_indicies)

        preds = torch.matmul(cl_embedding, drug_embedding.T)
        if preds.get_device() >= 0:
            preds = (preds * torch.eye(cl_indicies.shape[0]).to(preds.get_device())).sum(dim=1)
        else:
            preds = (preds * torch.eye(cl_indicies.shape[0])).sum(dim=1)
         
        cls, drugs = self.get_embs(preds.get_device())
        return preds, cls, drugs

    # predict durg sensitivities for all drugs with an input cell line
    def predict_drug_sens(self, cl_emb):
        drug_embs = self.drug_factors(self.drug_full_indicies) + self.drug_biases(self.drug_full_indicies)
        cl_embs = cl_emb.repeat(self.n_drug)
        preds = torch.matmul(cl_embs, drug_embs.T)
        if preds.get_device() >= 0:
            preds = (preds * torch.eye(self.n_drug).to(preds.get_device())).sum(dim=1)
        else:
            preds = (preds * torch.eye(self.n_drug).sum(dim=1))
        return preds


    def get_embs(self, device=0):
        if device >= 0:
            self.cl_full_indicies = self.cl_full_indicies.to(device)
            self.drug_full_indicies = self.drug_full_indicies.to(device)
        cl_embedding = self.cl_factors(self.cl_full_indicies) + self.cl_biases(self.cl_full_indicies)
        drug_embedding = self.drug_factors(self.drug_full_indicies) + self.drug_biases(self.drug_full_indicies)
        return cl_embedding, drug_embedding


class BasicFeatureTransformer(nn.Module):
    """
     Base mlp Model to perform feature transformation
    """

    def __init__(self, in_feature, out_feature, h_layers=[64, 32], dropout=0.2):
        super(BasicFeatureTransformer, self).__init__()
        self.initial_dropout = nn.Dropout(p=0.2)
        self.layers = nn.ModuleList()
        for node_num in h_layers:
            self.layers.append(nn.Sequential(
                nn.Linear(in_feature, node_num),
                nn.Tanh(),
                nn.Dropout(p=dropout)
            ))
            in_feature = node_num

        self.linear_final = nn.Linear(in_feature, out_feature)

    def forward(self, x):
        x = self.initial_dropout(x)
        for layer in self.layers:
            x = layer(x) # transform dimension of input cell line features
        return self.linear_final(x)


class Drug_Path_FeatureTransformer(nn.Module):
    """
     mlp Model to perform drug feature transformation, train pathway embeddings and concatenate with drug features
    """

    def __init__(self, in_feature, out_feature, n_path, h_layers=[64, 32], dropout=0.2, path_factors=16):
        super(Drug_Path_FeatureTransformer, self).__init__()
        self.initial_dropout = nn.Dropout(p=0.2)
        self.layers = nn.ModuleList()
        self.path_factors = nn.Embedding(n_path, path_factors, sparse=False)
        self.path_biases = nn.Embedding(n_path, 1, sparse=False)

        in_feature += path_factors
        for node_num in h_layers:
            self.layers.append(nn.Sequential(
                nn.Linear(in_feature, node_num),
                nn.Tanh(),
                nn.Dropout(p=dropout)
            ))
            in_feature = node_num

        self.linear_final = nn.Linear(in_feature, out_feature)

    def forward(self, x, path_idx):
        path_embs = self.path_factors(path_idx) + self.path_biases(path_idx)
        x = self.initial_dropout(x)
        x = torch.cat((x, path_embs), dim=1)
        for layer in self.layers:
            x = layer(x) # transform dimension of input cell line features
        return self.linear_final(x)
