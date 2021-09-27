'''
    train DeepGRMF model on gdsc and test on ccle
    @Author: Shuangxia Ren
'''

import copy
import csv
import math
import os
import random

from tqdm import tqdm
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from models.basic_models import Basic_MF_Model, BasicFeatureTransformer, Drug_Path_FeatureTransformer
from util.utils import *

# Set randome seeds
random.seed(1)
np.random.seed(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(1)

train_mf = True

'''
    Hyperparameters
'''
cl_alpha = 0.2
drug_alpha = 0.2
pathway_weight = 0.5 # drugs work on the same pathway to a target drug get additional similarity value, increment by this value

num_epochs = 1
learning_rate = 0.001
n_factors = 64 # trained cl and drug embedding dimisions
batch_size = 5096 # num of indices per forward pass
top_n_cl = 10 # top n celllines used for calculate similarity loss
top_n_drug = 5 # top n drugs used for calculate similarity loss
model_save_dir = 'saved_models/ccle/'
early_stop = 30
sim_loss = True # whether or not to add similarity loss
lamb = 0 # for Frobenius norm
weight_decay= 0.0

# Cell line feature transformer parameters
cl_batch_size = 25
cl_epochs = 1 # number of epochs for training cell line feature transformer model
cl_h_layers = [2048, 512, 128] # hidden layers for cell line feature transform model
cl_dropout = 0.2 # dropout rate for cell line feature transformer model
cl_weight_decay = 0.0
cl_lr = 0.00004

# Drug feature transformer parameters
drug_batch_size = 10
drug_h_layers = [192, 128] # hidden layers for drug feature transform model
drug_dropout = 0.2 # dropout rate for drug feature transformer model
drug_epochs = 1 # number of epochs for training drug feature transformer model
drug_weight_decay = 0.0
drug_lr = 0.0002
add_pathway_embs = True
pathway_factors = 16 # only used when adding pathway embeddings

# Device info
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Use device:', device)

if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

'''
   Preparing Data
'''
# Read data files
results_file = 'results/results_ccle.csv'
cell_lines_df = pd.read_csv('data/TCGA_GDSC/gdsc_exprs.csv', index_col=0) # cell line features
labels_df = pd.read_csv('data/gdsc1_sensitivity_update.csv', index_col=0)  # current labels for drug sensitivity
drugs_df = pd.read_csv('data/path_embed_processed.csv', index_col=0) #  indices should match the columns in labels_df
drug_ids = drugs_df.index

ccle_cl_df = pd.read_csv('data/ccle/ccle_exprs.csv', index_col=0)
ccle_labels_df = pd.read_csv('data/ccle/CCLE_drug_response_binaryAUC.csv', index_col=0)
ccle_drugs_df = pd.read_csv('data/ccle/CCLE_drug_embedding.csv', index_col=0)


# make sure indices are matching for ccle data
ccle_joint_indices = ccle_labels_df.index.intersection(ccle_cl_df.index)
ccle_labels_df = ccle_labels_df.loc[ccle_joint_indices]
ccle_cl_df = ccle_cl_df.loc[ccle_joint_indices]

# create output file
with open(results_file, 'w', newline='') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(['n_factor', 'top_n_cl', 'top_n_drug', 'cl_alpha', 'drug_alpha',
                        'pathway_weight', 'precision', 'recall', 'f1', 'AUROC', 'AUPR'])

# get drug pathway information
path2drug = {} # dict {pathway -> drugs}
drug2path = {} # dict {drug -> pathway}
drugId2pathId = {} # dict {drugId -> pathway Id}
pathName2PathId = {} # dict {pathway Id -> pathway Name}

for i in range(drugs_df.shape[0]):
    drug_row = drugs_df.iloc[i, :]
    drug_id = drug_row.name
    pathway = drugs_df.loc[drug_id, 'pathway_name']
    if not pathway in pathName2PathId:
        pathName2PathId[pathway] = len(pathName2PathId)
    if pathway in path2drug:
        path2drug[pathway].append(drug_id)
    else:
        path2drug[pathway] = [drug_id]
    drug2path[drug_id] = pathway
    drugId2pathId[drug_id] = pathName2PathId[pathway]


'''
    Prepare inputs and labels
'''
labels_df.columns = labels_df.columns.astype(int)
labels_df = labels_df.sample(frac=1) # shuffle data
cell_lines_df = cell_lines_df.loc[labels_df.index]
drugs_df = drugs_df.sample(frac=1) # shuffle drug embeddings
drug_ids = drugs_df.index

cell_lines_df = cell_lines_df.loc[labels_df.index] # make sure cell line indicies match label indicies
X = cell_lines_df.copy()
y = labels_df.loc[:, drug_ids.astype(int)]

X = torch.FloatTensor(X.values).to(device)
y = torch.FloatTensor(y.values)

y_indices = torch.LongTensor(np.argwhere(~np.isnan(y.cpu().numpy()))).to(device)
y = y.to(device)
print("# of non-nan indices: ", y_indices.shape[0])

p = np.random.permutation(y_indices.shape[0]) # shuffle
y_indices = y_indices[p]

drug_pathway_df = drugs_df.iloc[:, 0:2] # df contains drug pathway information
drugs_df = drugs_df.iloc[:, 2:] # drug embedding df
drugs_feats = torch.FloatTensor(drugs_df.values).to(device)

if add_pathway_embs:
    path_ids = [drugId2pathId[drug_id] for drug_id in drugs_df.index.tolist()]
    path_ids = torch.FloatTensor(path_ids)
    path_ids = path_ids.unsqueeze(dim=1).to(device)
    drugs_feats = torch.cat((drugs_feats, path_ids), dim=1)

'''
    Prepare cell line similarity matrix
'''
cl_sim_mx = creat_w_matrix_features(cell_lines_df, top_n_cl).reset_index(drop=True) # get similarity matrix for cell lines
cl_sim_array = [] # the index will be the cell line index in original df

for i in range(cl_sim_mx.shape[0]):
    cl_sim_array.append(np.where(cl_sim_mx.iloc[i,:].to_numpy() == 1))

cl_sim_array = torch.LongTensor(np.array(cl_sim_array)).to(device)

'''
    Prepare drug similarity matrix
'''
if pathway_weight == 0:  # use basic drug similarity matrix
    drug_sim_array = get_embedding_sim_array(drugs_df, top_n_drug).to(device)
else:  # add pathway information
    drug_sim_array = get_embedding_sim_array_with_group(drugs_df, top_n_drug, drug2path, path2drug,
                                                    group_value=pathway_weight).to(device)

'''
    Add drug information
'''
# add drug info to dict: drug id -> drug name
drugs = {}  # map drug id to drug name
drug_df = pd.read_csv('data/drug_info.csv')
for i in drug_df.index:
    drugs[str(drug_df.iloc[i, 0])] = drug_df.iloc[i, 1]

'''
    Prepare CCLE data for testing
'''
known_drug_indices = ccle_drugs_df.index[ccle_drugs_df['New drug'] == False].tolist()  # get ccle known drug indices
known_drug_ids = ccle_drugs_df.loc[known_drug_indices, 'Drug_id'].tolist()
new_drug_indicies = ccle_drugs_df.index[ccle_drugs_df['New drug'] == True].tolist() # get ccle new drug indicies


# get new drug features
new_drug_feats = ccle_drugs_df.loc[new_drug_indicies, 'drug_embedding']
new_drug_feats = np.array([np.fromstring(new_drug_feats.iloc[i].replace('[','').replace(']',''), sep=' ') for i in range(new_drug_feats.shape[0])])
new_drug_feats = torch.FloatTensor(new_drug_feats).to(device)
ccle_known_drug_label_df = ccle_labels_df.loc[:, known_drug_indices] # new cell lines for known drugs
ccle_new_drug_label_df = ccle_labels_df.loc[:, new_drug_indicies] # new cell lines for new drugs

if add_pathway_embs:
    paths = ccle_drugs_df.loc[new_drug_indicies, 'pathway_name'].tolist()
    new_drug_paths = [pathName2PathId[path_name] for path_name in paths]
    new_drug_paths = torch.FloatTensor(new_drug_paths).unsqueeze(dim=1).to(device)
    new_drug_feats = torch.cat((new_drug_feats, new_drug_paths), dim=1)

# stores drug id
drugId2Idx = {}
for i, idx in enumerate(drugs_df.index.tolist()):
    drugId2Idx[idx] = i

'''
    Training 
'''
if train_mf:
    mf_model = Basic_MF_Model(y.shape[0], y.shape[1], n_factors=n_factors)
    mf_model = mf_model.to(device)

    optimizer = optim.Adam(mf_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = nn.BCEWithLogitsLoss()
    best_model = copy.deepcopy(mf_model.state_dict())

    num_epoch_no_gain = 0
    mf_model.train()
    min_loss = float('inf')

    for epoch in range(num_epochs):
        '''
            Train
        '''
        if num_epoch_no_gain > early_stop:
            break
        total_loss = 0.0
        num_batches = math.ceil(y_indices.shape[0] / batch_size)
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, y_indices.shape[0])
            rows = y_indices[start_idx:end_idx, 0]
            cols = y_indices[start_idx:end_idx, 1]

            targets = y[rows, cols]
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                # Predict and calculate loss
                preds, cl_embs, drug_embs = mf_model(rows, cols)
                if sim_loss: # use both BCE loss and similarity loss
                    loss = similarity_loss(preds, targets, rows, cols,
                                           cl_embs=cl_embs, cl_sim_array=cl_sim_array, cl_alpha=cl_alpha, n_cl=top_n_cl,
                                           drug_embs=drug_embs, drug_sim_array=drug_sim_array, drug_alpha=drug_alpha,
                                           n_drug=top_n_drug, lamb=lamb)
                else:
                    loss = criterion(preds, targets) # only use BCE loss
                total_loss += loss
                loss.backward()
            optimizer.step()
        if epoch % 50 == 0:
            print('epoch:', epoch, ' Total Loss:', total_loss)

        if total_loss < min_loss:
            best_model = copy.deepcopy(mf_model.state_dict())
            min_loss = total_loss
            num_epoch_no_gain = 0
        else:
            num_epoch_no_gain += 1
        scheduler.step(total_loss)

    mf_model.load_state_dict(best_model)
    torch.save(mf_model, model_save_dir + 'mf_model.pt')
else:
    mf_model = torch.load(model_save_dir + 'mf_model.pt')

mf_model.eval()
cl_embs, drug_embs = mf_model.get_embs(device=X.get_device())

'''
    Train cell line embeddings transform model
'''
print(" * Train Cell Line Embedding Transformer ")
cl_feat_transformer = BasicFeatureTransformer(X.shape[1], cl_embs.shape[1], h_layers=cl_h_layers, dropout=cl_dropout)
cl_feat_transformer = cl_feat_transformer.to(device)
optimizer = optim.Adam(cl_feat_transformer.parameters(), lr=cl_lr, weight_decay=cl_weight_decay)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
criterion = nn.MSELoss()

dataset = torch.utils.data.TensorDataset(X, cl_embs.detach())
train_loader = torch.utils.data.DataLoader(dataset, batch_size=cl_batch_size, shuffle=False)  # already shuffled before

num_epoch_no_gain = 0
min_loss = float('inf')
cl_feat_transformer.train()
print('\nStart training cellline feature tranformer:')
for i in range(cl_epochs):
    total_loss = 0.0
    if num_epoch_no_gain > 5:
        print('Break at epoch:', i, ' min_loss:', min_loss)
        break
    for X_batch, y_batch in tqdm(train_loader, disable=True):  # for each batch
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = cl_feat_transformer(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            curr_loss = loss.item()
            total_loss += curr_loss
    if total_loss < min_loss:
        best_model = copy.deepcopy(cl_feat_transformer.state_dict())
        min_loss = total_loss
        num_epoch_no_gain = 0
    else:
        num_epoch_no_gain += 1
    scheduler.step(total_loss)
cl_feat_transformer.load_state_dict(best_model)

'''
    Train drug embeddings transform model
'''
print(" * Train Drug Embedding Transformer ")
if add_pathway_embs:
    drug_feat_transformer = Drug_Path_FeatureTransformer((drugs_feats.shape[1] - 1), drug_embs.shape[1],
                                                          len(path2drug), h_layers=drug_h_layers,
                                                          dropout=drug_dropout, path_factors=pathway_factors)
else:
    drug_feat_transformer = BasicFeatureTransformer(drugs_feats.shape[1], drug_embs.shape[1],
                                                    h_layers=drug_h_layers, dropout=drug_dropout)

drug_feat_transformer = drug_feat_transformer.to(device)
optimizer = optim.Adam(drug_feat_transformer.parameters(), lr=drug_lr, weight_decay=drug_weight_decay)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

dataset = torch.utils.data.TensorDataset(drugs_feats, drug_embs.detach())
train_loader = torch.utils.data.DataLoader(dataset, batch_size=drug_batch_size, shuffle=False) # already shuffled before

num_epoch_no_gain = 0
min_loss = float('inf')
drug_feat_transformer.train()
print('\nStart training drug feature tranformer:')
for i in range(drug_epochs):
    total_loss = 0.0
    if num_epoch_no_gain > 5:
        print('Break at epoch:', i, ' min_loss:', min_loss)
        break
    for X_batch, y_batch in tqdm(train_loader, disable=True):  # for each batch
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            if add_pathway_embs:
                outputs = drug_feat_transformer(X_batch[:, :-1], X_batch[:, -1].long())
            else:
                outputs = drug_feat_transformer(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            curr_loss = loss.item()
            total_loss += curr_loss
    if total_loss < min_loss:
        best_model = copy.deepcopy(drug_feat_transformer.state_dict())
        min_loss = total_loss
        num_epoch_no_gain = 0
    else:
        num_epoch_no_gain += 1
    scheduler.step(total_loss)
drug_feat_transformer.load_state_dict(best_model)

'''
    Test
'''

mf_model.eval()
cl_feat_transformer.eval()
drug_feat_transformer.eval()


ccle_cl_embs = torch.FloatTensor(ccle_cl_df.values).to(device)
ccle_cl_embs = cl_feat_transformer(ccle_cl_embs) # get cell line features

'''
    Test new cell lines and known drugs
'''
known_drug_ids = [drugId2Idx[idx] for idx in known_drug_ids]
known_drug_embs = drug_embs[known_drug_ids, :]

cl_embs = ccle_cl_embs.unsqueeze(dim=1).repeat(1, known_drug_embs.shape[0], 1) # for matrix multiplication
probs = torch.matmul(cl_embs, known_drug_embs.T)
probs = (probs * torch.eye(known_drug_embs.shape[0]).to(device)).sum(dim=2)
probs = torch.sigmoid(probs) # this is the final probabilities
probs = probs.detach().cpu().numpy()
preds = np.round(probs) # round probabilities to get predicted labels

probs = pd.DataFrame(probs, index=ccle_known_drug_label_df.index, columns=ccle_known_drug_label_df.columns)  # create df for slicing
preds = pd.DataFrame(preds, index=ccle_known_drug_label_df.index, columns=ccle_known_drug_label_df.columns)  # create df for slicing
known_drug_results, known_drug_cl_results, known_drug_flatten_results = calculate_scores(ccle_known_drug_label_df, preds, probs)
print('known drug new cl out shape:', preds.shape)

'''
    Test new cell lines and new drugs
'''
if add_pathway_embs:
    new_drug_feats = drug_feat_transformer(new_drug_feats[:, :-1], new_drug_feats[:, -1].long())
else:
    new_drug_feats = drug_feat_transformer(new_drug_feats)
cl_embs = ccle_cl_embs.unsqueeze(dim=1).repeat(1, new_drug_feats.shape[0], 1) # for matrix multiplication
probs = torch.matmul(cl_embs, new_drug_feats.T)
probs = (probs * torch.eye(new_drug_feats.shape[0]).to(device)).sum(dim=2)
probs = torch.sigmoid(probs) # this is the final probabilities
probs = probs.detach().cpu().numpy()
preds = np.round(probs) # round probabilities to get predicted labels
probs = pd.DataFrame(probs, index=ccle_new_drug_label_df.index, columns=ccle_new_drug_label_df.columns)  # create df for slicing
preds = pd.DataFrame(preds, index=ccle_new_drug_label_df.index, columns=ccle_new_drug_label_df.columns)  # create df for slicing
drug_results, cl_results, flatten_results = calculate_scores(ccle_new_drug_label_df, preds, probs)

print('new drug new cl out shape:', preds.shape)

print('==================================================')
print('          New drugs with new cell lines')
print('    ----------- Drug Metricies -----------')
print('          *Precision:', drug_results[0])
print('             *Recall:', drug_results[1])
print('                 *F1:', drug_results[2])
print('              *AUROC:', drug_results[3])
print('               *AUPR:', drug_results[4])
print('')
print('    --------- Cell line Metricies ---------')
print('          *Precision:', cl_results[0])
print('             *Recall:', cl_results[1])
print('                 *F1:', cl_results[2])
print('              *AUROC:', cl_results[3])
print('               *AUPR:', cl_results[4])
print('\n ***************** Micro ****************')
print('             *Precision:', flatten_results[0])
print('                *Recall:', flatten_results[1])
print('                    *F1:', flatten_results[2])
print('                 *AUROC:', flatten_results[3])
print('                  *AUPR:', flatten_results[4])
print('==================================================')


print('==================================================')
print('          New cell lines with known drugs')
print('    ----------- Drug Metricies -----------')
print('          *Precision:', known_drug_results[0])
print('             *Recall:', known_drug_results[1])
print('                 *F1:', known_drug_results[2])
print('              *AUROC:', known_drug_results[3])
print('               *AUPR:', known_drug_results[4])
print('')
print('    --------- Cell line Metricies ---------')
print('          *Precision:', known_drug_cl_results[0])
print('             *Recall:', known_drug_cl_results[1])
print('                 *F1:', known_drug_cl_results[2])
print('              *AUROC:', known_drug_cl_results[3])
print('               *AUPR:', known_drug_cl_results[4])
print('\n ***************** Micro ****************')
print('             *Precision:', known_drug_flatten_results[0])
print('                *Recall:', known_drug_flatten_results[1])
print('                    *F1:', known_drug_flatten_results[2])
print('                 *AUROC:', known_drug_flatten_results[3])
print('                  *AUPR:', known_drug_flatten_results[4])
print('==================================================')
