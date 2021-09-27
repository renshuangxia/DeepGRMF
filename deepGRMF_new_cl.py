'''
    Matrix factorization with embedding similarity for predicting drug sensitivity for new cell lines on known drugs
    @Author: Shuangxia Ren
'''

import copy
import math
import os, csv
import random

import torch.optim as optim
from sklearn.model_selection import KFold
from torch.optim import lr_scheduler
from tqdm import tqdm
from models.basic_models import Basic_MF_Model, BasicFeatureTransformer
from util.utils import *
import torch.nn as nn

# Set randome seeds
random.seed(1)
np.random.seed(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(1)

results_dir= 'results/new_cl' # place to store results
report_file = results_dir + '/Results.csv'

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with open(report_file, 'w', newline='') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(
        ['drug_id', 'drug_name', 'precision', 'recall', 'f1', 'AUROC', 'AUPR'])

'''
    Model loading and saving options
'''
train_mf = True #  Train MF model, if disabled, will load pretrained models from 'mf_model_path'
mf_model_path = 'saved_models/new_cl' # place to store trained models
if not os.path.exists(mf_model_path):
    os.makedirs(mf_model_path)


'''
  Hyperparameters
'''

early_stop = 5
n_split = 2

# For MF model
cl_alpha = 0.2 # alpha for cell line embedding similarity loss
drug_alpha = 0.2 # alpha for drug embedding similarity loss
num_epochs_mf = 1 # number of epochs for training for each fold
learning_rate = 0.001
n_factors = 64
mf_batch_size = 5096 # num of indicies per forward pass
top_n_cl = 10 # top n celllines used for calculate similarity loss
top_n_drug = 5 # top n drugs used for calculate similarity loss
sim_loss = True # whether or not to add similarity loss
lamb = 0 # for Frobenius norm


# Cell line feature transformer parameters
cl_batch_size = 10
cl_lr = 0.0001
cl_epochs = 10 # number of epochs for training cell line feature transformer model
cl_h_layers = [2000, 1500, 1024, 256] # hidden layers for cell line feature transform model
cl_dropout = 0.2 # dropout rate for cell line feature transformer model
weight_decay = 0.00001
# add value to drug similarity based on pathway so that drugs in the same pathway get higher value
# if set to 0, use embedding similarity directly
pathway_weight = 0.0


# Device info
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Use device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

'''
   Preparing Data
'''
cell_lines_df = pd.read_csv('data/TCGA_GDSC/gdsc_exprs.csv', index_col=0) # cell line features
labels_df = pd.read_csv('data/gdsc1_sensitivity_update.csv', index_col=0)  # current labels for drug sensitivity
drugs_df = pd.read_csv('data/path_embed_processed.csv', index_col=0) # drug embeddings

path2drug = {}
drug2path = {}
for i in range(drugs_df.shape[0]):
    drug_row = drugs_df.iloc[i, :]
    drug_id = drug_row.name
    pathway = drugs_df.loc[drug_id, 'pathway_name']
    if pathway in path2drug:
        path2drug[pathway].append(drug_id)
    else:
        path2drug[pathway] = [drug_id]
    drug2path[drug_id] = pathway

drugs_df = drugs_df.iloc[:, 2:]

results = {'drug':np.zeros((5, n_split)), 'cellline':np.zeros((5, n_split))} # stores results


# if results directory not exists, create one
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

if not os.path.exists(mf_model_path):
    os.makedirs(mf_model_path)


# add drug info to dict: drug id -> drug name
drugs = {}  # map drug id to drug name
drug_df = pd.read_csv('data/drug_info.csv', )
for i in drug_df.index:
    drugs[str(drug_df.iloc[i, 0])] = drug_df.iloc[i, 1]

labels_df = labels_df.sample(frac = 1) # shuffle cell line embeddings
drugs_df = drugs_df.sample(frac=1) # shuffle drug embeddings

# make sure column index type matches
drug_ids = drugs_df.index
labels_df.columns = labels_df.columns.astype(int)

cell_lines_df = cell_lines_df.loc[labels_df.index]
X = cell_lines_df.copy()
y = labels_df.loc[:, drug_ids.astype(int)]

total_probs = []
total_preds = []

drug_embs = torch.FloatTensor(drugs_df.values).to(device)

'''
   Preparing K-fold
'''
kf = KFold(n_splits=n_split, shuffle=False)  # We have shuffled the data before

X_trains, y_trains, X_tests, y_tests = [], [], [], []
for train_index, test_index in kf.split(X):  # split cell line embeddings and labels
    X_trains.append(X.iloc[train_index])
    y_trains.append(y.iloc[train_index])

    X_tests.append(X.iloc[test_index])
    y_tests.append(y.iloc[test_index])

'''
   Train and test
'''
for fold in range(len(X_trains)):
    print('              ******* CURRENT FOLD:', (fold + 1), '*******')
    X_train_df = X_trains[fold]
    X_test_df = X_tests[fold]
    y_train_df = y_trains[fold]
    y_test_df = y_tests[fold]

    '''
       Preparing similarity arrays
    '''
    cl_sim_array = get_embedding_sim_array(X_train_df, top_n_cl).to(device)
    if pathway_weight == 0:
        drug_sim_array = get_embedding_sim_array(drugs_df, top_n_drug).to(device)
    else:
        drug_sim_array = get_embedding_sim_array_with_group(drugs_df, top_n_drug, drug2path, path2drug, group_value=pathway_weight).to(device)

    '''
        Dataframe to torch tensor
    '''
    X_train = torch.FloatTensor(X_train_df.values).to(device)
    y_train = torch.FloatTensor(y_train_df.values).to(device)
    X_test = torch.FloatTensor(X_test_df.values).to(device)
    y_test = torch.FloatTensor(y_test_df.values).to(device)

    if train_mf:  # if trained MF models are available, don't bother to train them again
        print(" * Train MF model ")
        '''
           Initialize model
        '''

        mf_model = Basic_MF_Model(X_train.shape[0], drug_embs.shape[0], n_factors=n_factors)

        mf_model = mf_model.to(device)
        optimizer = optim.Adam(mf_model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
        criterion = nn.BCEWithLogitsLoss()

        y_train_indicies = torch.LongTensor(np.argwhere(~np.isnan(y_train.cpu().numpy()))).to(device)
        p = np.random.permutation(y_train_indicies.shape[0])  # shuffle for training
        y_train_indicies = y_train_indicies[p]

        '''
           Train matrix factorization model
        '''
        mf_model.train()
        num_epoch_no_gain = 0
        min_loss = float('inf')
        for epoch in range(num_epochs_mf): # training for mf model
            if num_epoch_no_gain > early_stop:
                break
            total_loss = 0.0
            num_batches = math.ceil(y_train_indicies.shape[0] / mf_batch_size)
            for batch_idx in range(num_batches):
                start_idx = batch_idx * mf_batch_size
                end_idx = min(start_idx + mf_batch_size, y_train_indicies.shape[0])
                rows = y_train_indicies[start_idx:end_idx, 0]
                cols = y_train_indicies[start_idx:end_idx, 1]
                targets = y_train[rows, cols]
                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    # Predict and calculate loss
                    preds, cl_embs, drug_embs = mf_model(rows, cols)
                    if sim_loss:
                        loss = similarity_loss(preds, targets, rows, cols,
                                               cl_embs=cl_embs, cl_sim_array=cl_sim_array, cl_alpha=cl_alpha, n_cl=top_n_cl,
                                               drug_embs=drug_embs, drug_sim_array=drug_sim_array, drug_alpha=drug_alpha,
                                               n_drug=top_n_drug, lamb=lamb)
                    else:
                        loss = criterion(preds, targets)
                    total_loss += loss
                    loss.backward()

                optimizer.step()
            if epoch % 50 == 0:
                print('epoch:', epoch, ' Total Loss:', total_loss)
            scheduler.step(total_loss)

            if total_loss < min_loss:
                best_model = copy.deepcopy(mf_model.state_dict())
                min_loss = total_loss
                num_epoch_no_gain = 0
            else:
                num_epoch_no_gain += 1

        mf_model.load_state_dict(best_model)
        torch.save(mf_model, mf_model_path + '/MF_' + str(fold) + '.pt')
    else:
        mf_model = torch.load(mf_model_path + '/MF_' + str(fold) + '.pt')

    mf_model.eval()
    cl_embs, drug_embs = mf_model.get_embs(device=X_train.get_device())
    mf_embs_dir = mf_model_path + '/MF_embeddings_' + str(fold)
    if not os.path.exists(mf_embs_dir):
        cl_embs_df = pd.DataFrame(cl_embs.detach().cpu().numpy(), index=X_train_df.index)
        drug_embs_df = pd.DataFrame(drug_embs.detach().cpu().numpy(), index=drugs_df.index)
        os.makedirs(mf_embs_dir)
        cl_embs_df.to_csv(mf_embs_dir + '/cl_embs.csv')
        drug_embs_df.to_csv(mf_embs_dir + '/drug_embs.csv')

    '''
        Train cell line embeddings transform model
    '''
    print(" * Train Cell Line Embedding Transformer ")
    cl_feat_transformer = BasicFeatureTransformer(X_train.shape[1], cl_embs.shape[1], h_layers=cl_h_layers, dropout=cl_dropout)
    cl_feat_transformer = cl_feat_transformer.to(device)
    optimizer = optim.Adam(cl_feat_transformer.parameters(), lr=cl_lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(X_train, cl_embs.detach())
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
        #print('Epoch:', i, ' Loss:', total_loss)
    cl_feat_transformer.load_state_dict(best_model)

    '''
       Test
    '''
    print('\n************ Start Testing ************')

    cl_feat_transformer.eval()
    y_labels = y.loc[X_test_df.index, :]
    num_batches = math.ceil(X_test.shape[0] / cl_batch_size)
    all_probs, all_preds = [], []
    for i in range(num_batches):
        start_idx = i * cl_batch_size
        end_idx = min(start_idx + cl_batch_size, y_labels.shape[0])
        cl_feats = cl_feat_transformer(X_test[start_idx:end_idx,:])
        cl_feats = cl_feats.unsqueeze(dim=1).repeat(1, drug_embs.shape[0], 1)
        probs = torch.matmul(cl_feats, drug_embs.T) # we use drug embeddings from MF model directly
        probs = (probs * torch.eye(drug_embs.shape[0]).to(device)).sum(dim=2)
        probs = torch.sigmoid(probs)
        probs = probs.detach().cpu().numpy()
        all_probs.append(probs)
        all_preds.append(np.round(probs))
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)

    total_preds.append(all_preds)
    total_probs.append(all_probs)

    all_probs = pd.DataFrame(all_probs, index=y_labels.index, columns=y_labels.columns)  # create df for slicing
    all_preds = pd.DataFrame(all_preds, index=y_labels.index, columns=y_labels.columns)  # create df for slicing


    drug_results, cl_results, flatten_results = calculate_scores(y_test_df, all_preds, all_probs)
    results['drug'][:, fold] = drug_results
    results['cellline'][:, fold] = cl_results
    print('==================================================')
    print('          New drugs with trained cell lines')
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

'''
    collect final results and store results in a
'''

for set in results:
    curr_res = results[set]
    results_file = results_dir + '/' + set + '.csv'
    res_array = results[set]
    res_df = pd.DataFrame(res_array,
                          index=['precision', 'recall', 'f1', 'AUROC', 'AUPR'],
                          columns=['Fold_' + str(i) for i in range(n_split)])
    res_array = res_array.mean(axis=1)
    print('\n==========================', set, '==========================')
    print('             *Precision:', res_array[0])
    print('                *Recall:', res_array[1])
    print('                    *F1:', res_array[2])
    print('                 *AUROC:', res_array[3])
    print('                  *AUPR:', res_array[4])
    print('==============================================================')
    res_df.to_csv(results_file)

total_probs = np.concatenate(total_probs, axis=0)
total_preds = np.concatenate(total_preds, axis=0)
total_probs = pd.DataFrame(total_probs, index=y.index, columns=y.columns)  # create df for slicing
total_preds = pd.DataFrame(total_preds, index=y.index, columns=y.columns)  # create df for slicing
drug_results, cl_results, flatten_results = calculate_scores(y, total_preds, total_probs)

'''
    Store final results (per drug)
'''
drugIds = y.columns.to_list()
for i in range(total_probs.shape[1]):
    drugId = drugIds[i]
    drugName = drugs[drugId] if drugId in drugs else ' '
    curr_probs = total_probs.iloc[:,i]
    curr_preds = total_preds.iloc[:,i]
    curr_labels = y.iloc[:,i]
    col_indicies = curr_labels.index[~curr_labels.apply(np.isnan)]
    curr_labels = curr_labels.loc[col_indicies]
    curr_preds = curr_preds[col_indicies]
    curr_probs = curr_probs[col_indicies]

    precision = precision_score(curr_labels, curr_preds)
    recall = recall_score(curr_labels, curr_preds)
    f1 = f1_score(curr_labels, curr_preds)
    auroc = roc_auc_score(curr_labels, curr_probs)
    aupr = average_precision_score(curr_labels, curr_probs)

    with open(report_file, 'a+', newline='') as outfile:
        csvwriter = csv.writer(outfile)
        csvwriter.writerow([drugId, drugName, precision, recall, f1, auroc, aupr])


print('======================= TOTOAL RESULTS ======================')
print('        ****************** Drug ******************')
print('             *Precision:', drug_results[0])
print('                *Recall:', drug_results[1])
print('                    *F1:', drug_results[2])
print('                 *AUROC:', drug_results[3])
print('                  *AUPR:', drug_results[4])
print('\n        ***************** Cell Line ****************')
print('             *Precision:', cl_results[0])
print('                *Recall:', cl_results[1])
print('                    *F1:', cl_results[2])
print('                 *AUROC:', cl_results[3])
print('                  *AUPR:', cl_results[4])
print('\n       ****************** Micro ****************')
print('             *Precision:', flatten_results[0])
print('                *Recall:', flatten_results[1])
print('                    *F1:', flatten_results[2])
print('                 *AUROC:', flatten_results[3])
print('                  *AUPR:', flatten_results[4])
print('==============================================================')
