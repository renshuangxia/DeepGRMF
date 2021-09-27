'''
    Matrix factorization with embedding similarity for predicting drug sensitivity for /new drugs
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

from models.basic_models import Basic_MF_Model, BasicFeatureTransformer, Drug_Path_FeatureTransformer
from util.utils import *
import torch.nn as nn

# Set randome seeds
random.seed(1)
np.random.seed(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(1)

results_dir= 'results/new_drug/tcga' # place to store results
report_file = results_dir + '/Results.csv' # csv file to save results

if not os.path.exists(results_dir):
    os.makedirs(results_dir)

with open(report_file, 'w', newline='') as file:
    csvwriter = csv.writer(file)
    csvwriter.writerow(
        ['drug_id', 'drug_name', 'precision', 'recall', 'f1', 'AUROC', 'AUPR'])

'''
    Model loading and saving options
'''
train_mf = True

mf_model_path = 'saved_models/new_drug/tcga' # place to store trained models
if not os.path.exists(mf_model_path):
    os.makedirs(mf_model_path)


'''
  Hyperparameters
'''

early_stop = 30 # for MF model
n_split = 3

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
pathway_weight = 0.5

# Drug feature transformer parameters
drug_batch_size = 10
drug_h_layers = [192, 128] # hidden layers for drug feature transform model
drug_dropout = 0.2 # dropout rate for drug feature transformer model
drug_epochs = 1 # number of epochs for training drug feature transformer model
drug_lr = 0.0002
weight_decay = 0.0000
add_pathway_embs = True
pathway_factors = 16 # only used when adding pathway embeddings

'''
    Device info
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Use device:', device)

'''
   Preparing Data
'''
cell_lines_df = pd.read_csv('data/TCGA_GDSC/gdsc_exprs.csv', index_col=0) # cell line features
labels_df = pd.read_csv('data/gdsc1_sensitivity_update.csv', index_col=0)  # current labels for drug sensitivity
drugs_df = pd.read_csv('data/path_embed_processed.csv', index_col=0) # drug embeddings

# if results directory not exists, create one
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

labels_df.columns = labels_df.columns.astype(int)

# add drug info to dict: drug id -> drug name
drugs = {}  # map drug id to drug name
drug_df = pd.read_csv('data/drug_info.csv', )
for i in drug_df.index:
    drugs[str(drug_df.iloc[i, 0])] = drug_df.iloc[i, 1]

'''
    Shuffle and make sure indicies are matching
'''
labels_df = labels_df.sample(frac = 1) # shuffle cell line embeddings
drugs_df = drugs_df.sample(frac=1) # shuffle drug embeddings

cell_lines_df = cell_lines_df.loc[labels_df.index] # make sure indicies of inputs and labels are matching
cell_line_embs = torch.FloatTensor(cell_lines_df.values ).to(device) # cell line embeddings
drug_ids = drugs_df.index
y = labels_df.loc[:, drug_ids.astype(int)]

drug_pathway_df = drugs_df.iloc[:, 0:2]
drugs_df = drugs_df.iloc[:, 2:]


'''
   Preparing K-fold
'''
kf = KFold(n_splits=n_split, shuffle=False)  # We have shuffled the data before
X_trains, y_trains, X_tests, y_tests = [], [], [], []
j = 0
for train_index, test_index in kf.split(drugs_df):  # split drug embeddings
    X_trains.append(drugs_df.iloc[train_index])
    y_trains.append(y.loc[:, X_trains[j].index])
    X_tests.append(drugs_df.iloc[test_index])
    y_tests.append(y.loc[:, X_tests[j].index])
    j+= 1


total_probs = []
total_preds = []
results = {'drug':np.zeros((5, n_split)), 'cellline':np.zeros((5, n_split))} # stores results

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
    cl_sim_array = get_embedding_sim_array(cell_lines_df, top_n_cl).to(device)

    curr_drug_pathway_df = drug_pathway_df.loc[X_train_df.index]

    # pathway and drug maps for creating similarity matrix
    path2drug = {}
    drug2path = {}
    drugId2pathId = {}  # dict {drugId -> pathway Id}
    pathName2PathId = {}  # dict {pathway Id -> pathway Name}
    for i in range(curr_drug_pathway_df.shape[0]):
        drug_row = curr_drug_pathway_df.iloc[i, :]
        drug_id = drug_row.name
        pathway = curr_drug_pathway_df.loc[drug_id, 'pathway_name']
        if not pathway in pathName2PathId:
            pathName2PathId[pathway] = len(pathName2PathId)
        if pathway in path2drug:
            path2drug[pathway].append(drug_id)
        else:
            path2drug[pathway] = [drug_id]
        drug2path[drug_id] = pathway
        drugId2pathId[drug_id] = pathName2PathId[pathway]

    drugId2pathId = {drug_id: pathName2PathId[drug_pathway_df.at[drug_id, 'pathway_name']] for drug_id in
                     drug_pathway_df.index.tolist()}

    if pathway_weight == 0:
        drug_sim_array = get_embedding_sim_array(X_train_df, top_n_drug).to(device)
    else:
        drug_sim_array = get_embedding_sim_array_with_group(X_train_df, top_n_drug, drug2path, path2drug,
                                                            group_value=pathway_weight).to(device)

    '''
        Dataframe to torch tensor
    '''
    X_train = torch.FloatTensor(X_train_df.values).to(device)
    y_train = torch.FloatTensor(y_train_df.values).to(device)
    X_test = torch.FloatTensor(X_test_df.values).to(device)
    y_test = torch.FloatTensor(y_test_df.values).to(device)

    if add_pathway_embs:
        drug_train_path_ids = [drugId2pathId[drug_id] for drug_id in X_train_df.index.tolist()]
        drug_test_path_ids = [drugId2pathId[drug_id] for drug_id in X_test_df.index.tolist()]

        drug_train_path_ids = torch.FloatTensor(drug_train_path_ids).unsqueeze(dim=1).to(device)
        drug_test_path_ids = torch.FloatTensor(drug_test_path_ids).unsqueeze(dim=1).to(device)

        X_train = torch.cat((X_train, drug_train_path_ids), dim=1)
        X_test = torch.cat((X_test, drug_test_path_ids), dim=1)

    if train_mf:  # if trained MF models are available, don't bother to train them again
        print(" * Train MF model ")
        '''
           Initialize model
        '''
        mf_model = Basic_MF_Model(cell_line_embs.shape[0], X_train.shape[0], n_factors=n_factors)
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

            if total_loss < min_loss:
                best_model = copy.deepcopy(mf_model.state_dict())
                min_loss = total_loss
                num_epoch_no_gain = 0
            else:
                num_epoch_no_gain += 1
            scheduler.step(total_loss)

        mf_model.load_state_dict(best_model)
        torch.save(mf_model, mf_model_path + '/MF_' + str(fold) + '.pt')
    else:
        mf_model = torch.load(mf_model_path + '/MF_' + str(fold) + '.pt')

    mf_model.eval()
    cl_embs, drug_embs = mf_model.get_embs(device=X_train.get_device())


    '''
        Train drug embeddings transform model
    '''
    print(" * Train Drug Embedding Transformer ")
    if add_pathway_embs:
        drug_feat_transformer = Drug_Path_FeatureTransformer((X_train.shape[1] - 1), drug_embs.shape[1],
                                                             len(path2drug), h_layers=drug_h_layers,
                                                             dropout=drug_dropout, path_factors=pathway_factors)

    else:
        drug_feat_transformer = BasicFeatureTransformer(X_train.shape[1], drug_embs.shape[1],
                                                        h_layers=drug_h_layers, dropout=drug_dropout)

    drug_feat_transformer = drug_feat_transformer.to(device)
    optimizer = optim.Adam(drug_feat_transformer.parameters(), lr=drug_lr, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    dataset = torch.utils.data.TensorDataset(X_train, drug_embs.detach())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=drug_batch_size, shuffle=False) # already shuffled before
    criterion = nn.MSELoss()

    num_epoch_no_gain = 0
    min_loss = float('inf')
    drug_feat_transformer.train()
    print('\nStart training drug feature tranformer:')
    for i in range(drug_epochs):
        total_loss = 0.0
        if num_epoch_no_gain > 5:
            print('Break at epoch:', i, ' min_loss:', min_loss)
            break
        for X_batch, y_batch in tqdm(train_loader, disable=True):
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
    print('\n************ Start Testing ************')

    drug_feat_transformer.eval()

    '''
        Test for current cell lines with new drugs
    '''
    y_labels = y.loc[cell_lines_df.index, X_train_df.index]
    cl_feats = cl_embs.unsqueeze(dim=1).repeat(1, X_test.shape[0], 1)

    if add_pathway_embs:
        drug_feats = drug_feat_transformer(X_test[:, :-1], X_test[:, -1].long())
    else:
        drug_feats = drug_feat_transformer(X_test)

    probs = torch.matmul(cl_feats, drug_feats.T)
    probs = (probs * torch.eye(drug_feats.shape[0]).to(device)).sum(dim=2)
    probs = torch.sigmoid(probs)
    probs = probs.detach().cpu().numpy()
    preds = np.round(probs)

    total_preds.append(preds)
    total_probs.append(probs)

    probs = pd.DataFrame(probs, index=y_test_df.index, columns=y_test_df.columns)  # create df for slicing
    preds = pd.DataFrame(preds, index=y_test_df.index, columns=y_test_df.columns)  # create df for slicing

    drug_results, cl_results, flatten_results = calculate_scores(y_test_df, preds, probs)
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

total_probs = np.concatenate(total_probs, axis=1)
total_preds = np.concatenate(total_preds, axis=1)
total_probs = pd.DataFrame(total_probs, index=y.index, columns=y.columns)  # create df for slicing
total_preds = pd.DataFrame(total_preds, index=y.index, columns=y.columns)  # create df for slicing
drug_results, cl_results, flatten_results = calculate_scores(y, total_preds, total_probs)

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
print('\n        ***************** Micro ****************')
print('             *Precision:', flatten_results[0])
print('                *Recall:', flatten_results[1])
print('                    *F1:', flatten_results[2])
print('                 *AUROC:', flatten_results[3])
print('                  *AUPR:', flatten_results[4])
print('==============================================================')