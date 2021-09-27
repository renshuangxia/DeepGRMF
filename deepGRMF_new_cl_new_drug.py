'''
    Matrix factorization with embedding similarity for predicting drug sensitivity for new cell lines and new drugs
    Consists of three seperated models: 2 mlps for cell line and drug features, 1 for matrix factorization
    @Author: Shuangxia Ren
'''

import copy
import math
import os, csv
import random
import torch.nn as nn

import torch.optim as optim
from sklearn.model_selection import KFold
from torch.optim import lr_scheduler
from tqdm import tqdm

from models.basic_models import Basic_MF_Model, BasicFeatureTransformer, Drug_Path_FeatureTransformer
from util.utils import *

# Set randome seeds
random.seed(1)
np.random.seed(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
torch.manual_seed(1)

results_dir= 'results/new_cl_drug' # place to store results
report_file = results_dir + '/results.csv'

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
mf_model_path = 'saved_models/mlc_new_cl_drug' # place to store trained models
if not os.path.exists(mf_model_path):
    os.makedirs(mf_model_path)

'''
  Hyperparameters
'''

early_stop = 5
n_split = 3 # actual fold will be n_split * n_split

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

mf_weight_decay = 0.0
pathway_weight = 0.0

# Cell line feature transformer parameters
cl_batch_size = 25
cl_epochs = 1 # number of epochs for training cell line feature transformer model
cl_h_layers = [1024, 256] # hidden layers for cell line feature transform model
cl_dropout = 0.2 # dropout rate for cell line feature transformer model
cl_weight_decay = 0.0
cl_lr = 0.00001

# Drug feature transformer parameters
drug_batch_size = 10
drug_h_layers = [192, 128] # hidden layers for drug feature transform model
drug_dropout = 0.2 # dropout rate for drug feature transformer model
drug_epochs = 1 # number of epochs for training drug feature transformer model
drug_weight_decay = 0.0
drug_lr = 0.001
add_pathway_embs = True
pathway_factors = 16 # only used when adding pathway embeddings


# Device info
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

# make sure column index type matches

labels_df.columns = labels_df.columns.astype(int)

# add drug info to dict: drug id -> drug name
drugs = {}  # map drug id to drug name
drug_df = pd.read_csv('data/drug_info.csv', )
for i in drug_df.index:
    drugs[int(drug_df.iloc[i, 0])] = drug_df.iloc[i, 1]

labels_df = labels_df.sample(frac = 1) # shuffle cell line embeddings
drugs_df = drugs_df.sample(frac=1) # shuffle drug embeddings
drug_ids = drugs_df.index

cell_lines_df = cell_lines_df.loc[labels_df.index] # make sure cell line indicies match label indicies
X = cell_lines_df.copy()
y = labels_df.loc[:, drug_ids.astype(int)] # make sure label columns matche drug embedding indicies

drug_pathway_df = drugs_df.iloc[:, 0:2] # df contains drug pathway information
drugs_df = drugs_df.iloc[:, 2:] # drug embedding df

'''
   Preparing K-fold
'''
kf = KFold(n_splits=n_split, shuffle=False)  # We have shuffled the data before
drug_trains, drug_tests = [], []
for train_index, test_index in kf.split(drugs_df):  # split drug embeddings
    drug_trains.append(drugs_df.iloc[train_index])
    drug_tests.append(drugs_df.iloc[test_index])


total_preds, total_probs = [], []
fold_count = 0
'''
   Train and test
'''
for drug_fold in range(len(drug_trains)): # for each fold of drug, we train k fold for cell line features
    drug_train_df = drug_trains[drug_fold]
    drug_test_df = drug_tests[drug_fold]

    '''
      Preparing drug similarity array s
    '''
    # pathway and drug maps for creating similarity matrix
    curr_drug_pathway_df = drug_pathway_df.loc[drug_train_df.index]
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

    drugId2pathId = {drug_id : pathName2PathId[drug_pathway_df.at[drug_id, 'pathway_name']] for drug_id in drug_pathway_df.index.tolist()}

    if pathway_weight == 0:
        drug_sim_array = get_embedding_sim_array(drug_train_df, top_n_drug).to(device)
    else:
        drug_sim_array = get_embedding_sim_array_with_group(drug_train_df, top_n_drug, drug2path, path2drug,
                                                            group_value=pathway_weight).to(device)

    '''
        Split cell line features
    '''

    X_trains, y_trains, X_tests, y_tests = [], [], [], []

    for train_index, test_index in kf.split(X):  # split cell line embeddings and labels
        X_trains.append(X.iloc[train_index])
        curr_train_targets = y.iloc[train_index]
        y_trains.append(
            curr_train_targets.loc[:, drug_train_df.index])  # make sure y train doesn't contain test drug labels

        X_tests.append(X.iloc[test_index])
        curr_test_targets = y.iloc[test_index]
        y_tests.append(curr_test_targets.loc[:, drug_test_df.index])

    drug_fold_total_probs, drug_fold_total_preds = [], []

    for fold in range(len(X_trains)):
        fold_count += 1
        print('              ******* CURRENT FOLD:', (fold_count), '*******')

        X_train_df = X_trains[fold]
        X_test_df = X_tests[fold]
        y_train_df = y_trains[fold]
        y_test_df = y_tests[fold]

        '''
           Preparing cell line similarity array s
        '''
        cl_sim_array = get_embedding_sim_array(X_train_df, top_n_cl).to(device)

        '''
            Dataframe to torch tensor
        '''
        X_train = torch.FloatTensor(X_train_df.values).to(device) # Cell line features
        y_train = torch.FloatTensor(y_train_df.values).to(device) #
        X_test = torch.FloatTensor(X_test_df.values).to(device)
        y_test = torch.FloatTensor(y_test_df.values).to(device)
        drug_train = torch.FloatTensor(drug_train_df.values).to(device)
        drug_test = torch.FloatTensor(drug_test_df.values).to(device)

        if add_pathway_embs:
            drug_train_path_ids = [drugId2pathId[drug_id] for drug_id in drug_train_df.index.tolist()]
            drug_test_path_ids = [drugId2pathId[drug_id] for drug_id in drug_test_df.index.tolist()]

            drug_train_path_ids = torch.FloatTensor(drug_train_path_ids).unsqueeze(dim=1).to(device)
            drug_test_path_ids = torch.FloatTensor(drug_test_path_ids).unsqueeze(dim=1).to(device)

            drug_train = torch.cat((drug_train, drug_train_path_ids), dim=1)
            drug_test = torch.cat((drug_test, drug_test_path_ids), dim=1)

        if train_mf:  # if trained MF models are available, don't bother to train them again
            print(" * Train MF model ")
            '''
               Initialize model
            '''
            mf_model = Basic_MF_Model(X_train.shape[0], drug_train.shape[0], n_factors=n_factors)
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
                                                   n_drug=top_n_drug)
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
            torch.save(mf_model, mf_model_path + '/MF_' + str(fold_count) + '.pt')
        else:
            mf_model = torch.load(mf_model_path + '/MF_' + str(fold_count) + '.pt')

        mf_model.eval()
        cl_embs, drug_embs = mf_model.get_embs(device=X_train.get_device())

        '''
            Train cell line embeddings transform model
        '''
        print(" * Train Cell Line Embedding Transformer ")
        cl_feat_transformer = BasicFeatureTransformer(X_train.shape[1], cl_embs.shape[1], h_layers=cl_h_layers, dropout=cl_dropout)
        cl_feat_transformer = cl_feat_transformer.to(device)
        optimizer = optim.Adam(cl_feat_transformer.parameters(), lr=cl_lr, weight_decay=cl_weight_decay)
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
            if num_epoch_no_gain > 20:
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
            drug_feat_transformer = Drug_Path_FeatureTransformer((drug_train.shape[1] - 1), drug_embs.shape[1],
                                                                 len(path2drug), h_layers=drug_h_layers,
                                                                 dropout=drug_dropout, path_factors=pathway_factors)

        else:
            drug_feat_transformer = BasicFeatureTransformer(drug_train.shape[1], drug_embs.shape[1],
                                                            h_layers=drug_h_layers, dropout=drug_dropout)

        drug_feat_transformer = drug_feat_transformer.to(device)
        optimizer = optim.Adam(drug_feat_transformer.parameters(), lr=drug_lr, weight_decay=drug_weight_decay)
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

        dataset = torch.utils.data.TensorDataset(drug_train, drug_embs.detach())
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=drug_batch_size, shuffle=False) # already shuffled before

        num_epoch_no_gain = 0
        min_loss = float('inf')
        drug_feat_transformer.train()
        print('\nStart training drug feature tranformer:')
        for i in range(drug_epochs):
            total_loss = 0.0
            if num_epoch_no_gain > 20:
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
        print('\n************ Start Testing ************')

        cl_feat_transformer.eval()
        drug_feat_transformer.eval()

        '''
            Test for new cell lines with new drugs
        '''
        y_labels = y.loc[X_test_df.index, drug_test_df.index]
        cl_feats = cl_feat_transformer(X_test)  # get all trained cl features

        if add_pathway_embs:
            drug_feats = drug_feat_transformer(drug_test[:, :-1], drug_test[:, -1].long())
        else:
            drug_feats = drug_feat_transformer(drug_test)

        cl_feats = cl_feats.unsqueeze(dim=1).repeat(1, drug_test.shape[0], 1) # for matrix multiplication
        probs = torch.matmul(cl_feats, drug_feats.T)
        probs = (probs * torch.eye(drug_feats.shape[0]).to(device)).sum(dim=2)
        probs = torch.sigmoid(probs) # this is the final probabilities
        probs = probs.detach().cpu().numpy()
        preds = np.round(probs) # round probabilities to get predicted labels

        # append results for calculating final results
        drug_fold_total_preds.append(preds)
        drug_fold_total_probs.append(probs)

        # create dataframes for probabilities and predictions for calculating metric scores
        probs = pd.DataFrame(probs, index=y_labels.index, columns=y_labels.columns)  # create df for slicing
        preds = pd.DataFrame(preds, index=y_labels.index, columns=y_labels.columns)  # create df for slicing
        drug_results, cl_results, flatten_results = calculate_scores(y_labels, preds, probs)
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

    # concatenate results for all cell line folds for current fold of drugs
    drug_fold_total_probs = np.concatenate(drug_fold_total_probs, axis=0)
    drug_fold_total_preds = np.concatenate(drug_fold_total_preds, axis=0)

    # append results for current fold of drugs for final evaluation
    total_probs.append(drug_fold_total_probs)
    total_preds.append(drug_fold_total_preds)


'''
    Aggregate results obtained from each fold of drug features
'''
total_probs = np.concatenate(total_probs, axis=1)
total_preds = np.concatenate(total_preds, axis=1)
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

'''
    Pring final results
'''
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

total_probs.to_csv(results_dir + '/total_probs.csv')