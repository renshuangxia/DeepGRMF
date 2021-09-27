import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import pairwise_distances


# Get top n most similar items based on embeddings for every item, note end index is exclusive
def creat_w_matrix_features(df, n, group_df=None):
    euclidean_distance_matrix = pairwise_distances(df.values, metric='l1')
    euclidean_distance_matrix_mean = euclidean_distance_matrix.mean().mean()
    kernel = euclidean_distance_matrix/euclidean_distance_matrix_mean
    w_features = np.exp(-kernel)
    if group_df is not None:
        w_features += group_df.to_numpy()

    w_features *= np.ones(w_features.shape) - np.eye(w_features.shape[0])
    top_index = np.argpartition(w_features,np.argmin(w_features,axis=0))[:,-n:]

    w_features_new = np.zeros(w_features.shape)
    w_features_new[np.arange(w_features.shape[0])[:, None], top_index] = 1
    w_features_final = pd.DataFrame(w_features_new, index=df.index, columns=df.index)
    return w_features_final

# Get top n most similar items based on embeddings for every item,
def creat_w_matrix_features_for_test(df, n, test_idx_start=-1, group_df=None):
    euclidean_distance_matrix = pairwise_distances(df.values, metric='l1')
    euclidean_distance_matrix_mean = euclidean_distance_matrix.mean().mean()
    kernel = euclidean_distance_matrix/euclidean_distance_matrix_mean
    w_features = np.exp(-kernel)
    if group_df is not None:
        w_features += group_df.to_numpy()

    top_indicies = []
    for i in range(test_idx_start, df.shape[1]):
        curr_w_feature = w_features[i, 0:test_idx_start-1]
        print(curr_w_feature.shape)
        top_index = np.argmin(curr_w_feature)[-n:]
        top_indicies.append(top_index)

    top_indicies = np.array(top_indicies)
    w_features = w_features[test_idx_start:, 0:test_idx_start-1]
    w_features_new = np.zeros(w_features.shape)
    w_features_new[np.arange(w_features.shape[0])[:, None], top_indicies] = 1
    w_features_final = pd.DataFrame(w_features_new, index=df.index, columns=df.index)
    print('w_features_final', w_features_final)
    return w_features_final



# similarity loss for MF
def similarity_loss(probs, labels, rows, cols, cl_embs=None, cl_sim_array=None, n_cl=10,
                    drug_embs=None, drug_sim_array=None, cl_alpha=0.2, drug_alpha=0.2, n_drug=5, lamb=0):

    bce_loss = F.binary_cross_entropy_with_logits(probs, labels) # bce loss

    # Calculate cell line similarity loss
    cl_targets = rows.repeat(n_cl, 1).transpose(0 , 1).flatten() # add target indicies
    cl_nb_embs = cl_sim_array[rows, :].flatten() # add neighbour indicies
    cl_targets = cl_embs[cl_targets] # target embeddings
    cl_nb_embs = cl_embs[cl_nb_embs] # neighbour embeddings
    cl_loss = torch.mean((cl_targets - cl_nb_embs) ** 2)

    # Calculate drug similarity loss
    drug_targets = cols.repeat(n_drug, 1).transpose(0, 1).flatten()  # add target indicies
    drug_nb_embs = drug_sim_array[cols, :].flatten()  # add neighbour indicies
    drug_targets = drug_embs[drug_targets] # target embeddings
    drug_nb_embs = drug_embs[drug_nb_embs] # neighbour embeddings

    drug_loss = torch.mean((drug_targets - drug_nb_embs) ** 2)
    #print('bce: ', bce_loss, ' cl_loss:', cl_alpha * cl_loss, ' drug_loss:', drug_alpha * drug_loss)

    cl_norm = torch.norm(cl_embs)
    drug_norm = torch.norm(drug_embs)
    return  bce_loss + cl_alpha * cl_loss + drug_alpha * drug_loss + lamb * (cl_norm + drug_norm)



# get top n most similar indicies for target (drug/cellline).
# The indicies of targets are in the first dim of the output
def get_embedding_sim_array(df, top_n, end_idx = -1):
    if end_idx > -1:
        sim_mx = creat_w_matrix_features_for_test(df, top_n, test_idx_start=-1).reset_index(drop=True)  # get similarity matrix for cell lines
    else:
        sim_mx = creat_w_matrix_features(df, top_n).reset_index(drop=True)  # get similarity matrix for cell lines
    sim_array = []  # the index will be the cell line index in original df
    for i in range(sim_mx.shape[0]):
        sim_array.append(np.where(sim_mx.iloc[i, :].to_numpy() == 1))
    return torch.LongTensor(np.array(sim_array))


# get top n most similar indicies for target with group information
# e.g. give drug for the same pathway (in group_df = pathway_df) a value 0.5 by default, plus the embedding similarity and then use it for ranking
# group2id dict stores group name to the list of items, e.g. pathway -> drugs
# id2group dict stores id to group name e.g. drug -> pathway
def get_embedding_sim_array_with_group(df, top_n, id2group, group2id, group_value=0.5, end_idx = -1):
    group_df = pd.DataFrame(np.zeros((df.shape[0], df.shape[0])), index=df.index, columns=df.index) # initialization

    for i in range(df.shape[0]):
        id = df.iloc[i, :].name
        group = id2group[id]
        group_ls = [ p for p in group2id[group] if p != id]
        group_df.loc[id, group_ls] = group_value
        group_df.loc[id, id] = group_value # set for self

    if end_idx > -1:
        sim_mx = creat_w_matrix_features_for_test(df, top_n, test_idx_start=-1, group_df=group_df).reset_index(drop=True)  # get similarity matrix for cell lines
    else:
        sim_mx = creat_w_matrix_features(df, top_n, group_df=group_df).reset_index(drop=True)  # get similarity matrix for cell lines

    sim_array = []  # the index will be the cell line index in original df
    for i in range(sim_mx.shape[0]):
        sim_array.append(np.where(sim_mx.iloc[i, :].to_numpy() == 1))
    return torch.LongTensor(np.array(sim_array))


# Calculate scores, labels/preds/probs are all DF
def calculate_scores(labels, preds, probs):
    flatten_results = np.zeros(5)  # calculate metrics for flattened results
    total_preds, total_labels, total_probs = [], [], []

    drug_results = np.zeros((5, labels.shape[1])) # calculate metrics per drug
    valid_inidicies = []
    for i in range(labels.shape[1]):
        col_labels = labels.iloc[:, i]
        col_preds = preds.iloc[:, i]
        col_probs = probs.iloc[:, i]
        col_labels = col_labels.copy()
        col_indicies = col_labels.index[~col_labels.apply(np.isnan)]
        if len(col_indicies) == 0: # skip drug that has all celline label nan
            continue
        col_labels = col_labels.loc[col_indicies]
        col_preds = col_preds[col_indicies]
        col_probs = col_probs[col_indicies]
        col_labels_array = col_labels.to_numpy()
        if (col_labels_array == col_labels_array[0]).all(): # skip drug only have one label
            continue
        valid_inidicies.append(i)
        #print(i, ' labels:', col_labels)
        drug_results[0][i] = precision_score(col_labels, col_preds)
        drug_results[1][i] = recall_score(col_labels, col_preds)
        drug_results[2][i] = f1_score(col_labels, col_preds)
        drug_results[3][i] = roc_auc_score(col_labels, col_probs)
        drug_results[4][i] = average_precision_score(col_labels, col_probs)
    drug_results = drug_results[:, valid_inidicies].mean(axis=1)

    cl_results = np.zeros((5, labels.shape[0])) # calculate metrics per cell line
    valid_inidicies = []
    for i in range(labels.shape[0]):
        row_labels = labels.iloc[i,:]
        row_preds = preds.iloc[i, :]
        row_probs = probs.iloc[i, :]
        row_labels = row_labels.copy()
        row_indicies = row_labels.index[~row_labels.apply(np.isnan)]
        if len(row_indicies) == 0: # skip cell line that has all drug label nan
            continue
        row_labels = row_labels.loc[row_indicies]
        row_preds = row_preds[row_indicies]
        row_probs = row_probs[row_indicies]

        total_preds.append(row_preds.values)
        total_labels.append(row_labels.values)
        total_probs.append(row_probs.values)

        row_labels_array = row_labels.to_numpy()
        if (row_labels_array == row_labels_array[0]).all(): # skip cell line only have one label
            continue
        valid_inidicies.append(i)
        cl_results[0][i] = precision_score(row_labels, row_preds)
        cl_results[1][i] = recall_score(row_labels, row_preds)
        cl_results[2][i] = f1_score(row_labels, row_preds)
        cl_results[3][i] = roc_auc_score(row_labels, row_probs)
        cl_results[4][i] = average_precision_score(row_labels, row_probs)
    cl_results = cl_results[:, valid_inidicies].mean(axis=1)

    total_preds = np.concatenate(total_preds)
    total_labels = np.concatenate(total_labels)
    total_probs = np.concatenate(total_probs)

    flatten_results[0] = precision_score(total_labels, total_preds)
    flatten_results[1] = recall_score(total_labels, total_preds)
    flatten_results[2] = f1_score(total_labels, total_preds)
    flatten_results[3] = roc_auc_score(total_labels, total_probs)
    flatten_results[4] = average_precision_score(total_labels, total_probs)
    return drug_results, cl_results, flatten_results

