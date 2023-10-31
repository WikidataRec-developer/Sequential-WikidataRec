#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

from datetime import datetime, timedelta



editors = pd.read_csv("editors.csv").sort_values("timestamp", ascending=False)#[["user_id", "item_id"]] ## sort from recent to old
editors["timestamp"] = pd.to_datetime(editors["timestamp"])

items = pd.read_csv("items.csv")

print(editors.head(10))


def remove_unicode_char(x):
    return not "Unicode" in x

items = items[items.item_sentence.apply(lambda x: remove_unicode_char(x))].reset_index(drop=True)

items.sample(10)


print(len(set(editors.item_id)))
print(len(set(editors.user_id)))



non_exist_items = set(editors.item_id) - set(items.item_id)

editors = editors[~editors.item_id.isin(non_exist_items)].reset_index(drop=True) ## To make the Num of items in the editors file same as in the items file



## To make a label_encoder for the editor_id and item_id
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

users_encoder = LabelEncoder()
items_encoder = LabelEncoder()


users_encoder.fit(editors.user_id)
items_encoder.fit(items.item_id)


editors["user_encoded"] = users_encoder.transform(editors.user_id)
editors["item_encoded"] = items_encoder.transform(editors.item_id)

items["item_encoded"] = items_encoder.transform(items.item_id)




## To filter the editors and keep only the active ones
np.random.seed(1)

# 1- Get number of item edited by every user
item_count= editors.groupby(['user_id'])['item_id'].nunique()

count=0
active_users=[] #to store only users who have edited more than 4 items
for i in item_count:
    if i >= 10:
        active_users.append(item_count[item_count>=10].index[count])
        #print(cat_count[cat_count>3].index[count])
        count+=1

# 2- Remove if < 10
filtered_editors = editors[editors.user_id.isin(active_users)]

print('shape of original data: ', editors.shape)
print('shape of data after dropping inactive users: ', filtered_editors.shape)

n_all_users = editors.user_id.unique().shape[0]
n_active_users = filtered_editors.user_id.unique().shape[0]
print('Number of all users are: ')
print(n_all_users)
print('Number of active users are: ')
print(n_active_users)

#filtered_editors = np.random.choice(filtered_editors.user_id.unique().tolist(), size=1000) ## to only take subset of the filtered editors
#filtered_editors = editors[editors.user_id.isin(selected_users)]#[["user_encoded", "item_encoded"]]

print('# of editors after I discard the non-active editors')
print(filtered_editors.user_id.unique().shape[0])
print('# of items after I discard the non-active editors')
print(filtered_editors.item_id.unique().shape[0])
print('# of the interactions after I discard the non-active editors')
print(filtered_editors.shape)



## Because deep learning accept input and output row by row (as a supervised)
# Start with an empty DataFrame
import random
from numpy.random import randn
from numpy.random import seed

flattened_data = []

# Iterate over unique users
for user in filtered_editors['user_encoded'].unique()[:25]: # extract 25 user
    # Get all items this user has edited, preserving the order
    user_items = filtered_editors[filtered_editors['user_encoded'] == user]['item_encoded'].tolist()

    # Get all the unique items the user has not interacted with
    non_user_items = set(filtered_editors['item_encoded'].unique()) - set(user_items)

    # Sample only 200 non-interacted items or all if there are fewer than 200
    sampled_non_user_items = random.sample(non_user_items, min(200, len(non_user_items)))

    # Add rows for items the user has interacted with
    for item in user_items:
        flattened_data.append((user, item, 1))

    # Add rows for items the user has NOT interacted with
    for item in sampled_non_user_items:
        flattened_data.append((user, item, 0))

# Create the DataFrame
df_ = pd.DataFrame(flattened_data, columns=["users", "items", "labels"])

# Convert types
df_["users"] = df_["users"].astype(np.int32)
df_["items"] = df_["items"].astype(np.int32)
df_["labels"] = df_["labels"].astype(np.float32)

#df_.to_csv('C:\\Users\\ka1y20\\Dropbox\\PhD\\Technical\\sequential_model\\Alternative_dataset_3\\labelled_interaction_data.csv', index=False, header=True)

print('num of users in the labelled data')
print(df_.users.unique().shape[0])
print('num of items in the labelled data')
print(df_['items'].unique().shape[0])
print('# of the interactions in the labelled data')
print(df_.shape)



df_['items'].nunique() * df_['users'].nunique() - df_.shape[0]



##To get sentences of each item
item_sent_map = dict(zip(items.item_encoded, items.item_sentence))
def get_item_sentence(item):
    return item_sent_map[item]

df_["sentences"] = df_["items"].map(get_item_sentence)

##This is the comprehensive df that we need
print(df_)


def time_ordered_stratified(df_, col, random_state=1):
    """
    Creates a time-ordered stratified sequential split of a dataframe df.
    For label=1, the recent items are included in the sequential training set.
    For label=0, items are included randomly.
    """
    seq_dfs_label1 = []
    ratios = [1.0, 0.0]  # 4/5, 1/5

    for value in df_[col].unique():
        subset = df_[(df_[col] == value) & (df_['labels'] == 1)]
        seq_subset_len = int(ratios[0] * len(subset))

        seq_subset = subset.iloc[:seq_subset_len]
        seq_dfs_label1.append(seq_subset)

    seq_df_label0 = df_[df_['labels'] == 0].sample(frac=ratios[0], random_state=random_state)
    seq_dfs = seq_dfs_label1 + [seq_df_label0]

    seq_df = pd.concat(seq_dfs, ignore_index=True)
    return seq_df.reset_index(drop=True)

# Example usage:
seq_df = time_ordered_stratified(df_, 'users')
filtered_editors = filtered_editors[filtered_editors.user_encoded.isin(seq_df.users.unique())]




## Each local item is 1 based on the condition (which is the last interacted items up to K timespan)
#(Previously, the number of recent items and Now, the timestamp) AND each global item is 0 (All the items)

def add_local_mask(df, days=5):
    # Grouping by user and item and getting max timestamp
    all_dates = filtered_editors.groupby(["user_encoded", "item_encoded"]).max().timestamp

    # Mapping of each user to their starting date for the time window
    user_date_thresholds = {user: date - pd.DateOffset(days=days) for user, date in filtered_editors.groupby("user_encoded")["timestamp"].max().items()}

    # Initialize the date_mask
    date_mask = []

    # Iterate over rows in df_ to construct the mask
    for _, row in df.iterrows():
        if row['labels'] == 1:  # for actual interactions
            interaction_date = all_dates.get((row['users'], row['items']))
            threshold_date = user_date_thresholds[row['users']]
            date_mask.append(interaction_date > threshold_date)
        else:  # for non-interactions
            date_mask.append(False)  # or 0, depending on what you want the default value to be

    # Assigning date_mask to the DataFrame column
    df["local_mask"] = date_mask
    df["local_mask"] = df["local_mask"].astype(np.float32)

    # Remove rows where users have fewer than 3 local items
    users_local_counts = df.groupby("users")["local_mask"].sum()
    valid_users = users_local_counts[users_local_counts >= 3].index

    return df[df["users"].isin(valid_users)]

time_interval = 30  # in days
local_seq_df = add_local_mask(seq_df, time_interval)
local_seq_df = local_seq_df.drop_duplicates().reset_index(drop=True)




from sklearn.model_selection import StratifiedKFold
kfold = StratifiedKFold(n_splits=5, shuffle=False)

for fold , (trn_idx, val_idx) in enumerate(kfold.split(local_seq_df, local_seq_df.labels)):
    # make sure that all users represent it in the train and val sets.
    print(local_seq_df.iloc[trn_idx].users.nunique(), local_seq_df.iloc[val_idx].users.nunique())
    local_seq_df.loc[val_idx, "kfold"] = fold
    #break




runs_file = "./"
seq_df.to_csv(runs_file + "sequence_aware_data_users.csv", index=False)
local_seq_df.to_csv(runs_file + "sequence_aware_local_data_users.csv", index=False)






