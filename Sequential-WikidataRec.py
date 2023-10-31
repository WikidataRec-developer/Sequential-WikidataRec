#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

from datetime import datetime, timedelta

import tensorflow as tf
import tensorflow.keras.layers as ll

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Input, LayerNormalization, MultiHeadAttention, Dense

from transformers import BertTokenizer, TFBertModel
from transformers import logging

import os

# # Model
seq_df = pd.read_csv("sequence_aware_data_users.csv")
local_seq_df = pd.read_csv("sequence_aware_local_data_users.csv")





class TransformerEmbeddingLayer(ll.Layer):
    def __init__(self, model_name, emb_dim, **kwargs):
        super(TransformerEmbeddingLayer, self).__init__(**kwargs)
        self.emb_dim = emb_dim

        self.transformer = TFBertModel.from_pretrained(model_name) ## This is the pre-trained model
        self.fc = tf.keras.layers.Dense(units=emb_dim, activation="gelu")## fc: fully connected layer

        #self.transformer.trainable = False

    def call(self, input_ids, attention_mask=None):
        embeddings = self.transformer(input_ids, attention_mask=attention_mask)[0]
        embeddings = self.fc(embeddings)

        # Reshape the embeddings using K.reshape (to acceptable as graph)
        batch_size = K.shape(embeddings)[0]
        seq_length = K.shape(embeddings)[1]
        hidden_size = K.shape(embeddings)[2]
        embeddings = K.reshape(embeddings, (batch_size, seq_length, hidden_size))

        return embeddings

    def compute_output_shape(self, input_shape):
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        return (batch_size, sequence_length, self.emb_dim)




class KMeansLayer(ll.Layer):
    def __init__(self, n_clusters, **kwargs):
        self.n_clusters = n_clusters
        super(KMeansLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.centroids = self.add_weight(name='centroids',
                                         shape=(self.n_clusters, input_shape[-1]),
                                         initializer='uniform',
                                         trainable=False)

        super(KMeansLayer, self).build(input_shape)

    def call(self, x):
        # reduce input to match emb. dim.
        x = K.mean(x, axis=1)
        # Compute distances between each input point and each cluster centroid
        expanded_vectors = K.expand_dims(x, axis=1)
        expanded_centroids = K.expand_dims(self.centroids, axis=0)
        distances = K.sum(K.square(expanded_vectors - expanded_centroids), axis=-1)

        # Assign each input point to the closest centroid
        cluster_assignments = K.argmin(distances, axis=-1)

        # Update cluster centroids based on the mean of the assigned points
        new_centroids = []
        for i in range(self.n_clusters):
            assigned_points = tf.gather(x, tf.where(K.equal(cluster_assignments, i))[:, 0])
            new_centroid = K.mean(assigned_points, axis=0)
            new_centroids.append(new_centroid)
        self.centroids.assign(tf.stack(new_centroids))

        cluster_emb = tf.gather(self.centroids, cluster_assignments)

        return cluster_emb

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1])


# In[14]:


class ItemEncoder(ll.Layer):
    def __init__(self,
                 emb_dim,
                 model_name,
                 n_clusters,
                 item_input,
                 sentence_input,
                 local_mask_input,
                 item_input_dim,
                 vocab_size,
                 ):
        super(ItemEncoder, self).__init__()

        self.emb_dim = emb_dim ## It is a parameter: 384
        self.n_clusters = n_clusters ## It is a parameter: 14
        self.item_input = item_input ## It is all the items
        self.sentence_input = sentence_input ## It is the sentences of the items
        self.local_mask_input = local_mask_input ## It is the local items
        self.item_input_dim = item_input_dim ## Number of items + 1
        self.vocab_size = vocab_size ## It is the default vocabulary size: 30522

        self.item_embedding = ll.Embedding(input_dim=item_input_dim,
                                           output_dim=emb_dim,
                                           input_length=item_input.shape)

        self.sent_transformer_embedding = TransformerEmbeddingLayer(model_name=model_name, emb_dim=emb_dim)
        self.sent_transformer_embedding.trainable = False ## This is because the Bert is pre-trained model, so, we applied the transfer learning

        self.cluster = KMeansLayer(n_clusters)

        #self.flatten = ll.Flatten()
        #self.sent_embedding2 = ll.Embedding(input_dim=n_clusters+1, output_dim=emb_dim)

        self.item_softgate = ll.Lambda(lambda x: K.softmax(x, axis=-1))
        self.sent_softgate = ll.Lambda(lambda x: K.softmax(x, axis=-1))

        self.item_attention = ll.Multiply()
        self.sent_attention = ll.Multiply()

        self.item_representaions = ll.Concatenate(axis=1)

        self.local_item_representaions = ll.Multiply()

    def call(self, item, sentence, local_mask=None):
        sentence_tokens, sentence_mask = sentence

        item_emb = self.item_embedding(item) ## item ids embeddings

        sent_emb = self.sent_transformer_embedding(sentence, sentence_mask) ## item sentences transformer embeddings
        topic_emb = self.cluster(sent_emb)

        ##The input of the k-means
        #print('The input of the k-means')
        #print(topic_emb)

        item_soft = self.item_softgate(item_emb)
        topic_soft = self.sent_softgate(topic_emb)

        item_emb = self.item_attention([item_emb, item_soft])
        topic_emb = self.sent_attention([topic_emb, topic_soft])

        item_representaion = self.item_representaions([item_emb, topic_emb])

        if local_mask is not None:
            local_item_representaion = self.local_item_representaions([item_representaion, local_mask])
            return item_representaion, local_item_representaion

        return item_representaion


# In[16]:


class TransformerEncoderLayer(ll.Layer):
    def __init__(self, num_layers, sequence_length, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()  # Initialize the parent class (Layer).

        # Store the provided hyperparameters as instance variables.
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate

        # Initialize the multi-head attention mechanism.
        self.multihead_attention = MultiHeadAttention(
            key_dim=d_model, num_heads=num_heads
        )

        # Initialize the dropout layer for the first sub-layer.
        self.dropout1 = tf.keras.layers.Dropout(rate)

        # Initialize the layer normalization for the first sub-layer.
        self.layer_norm1 = LayerNormalization(epsilon=1e-6)

        # Initialize the feed-forward neural network (FFN) layers.
        self.dense_ffn = [
            Dense(dff, activation="relu"),  # First dense layer with ReLU activation.
            Dense(d_model),  # Second dense layer.
        ]

        # Initialize the dropout layer for the second sub-layer.
        self.dropout2 = tf.keras.layers.Dropout(rate)

        # Initialize the layer normalization for the second sub-layer.
        self.layer_norm2 = LayerNormalization(epsilon=1e-6)

    def positional_encoding(self, sequence_length, d_model):
        # Calculate positional encodings as a tensor and return it.
        pos_enc = np.array([
            [pos / np.power(10000, 2 * (i // 2) / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(sequence_length)
        ])
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # Apply sine function for even indices
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # Apply cosine function for odd indices
        return tf.convert_to_tensor(pos_enc, dtype=tf.float32)

    def transformer_encoder(self, inputs):
        # Calculate positional encodings and add them to the input.
        pos_enc = self.positional_encoding(self.sequence_length, self.d_model)
        inputs += pos_enc

        # Apply multi-head attention to the input.
        attn_output = self.multihead_attention(
            query=inputs, value=inputs, key=inputs
        )

        # Apply dropout to the attention output.
        attn_output = self.dropout1(attn_output)

        # Apply layer normalization to the sum of input and attention output.
        out1 = self.layer_norm1(inputs + attn_output)

        # Initialize the feed-forward neural network (FFN) output as the input.
        ffn_output = inputs

        # Apply the dense layers in the FFN.
        for layer in self.dense_ffn:
            ffn_output = layer(ffn_output)

        # Apply dropout to the FFN output.
        ffn_output = self.dropout2(ffn_output)

        # Apply layer normalization to the sum of the first layer output and FFN output.
        out2 = self.layer_norm2(out1 + ffn_output)

        return out2

    def call(self, inputs):
        x = inputs

        # Apply the transformer_encoder function for the specified number of layers.
        for _ in range(self.num_layers):
            x = self.transformer_encoder(x)

        return x


# In[17]:


x = ll.Input(shape=(384, 192), dtype=np.float32)
xx = np.random.rand(10, 384, 192)

transformer = TransformerEncoderLayer(num_layers=4, sequence_length=384, d_model=192, num_heads=4, dff=512, rate=0.1)
o = transformer(x)
o.shape


# In[24]:


class EditorEncoder(ll.Layer):
    def __init__(self,
                 emb_dim,
                 editor_input,
                 local_item_representaion_input,
                 editor_input_dim,
                 item_input_dim,
                 num_heads,
                 # transformer
                 num_layers,
                 dff, # deep feed forward
                 dropout_rate
                 ):
        super(EditorEncoder, self).__init__()

        self.editor_input = editor_input ## It is all the editors
        self.local_item_representaion_input = local_item_representaion_input ## It is the local items belong to each editor (short-term interests)
        self.editor_input_dim = editor_input_dim ## Number of editors + 1
        self.num_heads = num_heads

        self.editor_embedding = ll.Embedding(input_dim=editor_input_dim,
                                             output_dim=emb_dim,
                                             input_length=editor_input.shape)

        #########################
        self.local_item_embedding = ll.Embedding(input_dim=item_input_dim,
                                                 output_dim=emb_dim,
                                                 input_length=local_item_representaion_input.shape)
        self.transformer = TransformerEncoderLayer(num_layers=num_layers,
                                                   sequence_length=emb_dim*2,
                                                   d_model=emb_dim,
                                                   num_heads=num_heads,
                                                   dff=dff,
                                                   rate=dropout_rate
                                                   )
        self.flatten = ll.Flatten()
        self.fc = ll.Dense(emb_dim)
        ############################

        self.user_representaions = ll.Concatenate(axis=1)

    def call(self, editor, local_representaion):
        editor_emb = self.editor_embedding(editor) ## editor ids embeddings

        local_items = self.local_item_embedding(local_representaion) ## editor's local items embeddings
        local_seq = self.transformer(local_items) ## editor's sequential embeddings (multi-head self-attention)
        local_vec = self.flatten(local_seq)
        local_vec = self.fc(local_vec)

        unified_editor_rep = self.user_representaions([editor_emb, local_vec])

        return unified_editor_rep


# In[32]:


def Sequence_aware(item_input, ## all the items in the dataset
                   sent_input,
                   editor_input,
                   local_mask_input,
                   # item encoder args
                   emb_dim,
                   transformer_embedding_model_name,
                   n_clusters,
                   item_input_dim,
                   vocab_size,
                   # editor encoder args
                   editor_input_dim,
                   num_heads,
                   num_layers,
                   dff, # deep feed forward
                   dropout_rate,
                   # fc args
                   sigmoid,
                   ):

    item_encoder = ItemEncoder(emb_dim=emb_dim,
                               model_name=transformer_embedding_model_name,
                               n_clusters=n_clusters,
                               item_input=item_input,
                               sentence_input=sent_input,
                               local_mask_input=local_mask_input,
                               item_input_dim=item_input_dim,
                               vocab_size=vocab_size,
                               )

    item_representaion, local_item_representaion = item_encoder(item_input,
                                                                sent_input,
                                                                local_mask_input)

    editor_encoder = EditorEncoder(emb_dim=emb_dim,
                                   editor_input=editor_input,
                                   local_item_representaion_input=local_item_representaion,
                                   editor_input_dim=editor_input_dim,
                                   item_input_dim=item_input_dim,
                                   num_heads=num_heads,
                                   num_layers=num_layers,
                                   dff=dff,
                                   dropout_rate=dropout_rate
                                   )

    unified_editor_rep = editor_encoder(editor_input, local_item_representaion)

    output = ll.Dot(axes=-1)([unified_editor_rep, item_representaion])

    if sigmoid:
        output = tf.keras.activations.sigmoid(output)

    model = tf.keras.Model(inputs=[item_input, sent_input, editor_input, local_mask_input],
                           outputs=output,
                           name="Sequence_aware")

    #model.layers[-3]._name = "Unified-Editor-Rep"
    #model.layers[-1]._name = "Element-wise-product"

    model.compile(loss="binary_crossentropy", optimizer="adam") ## If cross-enropy with "softmax", it will result in vanishing / predicted (range 0-1) - actual (square error positive)

    return model


# In[33]:


emb_dim = 384 // 2
transformer_embedding_model_name = 'nlpaueb/legal-bert-small-uncased' #'bert-base-uncased' #' #
n_clusters = 14
max_length = 64
item_input_dim = seq_df["items"].values.max() + 1
vocab_size = 30522 # https://stackoverflow.com/questions/73232413/why-was-berts-default-vocabulary-size-set-to-30522
editor_input_dim = seq_df["users"].values.max() + 1
num_heads = emb_dim // 32 # 6

num_layers = 4
dff = 512
dropout_rate = 0.1

epochs = 10 ##try with bigger
batch_size = 64 ##try with bigger: 128


# In[34]:


# Tokenize the text
tokenizer = BertTokenizer.from_pretrained(transformer_embedding_model_name)

sentence_tokens = tokenizer(seq_df["sentences"].tolist(),
                            truncation=True,
                            padding='max_length',
                            max_length=max_length,
                            return_tensors='np'
                            )

sent_ids = sentence_tokens.input_ids
sent_msk = sentence_tokens.attention_mask

sent_ids.shape, sent_msk.shape


# In[36]:


item_input = ll.Input(shape=seq_df["items"].values.shape[1:], dtype=seq_df["items"].dtype)
sent_input = [ll.Input(shape=(max_length, ), dtype=np.int32), ll.Input(shape=(max_length, ), dtype=np.int32)]
editor_input = ll.Input(shape=seq_df["users"].values.shape[1:], dtype=seq_df["users"].dtype)
local_mask_input = ll.Input(shape=seq_df["local_mask"].shape[1:], dtype=seq_df["local_mask"].dtype)


model = Sequence_aware(item_input, # inputs
                       sent_input,
                       editor_input,
                       local_mask_input,
                       emb_dim,
                       transformer_embedding_model_name,
                       n_clusters,
                       item_input_dim,
                       vocab_size,
                       editor_input_dim,
                       num_heads,
                       num_layers,
                       dff, # deep feed forward
                       dropout_rate,
                       sigmoid=True,
                      )


# In[37]:


model.summary()


# In[40]:


kfolds = local_seq_df.kfold.nunique()

oof = np.zeros(len(local_seq_df)) # out-of-fold predictions

for fold in range(local_seq_df.kfold.nunique()):
    print("=>> Fold:", fold+1)
    # adding mask after split
    train_df, valid_df = local_seq_df[local_seq_df.kfold != fold], local_seq_df[local_seq_df.kfold == fold]

    # Splitting the sent_ids and sent_msk arrays using the indices
    sent_ids_trn, sent_msk_trn = sent_ids[train_df.index], sent_msk[train_df.index]
    sent_ids_val, sent_msk_val = sent_ids[valid_df.index], sent_msk[valid_df.index]

    train_set = [train_df["items"].values,
                 [sent_ids_trn, sent_msk_trn],
                 train_df["users"].values,
                 train_df["local_mask"].values
                ]

    valid_set = [valid_df["items"].values,
                 [sent_ids_val, sent_msk_val],
                 valid_df["users"].values,
                 valid_df["local_mask"].values
                ]

    train_labels = train_df["labels"].values
    valid_labels = valid_df["labels"].values

    model = Sequence_aware( item_input, # inputs
                            sent_input,
                            editor_input,
                            local_mask_input,
                            emb_dim,
                            transformer_embedding_model_name,
                            n_clusters,
                            item_input_dim,
                            vocab_size,
                            editor_input_dim,
                            num_heads,
                            num_layers,
                            dff, # deep feed forward
                            dropout_rate,
                            sigmoid=True,
                        )

    history = model.fit(train_set, train_labels,
                        epochs=2, #10
                        batch_size=batch_size,
                        validation_data=(valid_set, valid_labels),
                        )

    y_pred = model.predict(valid_set, verbose=0)[:, 0]
    oof[valid_df.index] = y_pred
    print("=" * 90)

    #break


# ## Evaluate / Test

# In[41]:


from sklearn.metrics import roc_auc_score

def recall_at_k(y_true, y_pred, k):
    relevant_indices = np.where(y_true == 1)[0]
    top_k_indices = np.argsort(y_pred)[-k:]

    num_relevant_in_top_k = np.intersect1d(relevant_indices, top_k_indices).shape[0]
    recall = num_relevant_in_top_k / len(relevant_indices)

    return recall

def ndcg_at_k(y_true, y_pred, k):
    relevant_indices = np.where(y_true == 1)[0]
    top_k_indices = np.argsort(y_pred)[-k:]

    dcg = np.sum(1 / np.log2(np.arange(2, k + 2)))  # DCG for ideal ranking
    idcg = np.sum(1 / np.log2(np.arange(2, len(relevant_indices) + 2)))  # Ideal DCG

    relevant_positions = np.searchsorted(top_k_indices, relevant_indices)
    ndcg = np.sum(1 / np.log2(relevant_positions + 2)) / idcg

    return ndcg

def auc_at_k(y_true, y_pred):
    auc = roc_auc_score(y_true, y_pred)
    return auc


# In[42]:


# Assuming you have y_true (ground truth labels) and y_pred (predicted scores) for a user
k_values = [10, 100, 200]

# assign y_true and y_pred
y_true = local_seq_df.labels.values
y_pred = oof

recalls = []
ndcgs = []
aucs = []

for k in k_values:
    recall = recall_at_k(y_true, y_pred, k)
    ndcg = ndcg_at_k(y_true, y_pred, k)
    auc = auc_at_k(y_true, y_pred)

    recalls.append(recall)
    ndcgs.append(ndcg)
    aucs.append(auc)

    print(f"Metrics at k = {k}:")
    print(f"Recall@{k}: {recall:.4f}")
    print(f"NDCG@{k}: {ndcg:.4f}")
    print(f"AUC@{k}: {auc:.4f}")
    print("="*30)


# In[43]:


seq_res = pd.DataFrame([recalls, ndcgs, aucs], index=["Recall", "NDCG", "AUC"], columns=["@10", "@100", "@200"])
seq_res


# In[44]:


import matplotlib.pyplot as plt
import seaborn as sns
# Transpose the DataFrame for better plotting
df_transposed = seq_res.transpose()

# Set up the style for the plots
sns.set(style="whitegrid")

# Create subplots for recall, NDCG, and AUC
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

# Plot Recall
sns.lineplot(data=df_transposed['Recall'], ax=axes[0], marker='o')
axes[0].set_title('Recall Scores')
axes[0].set_xlabel('Interval')
axes[0].set_ylabel('Score')

# Plot NDCG
sns.lineplot(data=df_transposed['NDCG'], ax=axes[1], marker='o')
axes[1].set_title('NDCG Scores')
axes[1].set_xlabel('Interval')
axes[1].set_ylabel('Score')

# Plot AUC
sns.lineplot(data=df_transposed['AUC'], ax=axes[2], marker='o')
axes[2].set_title('AUC Scores')
axes[2].set_xlabel('Interval')
axes[2].set_ylabel('Score')

# Adjust layout
plt.tight_layout()

# Show the plots
plt.show()


# In[45]:


seq_res.to_csv("seq_res.csv", index=True)


# In[ ]:




