from parameters_lstm import *

from torch import nn
from torch.autograd import Variable
import torch
import numpy as np
import time

from sklearn.feature_extraction.text import CountVectorizer

# Processes all sentences in out datasets to give useful containers of data concerning the corpus:
# word2id vocab
# dict of question id to list of words in the question
def process_whole_corpuses():
    list_dataset_paths = [ubuntu_corpus_path, android_corpus_path]
    all_txt = []
    all_txt_title = []
    all_txt_body = []
    ubuntu_id_to_data_title = {}
    ubuntu_id_to_data_body = {}
    android_id_to_data_title = {}
    android_id_to_data_body = {}

    for dataset_path in list_dataset_paths:
        lines = open(dataset_path, encoding = 'utf8').readlines()
        for line in lines:

            id_title_body_list = line.split('\t')
            idx = int(id_title_body_list[0])
            title_plus_body = id_title_body_list[1] + ' ' + id_title_body_list[2][:-1]
            all_txt.append(title_plus_body)
            all_txt_title.append(id_title_body_list[1])
            all_txt_body.append(id_title_body_list[2])

            if dataset_path == ubuntu_corpus_path:
                ubuntu_id_to_data_title[idx] = id_title_body_list[1].split()
                ubuntu_id_to_data_body[idx] = id_title_body_list[2].split()
            else:
                android_id_to_data_title[idx] = id_title_body_list[1].split()
                android_id_to_data_body[idx] = id_title_body_list[2].split()

    # vectorizer = CountVectorizer(binary=True, analyzer='word', token_pattern='[^\s]+[a-z]*[0-9]*')
    vectorizer = CountVectorizer(binary=True, token_pattern='[^\s]+[a-z]*[0-9]*', analyzer='word', max_df=0.3)

    vectorizer.fit(all_txt)

    return {
            'word_to_id': vectorizer.vocabulary_,
            'ubuntu_id_to_data_title': ubuntu_id_to_data_title,
            'ubuntu_id_to_data_body': ubuntu_id_to_data_body,
            'android_id_to_data_title': android_id_to_data_title,
            'android_id_to_data_body': android_id_to_data_body,
            'vectorizer': vectorizer
            }


# Get glove embeddings matrix only for words in our corpus (++Gain of gigabytes of memory)
# Matrix [num_words_with_embeddings x word_dim] is be fed to pytorch nn.Embedding module without gradient
# Function returns this nn.Embedding Object
def load_glove_embeddings(glove_path, word_to_id_vocab, embedding_dim=300):
    with open(glove_path, encoding = 'utf8') as f:
        glove_matrix = np.zeros((len(word_to_id_vocab), embedding_dim))
        for line in f.readlines():
            values = line.split()
            word = values[0]
            index = word_to_id_vocab.get(word)
            if index:
                try:
                    vector = np.array(values[1:], dtype='float32')
                    glove_matrix[index] = vector
                except:
                    pass

    glove_matrix = torch.from_numpy(glove_matrix).float()
    torch_embedding = nn.Embedding(glove_matrix.size(0), glove_matrix.size(1), padding_idx=padding_idx)
    torch_embedding.weight = nn.Parameter(glove_matrix)
    torch_embedding.weight.requires_grad = False

    return torch_embedding


# Takes a question id and the corresponding dict of question_id_to_words
# Builds a matrix of [1 x num_words x input_size] where first dim is for concatenation in future
# Use up to TRUNCATE_LENGTH number of words and pad if needed
def get_question_matrix_title(question_id, dict_qid_to_words, words_to_id_vocabulary, pytorch_embeddings, truncation_val_title):
    question_data = dict_qid_to_words[question_id]
    word_ids = []

    # Build list of ids of words in that question
    for word in question_data:
        if len(word_ids) == truncation_val_title: break

        try: word_ids.append(int(words_to_id_vocabulary[word.lower()]))
        except: pass

    # Pad if need more rows
    number_words_before_padding = len(word_ids)
    if number_words_before_padding < truncation_val_title: word_ids += [padding_idx] * (truncation_val_title - len(word_ids))

    question_in_embedded_form = pytorch_embeddings(torch.LongTensor(word_ids)).data
    return question_in_embedded_form.unsqueeze(0), number_words_before_padding


# Takes a question id and the corresponding dict of question_id_to_words
# Builds a matrix of [1 x num_words x input_size] where first dim is for concatenation in future
# Use up to TRUNCATE_LENGTH number of words and pad if needed
def get_question_matrix_body(question_id, dict_qid_to_words, words_to_id_vocabulary, pytorch_embeddings, truncation_val_body):
    question_data = dict_qid_to_words[question_id]
    word_ids = []

    # Build list of ids of words in that question
    for word in question_data:
        if len(word_ids) == truncation_val_body: break

        try: word_ids.append(int(words_to_id_vocabulary[word.lower()]))
        except: pass

    # Pad if need more rows
    number_words_before_padding = len(word_ids)
    if number_words_before_padding < truncation_val_body: word_ids += [padding_idx] * (truncation_val_body - len(word_ids))

    question_in_embedded_form = pytorch_embeddings(torch.LongTensor(word_ids)).data
    return question_in_embedded_form.unsqueeze(0), number_words_before_padding


# Given ids of main qs in this batch
# Returns:
# 1. ids in ordered list as:
# [ q_1+, q_1-, q_1--,..., q_1++, q_1-, q_1--,...,
# q_2+, q_2-, q_2--,..., q_2++, q_2-, q_2--,...,]
# All n main questions have their pos,neg,neg,neg,... interleaved
# 2. A dict mapping main question id --> its interleaved sequence length
def organize_ids_training(q_ids, data, num_differing_questions):
    sequence_ids = []
    dict_sequence_lengths = {}

    for q_main in q_ids:
        p_pluses = data[q_main][0]
        p_minuses = list(np.random.choice(data[q_main][1], num_differing_questions, replace=False))
        sequence_length = len(p_pluses) * num_differing_questions + len(p_pluses)
        dict_sequence_lengths[q_main] = sequence_length
        for p_plus in p_pluses:
            sequence_ids += [p_plus] + p_minuses

    return sequence_ids, dict_sequence_lengths


# Given ids of main qs in this android batch
# Returns:
# 1. list of ids of all the questions
# 2. list of q_mains, replicated to correspond to list of candidates
# 3. list of 1,0... 1 for pos, 0 for neg (wrt. candidates) to be used in AUC metric
def organize_test_ids(q_ids, data):
    processed_ids = []
    target_labels = []
    q_main_ids = []

    for q_main in q_ids:
        all_p = data[q_main][1]
        p_pluses = data[q_main][0]
        for p in all_p:
            if p in p_pluses:
                target_labels.append(1)
            else:
                target_labels.append(0)
        processed_ids += all_p
        q_main_ids += [q_main] * len(all_p)

    return processed_ids, q_main_ids, target_labels



# A tuple is (q+, q-, q--, q--- ...)
# Let all main questions be set Q
# Each q in Q has a number of tuples equal to number of positives |q+, q++, ...|
# Each q in Q will have a 2D matrix of: num_tuples x num_candidates_in_tuple
# Concatenate this matrix for all q in Q and you get a matrix of: |Q| x num_tuples x num_candidates_in_tuple

# The above is for candidates
# To do cosine_similarity, need same structure with q's
# Basically each q will be a matrix of repeated q's: num_tuples x num_candidates_in_tuple, all elts are q (repeated)

# This method constructs those matrices, use candidates=True for candidates matrix
def construct_qs_matrix_training(q_ids_sequential, lstm, h0, c0, word2vec, id2Data_title, id2Data_body, dict_sequence_lengths,
                                 num_differing_questions, word_to_id_vocab, truncation_val_title, truncation_val_body, candidates=False):
    if not candidates:
        q_ids_complete = []
        for q in q_ids_sequential:
            q_ids_complete += [q] * dict_sequence_lengths[q]

    else:
        q_ids_complete = q_ids_sequential

    qs_matrix_list_title = []
    qs_seq_length_title = []
    qs_matrix_list_body = []
    qs_seq_length_body = []

    for q in q_ids_complete:
        q_matrix_3d_title, q_num_words_title = get_question_matrix_title(q, id2Data_title, word_to_id_vocab, word2vec, truncation_val_title)
        qs_matrix_list_title.append(q_matrix_3d_title)
        qs_seq_length_title.append(q_num_words_title)
        q_matrix_3d_body, q_num_words_body = get_question_matrix_body(q, id2Data_body, word_to_id_vocab, word2vec, truncation_val_body)
        qs_matrix_list_body.append(q_matrix_3d_body)
        qs_seq_length_body.append(q_num_words_body)

    qs_padded_title = Variable(torch.cat(qs_matrix_list_title, 0))
    qs_hidden_title = lstm(qs_padded_title, (h0, c0))
    sum_h_qs_title = torch.sum(qs_hidden_title[0], dim=1)
    qs_padded_body = Variable(torch.cat(qs_matrix_list_body, 0))
    qs_hidden_body = lstm(qs_padded_body, (h0, c0))
    sum_h_qs_body = torch.sum(qs_hidden_body[0], dim=1)
    mean_pooled_h_qs_title = torch.div(sum_h_qs_title, torch.autograd.Variable(torch.FloatTensor(qs_seq_length_title)[:, np.newaxis]) + 10**(-8))
    mean_pooled_h_qs_body = torch.div(sum_h_qs_body, torch.autograd.Variable(torch.FloatTensor(qs_seq_length_body)[:, np.newaxis]) + 10**(-8))
    avg_pooled_h_qs_title_body = (mean_pooled_h_qs_title + mean_pooled_h_qs_body)/2
    qs_tuples = avg_pooled_h_qs_title_body.split(1 + num_differing_questions)
    final_matrix_tuples_by_constituent_qs_by_hidden_size = torch.stack(qs_tuples, dim=0, out=None)

    return final_matrix_tuples_by_constituent_qs_by_hidden_size


def construct_qs_matrix_testing(q_ids, lstm, h0, c0, word2vec, id2Data_title, id2Data_body, word_to_id_vocab, truncation_val_title, truncation_val_body, main=False):
    qs_matrix_list_title = []
    qs_seq_length_title = []
    qs_matrix_list_body = []
    qs_seq_length_body = []

    for q in q_ids:
        #if main: q = q[0]
        q_matrix_3d_title, q_num_words_title = get_question_matrix_title(q, id2Data_title, word_to_id_vocab, word2vec, truncation_val_title)
        qs_matrix_list_title.append(q_matrix_3d_title)
        qs_seq_length_title.append(q_num_words_title)
        q_matrix_3d_body, q_num_words_body = get_question_matrix_body(q, id2Data_body, word_to_id_vocab, word2vec, truncation_val_body)
        qs_matrix_list_body.append(q_matrix_3d_body)
        qs_seq_length_body.append(q_num_words_body)

    qs_padded_title = Variable(torch.cat(qs_matrix_list_title, 0))
    qs_hidden_title = lstm(qs_padded_title, (h0, c0))
    sum_h_qs_title = torch.sum(qs_hidden_title[0], dim=1)
    qs_padded_body = Variable(torch.cat(qs_matrix_list_body, 0))
    qs_hidden_body = lstm(qs_padded_body, (h0, c0))
    sum_h_qs_body = torch.sum(qs_hidden_body[0], dim=1)
    mean_pooled_h_qs_title = torch.div(sum_h_qs_title, torch.autograd.Variable(torch.FloatTensor(qs_seq_length_title)[:, np.newaxis]) + 10**(-8))
    mean_pooled_h_qs_body = torch.div(sum_h_qs_body, torch.autograd.Variable(torch.FloatTensor(qs_seq_length_body)[:, np.newaxis]) + 10**(-8))
    avg_pooled_h_qs_title_body = (mean_pooled_h_qs_title + mean_pooled_h_qs_body)/2
    #if main: avg_pooled_h_qs_title_body = torch.cat([avg_pooled_h_qs_title_body] * q_ids[0][1], 0)

    return avg_pooled_h_qs_title_body


# For categorization of questions by neural net, build a matrix of numq * lstm_hidden_layer_size
# Takes in list of q ids
# Matrix is to be fed as a batch to a neural network after being stacked with a similar matrix for another domain and compared to target
def construct_qs_matrix_domain_classification(domain_ids, lstm, h0, c0, word2vec, domain_specific_id_to_data_title, domain_specific_id_to_data_body, word_to_id_vocab, truncation_val_title, truncation_val_body):
    qs_matrix_list_title = []
    qs_seq_length_title = []
    qs_matrix_list_body = []
    qs_seq_length_body = []

    for q in domain_ids:
        q_matrix_3d_title, q_num_words_title = get_question_matrix_title(q, domain_specific_id_to_data_title, word_to_id_vocab, word2vec, truncation_val_title)
        qs_matrix_list_title.append(q_matrix_3d_title)
        qs_seq_length_title.append(q_num_words_title)
        q_matrix_3d_body, q_num_words_body = get_question_matrix_body(q, domain_specific_id_to_data_body, word_to_id_vocab, word2vec, truncation_val_body)
        qs_matrix_list_body.append(q_matrix_3d_body)
        qs_seq_length_body.append(q_num_words_body)

    qs_padded_title = Variable(torch.cat(qs_matrix_list_title, 0))
    qs_hidden_title = lstm(qs_padded_title, (h0, c0))
    sum_h_qs_title = torch.sum(qs_hidden_title[0], dim=1)
    qs_padded_body = Variable(torch.cat(qs_matrix_list_body, 0))
    qs_hidden_body = lstm(qs_padded_body, (h0, c0))
    sum_h_qs_body = torch.sum(qs_hidden_body[0], dim=1)
    mean_pooled_h_qs_title = torch.div(sum_h_qs_title, torch.autograd.Variable(torch.FloatTensor(qs_seq_length_title)[:, np.newaxis]) + 10**(-8))
    mean_pooled_h_qs_body = torch.div(sum_h_qs_body, torch.autograd.Variable(torch.FloatTensor(qs_seq_length_body)[:, np.newaxis]) + 10**(-8))
    avg_pooled_h_qs_title_body = (mean_pooled_h_qs_title + mean_pooled_h_qs_body)/2

    return avg_pooled_h_qs_title_body
