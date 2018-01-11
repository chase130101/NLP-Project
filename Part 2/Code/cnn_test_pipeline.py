from parameters_cnn import *

from preprocess_datapoints_cnn import *
from preprocess_text_to_tensors_cnn import *
from domain_classifier_model_cnn import *
from meter import *

import torch
from torch.autograd import Variable


truncation_val_title = 15
truncation_val_body = 85


# Initialize the data sets
processed_corpus = process_whole_corpuses()
word_to_id_vocab = processed_corpus['word_to_id']
word2vec = load_glove_embeddings(glove_path, word_to_id_vocab)
android_id_to_data_title = processed_corpus['android_id_to_data_title']
android_id_to_data_body = processed_corpus['android_id_to_data_body']


''' Data Sets '''
test_data_android = android_id_to_similar_different(dev=False)
test_question_ids_android = list(test_data_android.keys())


''' Encoder (LSTM) '''
#cnn = torch.load('model directory here')


''' Procedural parameters '''
auc_scorer = AUCMeter()


def eval_model(cnn, ids, data, word2vec, id2Data_title, id2Data_body, word_to_id_vocab, truncation_val_title, truncation_val_body):
    lstm.eval()
    auc_scorer.reset()

    candidate_ids, q_main_ids, labels = organize_test_ids(ids, data)
    num_q_main = len(q_main_ids)
    len_pieces = round(num_q_main/50)
    print(num_q_main)

    for i in range(0, num_q_main, len_pieces):
        print(i, end = ' ')
        q_main_id_num_repl_tuple = q_main_ids[i:i+len_pieces]
        candidates = candidate_ids[i:i+len_pieces]
        current_labels = torch.from_numpy(np.array(labels[i:i+len_pieces])).long()

        candidates_qs_matrix = construct_qs_matrix_testing(candidates, cnn, word2vec, id2Data_title, id2Data_body,
        word_to_id_vocab, truncation_val_title, truncation_val_body, main=False)
        main_qs_matrix = construct_qs_matrix_testing(q_main_id_num_repl_tuple, cnn, word2vec, id2Data_title, id2Data_body,
        word_to_id_vocab, truncation_val_title, truncation_val_body, main=True)

        similarity_matrix_this_batch = torch.nn.functional.cosine_similarity(candidates_qs_matrix, main_qs_matrix, eps=1e-6).data
        auc_scorer.add(similarity_matrix_this_batch, current_labels)

    auc_score = auc_scorer.value()

    return auc_score


# Evaluate on dev/test set for AUC score
test_AUC_score = eval_model(cnn, test_question_ids_android, test_data_android, word2vec, android_id_to_data_title, android_id_to_data_body,
                            word_to_id_vocab, truncation_val_title, truncation_val_body)

print("Test AUC score:", test_AUC_score)