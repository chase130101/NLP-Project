from preprocess import *
from scoring_metrics import *
from lstm_utils import *

import torch
from torch.autograd import Variable

import time


saved_model_name = "lstm"


'''Hyperparams dashboard'''
margin = 0.2
dropout = 0.2
lr = 10**-3
truncation_val_title = 40
truncation_val_body = 60


''' Data Prep '''
word2vec = get_words_and_embeddings()
id2Data = questionID_to_questionData_truncate(truncation_val_title, truncation_val_body)

training_data = training_id_to_similar_different()
trainingQuestionIds = list(training_data.keys())

dev_data = devTest_id_to_similar_different(dev=True)
dev_question_ids = list(dev_data.keys())


''' Model Specs '''
input_size = len(word2vec[list(word2vec.keys())[0]])
hidden_size = 100
num_layers = 1
bias = True
batch_first = True
bidirectional = True


lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
loss_function = torch.nn.MultiMarginLoss(margin=margin)
optimizer = torch.optim.Adam(lstm.parameters(), lr=lr)


first_dim = num_layers * 2 if bidirectional else num_layers
h0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)
c0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)


''' Procedural parameters '''
batch_size = 20
num_differing_questions = 20
num_epochs = 5
num_batches = round(len(trainingQuestionIds)/batch_size)


def train_model(lstm, optimizer, batch_ids, batch_data, word2vec, id2Data, truncation_val_title, truncation_val_body):
    lstm.train()
    optimizer.zero_grad()

    sequence_ids, dict_sequence_lengths = organize_ids_training(batch_ids, batch_data, num_differing_questions)

    candidates_qs_tuples_matrix = construct_qs_matrix_training(sequence_ids, lstm, h0, c0, word2vec, id2Data, dict_sequence_lengths, input_size, num_differing_questions, truncation_val_title,\
                                                               truncation_val_body, candidates=True)
    main_qs_tuples_matrix = construct_qs_matrix_training(batch_ids, lstm, h0, c0, word2vec, id2Data, dict_sequence_lengths, input_size, num_differing_questions, truncation_val_title,\
                                                         truncation_val_body, candidates=False)
    similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix, dim=2, eps=1e-08)

    target = Variable(torch.LongTensor([0] * int(len(sequence_ids)/(1+num_differing_questions))))
    loss_batch = loss_function(similarity_matrix, target)

    loss_batch.backward()
    optimizer.step()

    print("loss_on_batch:", loss_batch.data[0], " time_on_batch:", time.time() - start)
    return


def eval_model(lstm, ids, data, word2vec, id2Data, truncation_val_title, truncation_val_body):
    lstm.eval()
    sequence_ids, p_pluses_indices_dict = organize_test_ids(ids, data)

    candidates_qs_tuples_matrix = construct_qs_matrix_testing(sequence_ids, lstm, h0, c0, word2vec, id2Data, input_size, num_differing_questions, truncation_val_title, truncation_val_body,\
                                                              candidates=True)
    main_qs_tuples_matrix = construct_qs_matrix_testing(ids, lstm, h0, c0, word2vec, id2Data, input_size, num_differing_questions, truncation_val_title, truncation_val_body, candidates=False)

    similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix, dim=2, eps=1e-08)
    MRR_score = get_MRR_score(similarity_matrix, p_pluses_indices_dict)
    MAP_score = get_MAP_score(similarity_matrix, p_pluses_indices_dict)
    avg_prec_at_1 = avg_precision_at_k(similarity_matrix, p_pluses_indices_dict, 1)
    avg_prec_at_5 = avg_precision_at_k(similarity_matrix, p_pluses_indices_dict, 5) 
    return MRR_score, MAP_score, avg_prec_at_1, avg_prec_at_5


'''Begin training'''
for epoch in range(num_epochs):

    # Train on whole training data set
    for batch in range(1, num_batches+1):
        start = time.time()
        questions_this_training_batch = trainingQuestionIds[batch_size * (batch - 1):batch_size * batch]
        print("Working on batch #: ", batch)
        train_model(lstm, optimizer, questions_this_training_batch, training_data, word2vec, id2Data, truncation_val_title, truncation_val_body)

    dev_scores = eval_model(lstm, dev_question_ids, dev_data, word2vec, id2Data, truncation_val_title, truncation_val_body)
    print("MRR score on dev set:", dev_scores[0])

    # Save model for this epoch
    torch.save(lstm, '../Pickle/' + saved_model_name + '_e' + str(epoch) + '_b' + str(batch)+ '.pt')