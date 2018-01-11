from parameters_lstm import *

from preprocess_datapoints_lstm import *
from preprocess_text_to_tensors_lstm import *
from domain_classifier_model_lstm import *
from meter import *

import torch
from torch.autograd import Variable
import time

saved_model_name_lstm = 'lstm_domain_adapt'

# Initialize the data sets
processed_corpus = process_whole_corpuses()
word_to_id_vocab = processed_corpus['word_to_id']
word2vec = load_glove_embeddings(glove_path, word_to_id_vocab)
ubuntu_id_to_data_title = processed_corpus['ubuntu_id_to_data_title']
android_id_to_data_title = processed_corpus['android_id_to_data_title']
ubuntu_id_to_data_body = processed_corpus['ubuntu_id_to_data_body']
android_id_to_data_body = processed_corpus['android_id_to_data_body']


''' Data Sets '''
training_data_ubuntu = ubuntu_id_to_similar_different()
training_question_ids_ubuntu = list(training_data_ubuntu.keys())
# Note: Remember to edit batch_size accordingly if testing on smaller size data sets


''' Encoder (LSTM) '''
lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
loss_function_lstm = torch.nn.MultiMarginLoss(margin=margin)
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=lr_lstm, weight_decay = 0.00001)

h0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)
c0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)


''' Domain Classifier (Neural Net) '''
neural_net = DomainClassifier(input_size_nn, first_hidden_size_nn, second_hidden_size_nn)
loss_function_nn = nn.CrossEntropyLoss()
optimizer_nn = torch.optim.Adam(neural_net.parameters(), lr=lr_nn, weight_decay = 0.00001)


''' Procedural parameters '''
num_batches = round(len(training_question_ids_ubuntu) / batch_size)
auc_scorer = AUCMeter()


def train_lstm_question_similarity(lstm, batch_ids, batch_data, word2vec, id2Data_title, id2Data_body, word_to_id_vocab, truncation_val_title, truncation_val_body):
    lstm.train()
    sequence_ids, dict_sequence_lengths = organize_ids_training(batch_ids, batch_data, num_differing_questions)

    candidates_qs_tuples_matrix = construct_qs_matrix_training(sequence_ids, lstm, h0, c0, word2vec, id2Data_title, id2Data_body,
        dict_sequence_lengths, num_differing_questions, word_to_id_vocab, truncation_val_title, truncation_val_body, candidates=True)
    main_qs_tuples_matrix = construct_qs_matrix_training(batch_ids, lstm, h0, c0, word2vec, id2Data_title, id2Data_body,
        dict_sequence_lengths, num_differing_questions, word_to_id_vocab, truncation_val_title, truncation_val_body, candidates=False)

    similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix, dim=2, eps=1e-6)
    target = Variable(torch.LongTensor([0] * int(len(sequence_ids) / (1 + num_differing_questions))))
    loss_batch = loss_function_lstm(similarity_matrix, target)

    print("lstm multi-margin loss on batch:", loss_batch.data[0])
    return loss_batch


def train_nn_domain_classification(neural_net, lstm, h0, c0, ids_ubuntu, ids_android, word2vec,
    ubuntu_id_to_data_title, ubuntu_id_to_data_body, android_id_to_data_title, android_id_to_data_body, truncation_val_title, truncation_val_body):
    neural_net.train()
    lstm.train()

    qs_matrix_ubuntu = construct_qs_matrix_domain_classification(ids_ubuntu, lstm, h0, c0, word2vec,
        ubuntu_id_to_data_title, ubuntu_id_to_data_body, word_to_id_vocab, truncation_val_title, truncation_val_body)
    qs_matrix_android = construct_qs_matrix_domain_classification(ids_android, lstm, h0, c0, word2vec,
        android_id_to_data_title, android_id_to_data_body, word_to_id_vocab, truncation_val_title, truncation_val_body)
    overall_qs_matrix = torch.cat([qs_matrix_ubuntu, qs_matrix_android])

    out = neural_net.forward(overall_qs_matrix)
    target_vector = Variable(torch.cat([torch.zeros(20).long(), torch.ones(20).long()]))
    loss_batch = loss_function_nn(out.double(), target_vector)

    print("Neural net cross-entropy loss on batch:", loss_batch.data[0])
    return loss_batch.float()


'''Begin training'''
for epoch in range(num_epochs):

    # Train on whole training data set
    for batch in range(1, num_batches + 1):
        start = time.time()
        optimizer_lstm.zero_grad()
        optimizer_nn.zero_grad()
        print("Working on batch #: ", batch)

        # Train on ubuntu similar question retrieval
        ids_this_batch_for_lstm = training_question_ids_ubuntu[batch_size * (batch - 1):batch_size * batch]
        loss_batch_similarity = train_lstm_question_similarity(lstm, ids_this_batch_for_lstm,
        training_data_ubuntu, word2vec, ubuntu_id_to_data_title, ubuntu_id_to_data_body, word_to_id_vocab, truncation_val_title, truncation_val_body)

        # Train on ubuntu-android domain classification task
        ids_randomized_ubuntu = get_20_random_ids(training_question_ids_ubuntu)
        ids_randomized_android = get_20_random_ids(dev_question_ids_android)
        loss_batch_domain_classification = train_nn_domain_classification(neural_net, lstm, h0, c0,
            ids_randomized_ubuntu, ids_randomized_android, word2vec, ubuntu_id_to_data_title, ubuntu_id_to_data_body, android_id_to_data_title, android_id_to_data_body,
                                                                         truncation_val_title, truncation_val_body)

        # Overall loss = multi-margin loss - LAMBDA * cross entropy loss
        overall_loss = loss_batch_similarity - (lamb * loss_batch_domain_classification)
        overall_loss.backward()
        optimizer_lstm.step()
        optimizer_nn.step()

        print("Time on batch:", time.time() - start)
        
    # Save model for this epoch
    torch.save(lstm, '../Pickle/' + saved_model_name_lstm + '_e' + str(epoch) + '_b' + str(batch) + '.pth')