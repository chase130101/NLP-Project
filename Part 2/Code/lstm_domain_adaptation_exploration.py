from parameters_lstm import *

from preprocess_datapoints_lstm import *
from preprocess_text_to_tensors_lstm import *
from domain_classifier_model_lstm import *
from meter import *

import torch
from torch.autograd import Variable
import numpy
import time

saved_model_name_lstm = 'lstm_domain_adapt_exp'

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

np.random.seed(1)
data_android = android_id_to_similar_different(dev=True)
all_dev_question_ids_android = list(data_android.keys())
training_question_ids_android = all_dev_question_ids_android[:500]
dev_question_ids_android = all_dev_question_ids_android[500:]


''' Encoder (LSTM) '''
lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional)
loss_function_lstm = torch.nn.MultiMarginLoss(margin=margin)
optimizer_lstm = torch.optim.Adam(lstm.parameters(), lr=lr_lstm)

h0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)
c0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)


''' Domain Classifier (Neural Net) '''
neural_net = DomainClassifier(input_size_nn, first_hidden_size_nn, second_hidden_size_nn)
loss_function_nn = nn.CrossEntropyLoss()
optimizer_nn = torch.optim.Adam(neural_net.parameters(), lr=lr_nn)

''' Procedural parameters '''
num_batches = round(len(training_question_ids_ubuntu) / batch_size)
auc_scorer = AUCMeter()


def train_lstm_question_similarity(lstm, batch_ids_ubuntu, batch_ids_android, batch_data_ubuntu, batch_data_android, word2vec, ubuntu_id_to_data_title, ubuntu_id_to_data_body,\
                                   android_id_to_data_title, android_id_to_data_body, word_to_id_vocab, truncation_val_title, truncation_val_body):
    lstm.train()
    sequence_ids_ubuntu, dict_sequence_lengths_ubuntu = organize_ids_training(batch_ids_ubuntu, batch_data_ubuntu, num_differing_questions)
    sequence_ids_android, dict_sequence_lengths_android = organize_ids_training(batch_ids_android, batch_data_android, num_differing_questions)
    
    candidates_qs_tuples_matrix_ubuntu = construct_qs_matrix_training(sequence_ids_ubuntu, lstm, h0, c0, word2vec, ubuntu_id_to_data_title, ubuntu_id_to_data_body,
        dict_sequence_lengths_ubuntu, num_differing_questions, word_to_id_vocab, truncation_val_title, truncation_val_body, candidates=True)
    
    candidates_qs_tuples_matrix_android = construct_qs_matrix_training(sequence_ids_android, lstm, h0, c0, word2vec, android_id_to_data_title, android_id_to_data_body,
        dict_sequence_lengths_android, num_differing_questions, word_to_id_vocab, truncation_val_title, truncation_val_body, candidates=True)
    
    main_qs_tuples_matrix_ubuntu = construct_qs_matrix_training(batch_ids_ubuntu, lstm, h0, c0, word2vec, ubuntu_id_to_data_title, ubuntu_id_to_data_body,
        dict_sequence_lengths_ubuntu, num_differing_questions, word_to_id_vocab, truncation_val_title, truncation_val_body, candidates=False)
    
    main_qs_tuples_matrix_android = construct_qs_matrix_training(batch_ids_android, lstm, h0, c0, word2vec, android_id_to_data_title, android_id_to_data_body,
        dict_sequence_lengths_android, num_differing_questions, word_to_id_vocab, truncation_val_title, truncation_val_body, candidates=False)
        
    similarity_matrix_ubuntu = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix_ubuntu, main_qs_tuples_matrix_ubuntu, dim=2, eps=1e-6)
    target_ubuntu = Variable(torch.LongTensor([0] * int(len(sequence_ids_ubuntu) / (1 + num_differing_questions))))
    
    similarity_matrix_android = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix_android, main_qs_tuples_matrix_android, dim=2, eps=1e-6)
    target_android = Variable(torch.LongTensor([0] * int(len(sequence_ids_android) / (1 + num_differing_questions))))
    
    loss_batch = (loss_function_lstm(similarity_matrix_ubuntu, target_ubuntu) + loss_function_lstm(similarity_matrix_android, target_android))/2

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
    target_vector = Variable(torch.LongTensor(torch.cat([torch.zeros(20).long(), torch.ones(20).long()])))
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
        ids_this_batch_for_lstm_ubuntu = training_question_ids_ubuntu[batch_size * (batch - 1):batch_size * batch]
        ids_this_batch_for_lstm_android = np.array(training_question_ids_android)[np.random.choice(np.arange(len(training_question_ids_android)), size = round(batch_size/6), replace = False)]
        loss_batch_similarity = train_lstm_question_similarity(lstm, ids_this_batch_for_lstm_ubuntu, ids_this_batch_for_lstm_android,
        training_data_ubuntu, data_android, word2vec, ubuntu_id_to_data_title, ubuntu_id_to_data_body, android_id_to_data_title, android_id_to_data_body, word_to_id_vocab,
                                                               truncation_val_title, truncation_val_body)

        # Train on ubuntu-android domain classification task
        ids_randomized_ubuntu = get_20_random_ids(training_question_ids_ubuntu)
        ids_randomized_android = get_20_random_ids(training_question_ids_android)
        loss_batch_domain_classification = train_nn_domain_classification(neural_net, lstm, h0, c0,
            ids_randomized_ubuntu, ids_randomized_android, word2vec, ubuntu_id_to_data_title, ubuntu_id_to_data_body, android_id_to_data_title, android_id_to_data_body,
                                                                         truncation_val_title, truncation_val_body)

        # Overall loss = multi-margin loss - LAMBDA * cross entropy loss
        overall_loss = loss_batch_similarity - (lamb * loss_batch_domain_classification)
        overall_loss.backward()
        optimizer_lstm.step()
        optimizer_nn.step()

        print("Time_on_batch:", time.time() - start)
        
    torch.save(lstm, '../Pickle/' + saved_model_name_lstm + '_e' + str(epoch) + '_b' + str(batch) + '.pth')