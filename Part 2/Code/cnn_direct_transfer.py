from parameters_cnn import *

from preprocess_datapoints_cnn import *
from preprocess_text_to_tensors_cnn import *
from meter import *

import torch
from torch.autograd import Variable
import time

saved_model_name = 'cnn_dir_transfer'

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


''' Encoder (CNN) '''
# CNN model
cnn = torch.nn.Sequential()
cnn.add_module('conv', torch.nn.Conv1d(in_channels = input_size, out_channels = hidden_size, kernel_size = kernel_size, padding = padding, dilation = dilation, groups = groups, bias = bias))
cnn.add_module('tanh', torch.nn.Tanh())
cnn.add_module('norm', torch.nn.BatchNorm1d(num_features = hidden_size))

loss_function_cnn = torch.nn.MultiMarginLoss(margin=margin)
optimizer_cnn = torch.optim.Adam(cnn.parameters(), lr=lr_cnn)


''' Procedural parameters '''
num_batches = round(len(training_question_ids_ubuntu) / batch_size)


def train_cnn_question_similarity(cnn, batch_ids, batch_data, word2vec, id2Data_title, id2Data_body, word_to_id_vocab, truncation_val_title, truncation_val_body):
    cnn.train()
    sequence_ids, dict_sequence_lengths = organize_ids_training(batch_ids, batch_data, num_differing_questions)

    candidates_qs_tuples_matrix = construct_qs_matrix_training(sequence_ids, cnn, word2vec, id2Data_title, id2Data_body,
        dict_sequence_lengths, num_differing_questions, word_to_id_vocab, truncation_val_title, truncation_val_body, candidates=True)
    main_qs_tuples_matrix = construct_qs_matrix_training(batch_ids, cnn, word2vec, id2Data_title, id2Data_body,
        dict_sequence_lengths, num_differing_questions, word_to_id_vocab, truncation_val_title, truncation_val_body, candidates=False)

    similarity_matrix = torch.nn.functional.cosine_similarity(candidates_qs_tuples_matrix, main_qs_tuples_matrix, dim=2, eps=1e-6)
    target = Variable(torch.LongTensor([0] * int(len(sequence_ids) / (1 + num_differing_questions))))
    loss_batch = loss_function_cnn(similarity_matrix, target)

    print("cnn multi-margin loss on batch:", loss_batch.data[0])
    return loss_batch


'''Begin training'''
for epoch in range(num_epochs):

    # Train on whole training data set
    for batch in range(1, num_batches + 1):
        start = time.time()
        optimizer_cnn.zero_grad()
        print("Working on batch #: ", batch)

        # Train on ubuntu similar question retrieval
        ids_this_batch_for_cnn = training_question_ids_ubuntu[batch_size * (batch - 1):batch_size * batch]
        loss_batch_similarity = train_cnn_question_similarity(cnn, ids_this_batch_for_cnn,
        training_data_ubuntu, word2vec, ubuntu_id_to_data_title, ubuntu_id_to_data_body, word_to_id_vocab, truncation_val_title, truncation_val_body)

        overall_loss = loss_batch_similarity
        overall_loss.backward()
        optimizer_cnn.step()

        print("Time on batch:", time.time() - start)

    # Save model for this epoch
    torch.save(cnn, '../Pickle/' + saved_model_name + '_e' + str(epoch) + '_b' + str(batch) + '.pth')