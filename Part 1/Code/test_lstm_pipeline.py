from preprocess import *
from scoring_metrics import *
from lstm_utils import *


truncation_val_title = 40
truncation_val_body = 60
num_differing_questions = 20
hidden_size = 100
num_layers = 1
bidirectional = True
first_dim = num_layers * 2 if bidirectional else num_layers
h0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)
c0 = Variable(torch.zeros(first_dim, 1, hidden_size), requires_grad=False)


''' Data Prep '''
dev = False
testing_data = devTest_id_to_similar_different(dev)
testingQuestionIds = list(testing_data.keys())
word2vec = get_words_and_embeddings()
id2Data = questionID_to_questionData_truncate(truncation_val_title, truncation_val_body)
input_size = input_size = len(word2vec[list(word2vec.keys())[0]])


''' Model (Specify pickled model name)'''
lstm = torch.load('../Some_Pickled_Models/lstm_good.pt')
lstm.eval()


'''Begin testing'''
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

test_scores = eval_model(lstm, testingQuestionIds, testing_data, word2vec, id2Data, truncation_val_title, truncation_val_body)

print("MRR score on test set:", test_scores[0])
print("MAP score on test set:", test_scores[1])
print("Precision at 1 score on test set:", test_scores[2])
print("Precision at 5 score on test set:", test_scores[3])