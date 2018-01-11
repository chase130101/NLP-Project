from parameters_lstm import *

from preprocess_datapoints_lstm import *
from preprocess_text_to_tensors_lstm import *
from meter import *

from sklearn.feature_extraction.text import CountVectorizer as CountVectorizer

''' Data processing helpers only for TF-IDF implementation '''

# Processes all sentences in out datasets to give useful containers of data concerning the corpus:
# word2id vocab
# dict of question id to list of words in the question
# Processes all sentences in out datasets to give useful containers of data concerning the corpus:
# word2id vocab
# dict of question id to list of words in the question
def process_only_android_corpus():
    dataset_path = android_corpus_path
    all_txt = []
    all_txt_title = []
    all_txt_body = []
    android_id_to_data_title = {}
    android_id_to_data_body = {}

    lines = open(dataset_path, encoding = 'utf8').readlines()
    for line in lines:

        id_title_body_list = line.split('\t')
        idx = int(id_title_body_list[0])
        title_plus_body = id_title_body_list[1] + ' ' + id_title_body_list[2][:-1]
        all_txt.append(title_plus_body)
        all_txt_title.append(id_title_body_list[1])
        all_txt_body.append(id_title_body_list[2])

        android_id_to_data_title[idx] = id_title_body_list[1]
        android_id_to_data_body[idx] = id_title_body_list[2]

    # vectorizer = CountVectorizer(binary=True, analyzer='word', token_pattern='[^\s]+[a-z]*[0-9]*')
    vectorizer = CountVectorizer(binary=True, analyzer='word', max_df=0.2)

    vectorizer.fit(all_txt)

    return {
            'word_to_id': vectorizer.vocabulary_,
            'android_id_to_data_title': android_id_to_data_title,
            'android_id_to_data_body': android_id_to_data_body,
            'vectorizer': vectorizer
            }


''' Data Sets '''
processed_corpus = process_only_android_corpus()
android_id_to_data_title = processed_corpus['android_id_to_data_title']
android_id_to_data_body = processed_corpus['android_id_to_data_body']
vectorizer = processed_corpus['vectorizer']

test_data_android = android_id_to_similar_different(dev=False)
test_question_ids_android = list(test_data_android.keys())



auc_scorer = AUCMeter()

''' Begin Evaluation'''
candidate_ids, q_main_ids, labels = organize_test_ids(test_question_ids_android, test_data_android)
list_of_scores = []

index_into_list_all_all_candidate_ids = 0
q_main_id_ind = 0
for q_main_id in q_main_ids:
    if index_into_list_all_all_candidate_ids % 1000 == 0:  
        print(index_into_list_all_all_candidate_ids, end = ' ')
    q_main_sentence_title = android_id_to_data_title[q_main_id]
    q_main_sentence_body = android_id_to_data_body[q_main_id]
    q_main_vector_title = torch.from_numpy((vectorizer.transform([q_main_sentence_title]).toarray()[0])).float().unsqueeze(0)
    q_main_vector_body = torch.from_numpy((vectorizer.transform([q_main_sentence_body]).toarray()[0])).float().unsqueeze(0)
    q_main_vector = (q_main_vector_title + q_main_vector_body)/2

    q_candidate_id = candidate_ids[index_into_list_all_all_candidate_ids]
    q_candidate_sentence_title = android_id_to_data_title[q_candidate_id]
    q_candidate_sentence_body = android_id_to_data_body[q_candidate_id]
    q_candidate_vector_title = torch.from_numpy((vectorizer.transform([q_candidate_sentence_title]).toarray()[0])).float().unsqueeze(0)
    q_candiate_vector_body = torch.from_numpy((vectorizer.transform([q_candidate_sentence_body]).toarray()[0])).float().unsqueeze(0)
    q_candidate_vector = (q_candidate_vector_title + q_candiate_vector_body)/2

    score_cos_sim = Variable(torch.nn.functional.cosine_similarity(q_candidate_vector, q_main_vector)).data[0]
    list_of_scores.append(score_cos_sim)

    index_into_list_all_all_candidate_ids += 1
        

target = torch.FloatTensor(labels)
auc_scorer.reset()
auc_scorer.add(torch.FloatTensor(list_of_scores), target)
auc_score = auc_scorer.value()

print("AUC Score using TF-IDF:", auc_score)