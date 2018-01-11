from sklearn.feature_extraction.text import CountVectorizer

stop_word_frequency_float = 0.3

# Word mapped to its embedding vector
def get_words_and_embeddings():
    filepath = "../Data1/vectors_pruned.200.txt"
    lines = open(filepath).readlines()
    word2vec = {}
    for line in lines:
        word_coordinates_list = line.split(" ")
        word2vec[word_coordinates_list[0]] = word_coordinates_list[1:-1]
    return word2vec

# Question id mapped to the content: title + body text
def questionID_to_questionData():
    filepath = "../Data1/text_tokenized.txt"
    lines = open(filepath, encoding = 'utf8').readlines()

    id2Data = {}
    for line in lines:
        id_title_body_list = line.split('\t')
        title_text = id_title_body_list[1].split(" ")
        body_text = id_title_body_list[2].split(" ")
        id2Data[int(id_title_body_list[0])] = title_text + body_text
    return id2Data

def questionID_to_questionData_truncate(max_length_title, max_length_body):
    filepath = "../Data1/text_tokenized.txt"
    lines = open(filepath, encoding = 'utf8').readlines()

    # First, get a list of stop_words so that these are not put in the word list of a question
    # Need to go through a complete cycle in the text once here then once more below
    all_text = []
    for line in lines:
        id_title_body_list = line.split('\t')
        question_text = id_title_body_list[1] + ' ' + id_title_body_list[2]
        all_text.append(question_text)

    vectorizer = CountVectorizer(binary=True, analyzer='word', max_df=stop_word_frequency_float, token_pattern='[^\s]+[a-z]*[0-9]*')
    vectorizer.fit(all_text)
    stop_words = vectorizer.stop_words_

    id2Data = {}
    for line in lines:
        id_title_body_list = line.split('\t')
        title_text = id_title_body_list[1].split(" ")
        body_text = id_title_body_list[2].split(" ")
        #whole_text = title_text + body_text
        pruned_text_title = [word for word in title_text if word not in stop_words]
        pruned_text_body = [word for word in body_text if word not in stop_words]

        if len(pruned_text_title) >= max_length_title and len(pruned_text_body) >= max_length_body:
            id2Data[int(id_title_body_list[0])] = (pruned_text_title[:max_length_title], pruned_text_body[:max_length_body])
            
        elif len(pruned_text_title) >= max_length_title and len(pruned_text_body) < max_length_body:
            id2Data[int(id_title_body_list[0])] = (pruned_text_title[:max_length_title], pruned_text_body)
            
        elif len(pruned_text_title) < max_length_title and len(pruned_text_body) >= max_length_body:
            id2Data[int(id_title_body_list[0])] = (pruned_text_title, pruned_text_body[:max_length_body])
            
        else:
            id2Data[int(id_title_body_list[0])] = (pruned_text_title, pruned_text_body)
            
    return id2Data


# For training data, question id mapped to [[similar_questions_ids], [different_questions_ids]]
def training_id_to_similar_different():
    filepath = "../Data1/train_random.txt"
    lines = open(filepath, encoding = 'utf8').readlines()

    training_data = {}
    for line in lines:
        id_similarids_diffids = line.split('\t')
        question_id = int(id_similarids_diffids[0])
        similar_ids = id_similarids_diffids[1].split(" ")
        different_ids = id_similarids_diffids[2].split(" ")

        for i in range(len(similar_ids)): similar_ids[i] = int(similar_ids[i])
        for j in range(len(different_ids)): different_ids[j] = int(different_ids[j])

        training_data[question_id] = [ similar_ids, different_ids ]
    return training_data


# For test data, question id mapped to [[similar_questions_ids], [all_questions_ids]]
# For dev set, use dev=True, for test set use dev=False
# Note:
#   [similar_questions_ids] is a subset of [all_questions_ids]
def devTest_id_to_similar_different(dev=True):
    filepath = "../Data1/dev.txt" if dev else "../Data1/test.txt"
    lines = open(filepath, encoding = 'utf8').readlines()

    evaluation_data = {}
    for line in lines:
        id_similarids_diffids = line.split('\t')
        question_id = int(id_similarids_diffids[0])
        similar_ids = id_similarids_diffids[1].split(" ") if id_similarids_diffids[1] != '' else []
        different_ids = id_similarids_diffids[2].split(" ")
        for i in range(len(similar_ids)): similar_ids[i] = int(similar_ids[i])
        for j in range(len(different_ids)): different_ids[j] = int(different_ids[j])
        evaluation_data[question_id] = [ similar_ids, different_ids ]
    return evaluation_data

# word2vec = get_words_and_embeddings()
# id2Data = questionID_to_questionData()
# training_data = training_id_to_similar_different()
# dev_data = devTest_id_to_similar_different(dev=True)
# test_data = devTest_id_to_similar_different(dev=False)