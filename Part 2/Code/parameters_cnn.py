''' Params Dashboard '''

''' Procedural parameters '''
batch_size = 40
num_differing_questions = 20
num_epochs = 2


''' Model specs CNN '''
margin = 0.1
lr_cnn = 10**-3

input_size = 300
hidden_size = 500
kernel_size = 3
stride = 1
padding = 0
dilation = 1
groups = 1
bias = True
bidirectional = False


''' Model specs NN '''
lr_nn = -10**-4
lamb = 10**-6

input_size_nn = 2*hidden_size if bidirectional else hidden_size
first_hidden_size_nn = 300
second_hidden_size_nn = 150


''' Data processing specs '''
truncation_val_title = 15
truncation_val_body = 85
padding_idx = 0

glove_path = '../glove.840B.300d.txt'
android_corpus_path = '../android_dataset/corpus.tsv'
ubuntu_corpus_path = '../ubuntu_dataset/text_tokenized.txt'