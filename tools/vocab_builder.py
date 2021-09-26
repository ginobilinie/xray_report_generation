import re
import glob
import sentencepiece as spm

def get_report(file_name):
    f = open(file_name, 'r')
    return f.read()

def preprocess(text):
    # Replace words
    text = re.sub('[^a-zA-Z\s]', '', text)
    # Convert to lower case
    text = text.lower()
    # Remove white space
    text = ' '.join(text.split())
    return text

# Hyper-parameters
word_size = 4500 # Most frequent words
total_size = 5000 # total_size = word_size + others
dataset_dir = '/home/hoang/Datasets/MIMIC/'
file_list = glob.glob(dataset_dir + 'files/**/*.txt', recursive=True)

# Compute covering ratio
word_freq = dict()
for file_name in file_list:
    file_text = preprocess(get_report(file_name))
    for token in file_text.split():
        if token not in word_freq:
            word_freq[token] = 1
        else:
            word_freq[token]+= 1

sorted_word_freq = sorted([(k,v) for k,v in word_freq.items()], key=lambda x: x[1], reverse=True)

total_filter = 0
for k,v in sorted_word_freq[:word_size]:
    total_filter += v

total_all = 0
for k,v in sorted_word_freq[:]:
    total_all += v

print('Covering', total_filter / total_all)

# Set up the vocabulary model
sorted_word_freq = sorted([(k,v) for k,v in word_freq.items()], key=lambda x: x[1], reverse=True)[:word_size]
word_list = [k for k,v in sorted_word_freq]
punc_list = ['`','~','!','@','#','$','%','^','&','*','-','_','+','=',
             '\\','|',':',';','"','\'',',','.','?','/',
             '(',')','{','}','[',']','<','>']

# Save the vocabulary model
mode_type = 'unigram'
spm.SentencePieceTrainer.train(
    input=file_list,
    model_prefix=dataset_dir + 'mimic_{}_{}'.format(mode_type,total_size), 
    vocab_size=total_size, 
    model_type=mode_type, 
    unk_id=0, bos_id=1, eos_id=2, pad_id=3,
    user_defined_symbols=punc_list + word_list,
)

# Load the vocabulary model
vocab = spm.SentencePieceProcessor(model_file=dataset_dir + 'mimic_{}_{}.model'.format(mode_type,total_size))
data = vocab.encode(preprocess(get_report(file_list[0])), out_type=int)
print(vocab.id_to_piece(data))