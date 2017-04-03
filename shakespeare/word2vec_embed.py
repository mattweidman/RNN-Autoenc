import gensim
import numpy as np

# get text
text_fname = "shakes_out.txt"
with open(text_fname, 'r') as f:
    lines = f.readlines()

# find length of longest line
longest_line = ""
for line in lines:
    if len(line.split(' ')) > len(longest_line.split(' ')):
        longest_line = line
line_len = len(longest_line.split(' '))

# number of lines
num_lines = len(lines)

# size of word2vec embeddings
embed_size = 100

# create numpy array
text_tensor = np.zeros((num_lines, line_len, embed_size))

# get word2vec model
model_fname = "w2vmodel"
model = gensim.models.Word2Vec.load(model_fname)

## write word2vec vectors in numpy array
# commented out because numpy array is 5.6 GB
#for i in range(num_lines):
#    line = lines[i].split(' ')
#    for j in range(line_len):
#        if j < len(line) and line[j] in model:
#            word = line[j]
#            text_tensor[i,j,:] = model[word]
#        else:
#            text_tensor[i,j,:] = model['']

## convert some vectors back to words
#test_line_num = 10
#sentence_arr = []
#for vec in text_tensor[test_line_num]:
#    word = model.similar_by_vector(vec, topn=1)
#    sentence_arr.append(word)
#sentence = ' '.join([tup[0][0] for tup in sentence_arr])
#print(sentence)

# save numpy array
#np.save("shakes.npy", text_tensor)
