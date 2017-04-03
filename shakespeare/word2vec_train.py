import gensim

class SentenceFile:
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            for line in f.readlines():
                yield line.split(' ')

sf = SentenceFile("shakes_out.txt")
model = gensim.models.Word2Vec(sf, min_count=10, size=100, workers=4)
model.save('w2vmodel')
print("king-queen similarity: " + str(model.similarity('king', 'queen')))
print("king-island similarity: " + str(model.similarity('king', 'island')))
