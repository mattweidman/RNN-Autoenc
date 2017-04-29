from text_format import DataSet

def get_vocab_size(filename, min_count):
    dataset = DataSet(filename, 100, min_count=min_count)
    return len(dataset.word_list)

if __name__ == "__main__":
    filename = "../shakespeare/shakes_out.txt"
    min_counts = [1, 5, 10, 50, 100]
    for min_count in min_counts:
        vocab_size = get_vocab_size(filename, min_count)
        print(min_count, vocab_size)
