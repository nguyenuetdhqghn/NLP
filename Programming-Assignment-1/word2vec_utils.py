import numpy as np

def load_word2vec(filepath):
    word_vectors = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        vocab_size = int(f.readline().strip())
        dim = int(f.readline().strip())
        print(f"File khai báo: {vocab_size} từ, {dim} chiều")

        for line in f:
            parts = line.strip().split()
            if len(parts) < dim + 1:
                continue
            word = parts[0]
            try:
                vector = np.array(parts[1:dim+1], dtype=float)
                word_vectors[word] = vector
            except ValueError:
                continue
    return word_vectors

def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)