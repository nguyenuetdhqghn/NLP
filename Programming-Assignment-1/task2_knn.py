import numpy as np
from word2vec_utils import load_word2vec, cosine_similarity
from result_utils import append_result

WORD2VEC_PATH = "word2vec/W2V_150.txt"


def k_nearest_words(word, word_vectors, k=5, logger=print):
    if word not in word_vectors:
        logger(f"Từ '{word}' không có trong từ điển")
        return []

    target_vec = word_vectors[word]
    words = list(word_vectors.keys())
    vectors = np.array([word_vectors[w] for w in words])

    norms = np.linalg.norm(vectors, axis=1)
    target_norm = np.linalg.norm(target_vec)
    similarities = vectors.dot(target_vec) / (norms * target_norm + 1e-10)

    idx_sorted = np.argsort(-similarities)
    result = []
    for idx in idx_sorted:
        if words[idx] != word:
            result.append((words[idx], similarities[idx]))
        if len(result) == k:
            break
    return result


def main():
    output_lines = []

    def log(message):
        print(message)
        output_lines.append(str(message))

    log("Load word2vec")
    word_vectors = load_word2vec(WORD2VEC_PATH)
    log(f"Đã load {len(word_vectors)} từ\n")

    test_words = ["sinh_viên", "học_sinh", "đẹp"]
    for w in test_words:
        log(f"Top 5 từ gần nhất với '{w}':")
        neighbors = k_nearest_words(w, word_vectors, k=5, logger=log)
        for word, sim in neighbors:
            log(f"  {word}: {sim:.4f}")
        log("")

    append_result("Task 2 - KNN", "\n".join(output_lines))


if __name__ == "__main__":
    main()