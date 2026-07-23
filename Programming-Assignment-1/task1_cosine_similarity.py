from scipy.stats import spearmanr
from word2vec_utils import load_word2vec, cosine_similarity
from result_utils import append_result

WORD2VEC_PATH = "word2vec/W2V_150.txt"
VISIM_PATH = "datasets/ViSim-400/Visim-400.txt"


def load_visim400(filepath):
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 6:
                continue
            word1, word2, pos, sim1, sim2, std = parts
            pairs.append((word1, word2, float(sim1), float(sim2)))
    return pairs


def main():
    output_lines = []

    def log(message):
        print(message)
        output_lines.append(str(message))

    log("Load word2vec")
    word_vectors = load_word2vec(WORD2VEC_PATH)
    log(f"Đã load {len(word_vectors)} từ")

    pairs = load_visim400(VISIM_PATH)
    log(f"Đã load {len(pairs)} cặp từ ViSim-400")

    model_scores, human_sim1, human_sim2 = [], [], []
    missing = 0

    for word1, word2, sim1, sim2 in pairs:
        if word1 in word_vectors and word2 in word_vectors:
            sim = cosine_similarity(word_vectors[word1], word_vectors[word2])
            model_scores.append(sim)
            human_sim1.append(sim1)
            human_sim2.append(sim2)
        else:
            missing += 1

    log(f"Số cặp bị thiếu từ trong từ điển: {missing}/{len(pairs)}")

    corr1, _ = spearmanr(model_scores, human_sim1)
    corr2, _ = spearmanr(model_scores, human_sim2)
    log(f"Spearman correlation với Sim1: {corr1:.4f}")
    log(f"Spearman correlation với Sim2: {corr2:.4f}")

    append_result("Task 1 - Cosine Similarity", "\n".join(output_lines))


if __name__ == "__main__":
    main()