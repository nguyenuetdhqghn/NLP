import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.utils import resample
from word2vec_utils import load_word2vec
from result_utils import append_result

WORD2VEC_PATH = "word2vec/W2V_150.txt"
TRAIN_DIR = "antonym-synonym set"
VICON_DIR = "datasets/ViCon-400"
VICON_FILES = ["400_noun_pairs.txt", "400_verb_pairs.txt", "600_adj_pairs.txt"]


def load_pairs(filepath, label):
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1], label))
    return pairs


def load_vicon_file(filepath):
    pairs = []
    with open(filepath, 'r', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            word1, word2, relation = parts[0], parts[1], parts[2]
            if relation.upper() not in ("SYN", "ANT"):
                continue
            label = 1 if relation.upper() == "SYN" else 0
            pairs.append((word1, word2, label))
    return pairs


def load_vicon400(vicon_dir, files, logger=print):
    all_pairs = []
    for filename in files:
        filepath = os.path.join(vicon_dir, filename)
        file_pairs = load_vicon_file(filepath)
        logger(f"  {filename}: {len(file_pairs)} cặp")
        all_pairs.extend(file_pairs)
    return all_pairs


def remove_overlap(train_pairs, test_pairs, logger=print):
    test_set = set()
    for w1, w2, _ in test_pairs:
        test_set.add((w1, w2))
        test_set.add((w2, w1))
    filtered = [p for p in train_pairs if (p[0], p[1]) not in test_set]
    logger(f"Train sau khi loại overlap: {len(filtered)}/{len(train_pairs)}")
    return filtered


def get_features(word1, word2, word_vectors, dim=150):
    v1 = word_vectors.get(word1, np.zeros(dim))
    v2 = word_vectors.get(word2, np.zeros(dim))
    return np.concatenate([v1, v2, np.abs(v1 - v2)])


def build_dataset(pairs, word_vectors, logger=print):
    X, y = [], []
    skipped = 0
    for word1, word2, label in pairs:
        if word1 in word_vectors and word2 in word_vectors:
            X.append(get_features(word1, word2, word_vectors))
            y.append(label)
        else:
            skipped += 1
    logger(f"Skipped {skipped} pairs due to missing embeddings")
    return np.array(X), np.array(y), skipped


def balance_dataset(X, y):
    X_syn = X[y == 1]
    X_ant = X[y == 0]
    if len(X_syn) > len(X_ant):
        X_ant_up = resample(X_ant, replace=True, n_samples=len(X_syn), random_state=42)
        X_balanced = np.vstack([X_syn, X_ant_up])
        y_balanced = np.array([1] * len(X_syn) + [0] * len(X_ant_up))
    else:
        X_syn_up = resample(X_syn, replace=True, n_samples=len(X_ant), random_state=42)
        X_balanced = np.vstack([X_syn_up, X_ant])
        y_balanced = np.array([1] * len(X_syn_up) + [0] * len(X_ant))
    return X_balanced, y_balanced


def main():
    output_lines = []

    def log(message):
        print(message)
        output_lines.append(str(message))

    log("Load word2vec")
    word_vectors = load_word2vec(WORD2VEC_PATH)
    log(f"Đã load {len(word_vectors)} từ\n")

    syn_pairs = load_pairs(os.path.join(TRAIN_DIR, "Synonym_vietnamese.txt"), 1)
    ant_pairs = load_pairs(os.path.join(TRAIN_DIR, "Antonym_vietnamese.txt"), 0)
    train_pairs = syn_pairs + ant_pairs
    log(f"Train (trước khi lọc): {len(syn_pairs)} cặp SYN, {len(ant_pairs)} cặp ANT")

    log("Load test set ViCon-400")
    test_pairs = load_vicon400(VICON_DIR, VICON_FILES, logger=log)

    train_pairs = remove_overlap(train_pairs, test_pairs, logger=log)

    X_train, y_train, skipped_train = build_dataset(train_pairs, word_vectors, logger=log)
    log(f"Train set sau xử lý: {len(X_train)} mẫu, bỏ qua {skipped_train} cặp thiếu từ")
    n_syn = int((y_train == 1).sum())
    n_ant = int((y_train == 0).sum())
    log(f"  Phân bố lớp: {n_syn} SYN / {n_ant} ANT\n")

    X_test, y_test, skipped_test = build_dataset(test_pairs, word_vectors, logger=log)
    log(f"Test set sau xử lý: {len(X_test)} mẫu, bỏ qua {skipped_test} cặp thiếu từ\n")

    log("Logistic Regression (mặc định)")
    clf_lr = LogisticRegression(max_iter=1000)
    clf_lr.fit(X_train, y_train)
    y_pred_lr = clf_lr.predict(X_test)
    log(classification_report(y_test, y_pred_lr, target_names=["ANT", "SYN"]))

    log("Logistic Regression (class_weight='balanced')")
    clf_lr_bal = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf_lr_bal.fit(X_train, y_train)
    y_pred_lr_bal = clf_lr_bal.predict(X_test)
    log(classification_report(y_test, y_pred_lr_bal, target_names=["ANT", "SYN"]))

    log("MLP Classifier (dữ liệu gốc, chưa balance)")
    clf_mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
    clf_mlp.fit(X_train, y_train)
    y_pred_mlp = clf_mlp.predict(X_test)
    log(classification_report(y_test, y_pred_mlp, target_names=["ANT", "SYN"]))

    log("MLP Classifier (sau khi oversample cho cân bằng)")
    X_train_bal, y_train_bal = balance_dataset(X_train, y_train)
    log(f"Sau khi balance: {len(X_train_bal)} mẫu ({(y_train_bal==1).sum()} SYN / {(y_train_bal==0).sum()} ANT)\n")
    clf_mlp_bal = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42)
    clf_mlp_bal.fit(X_train_bal, y_train_bal)
    y_pred_mlp_bal = clf_mlp_bal.predict(X_test)
    log(classification_report(y_test, y_pred_mlp_bal, target_names=["ANT", "SYN"]))

    append_result("Task 3 - Classifier", "\n".join(output_lines))


if __name__ == "__main__":
    main()