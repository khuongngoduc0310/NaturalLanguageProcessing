import pandas as pd
import os
import math

file_path = r".\Test\Test_Full\Khoa hoc"
save_path = r".\tfidf_save"

special_chars = r"""' … " - . ’ , & : “ ” • { } ? ( ) [ ] ~ @ # $ % ^ * < > / \ !  0 1 2 3 4 5 6 7 8 9"""
special_chars = special_chars.split(' ')

VOCAB = dict()


def getAllVocab(D):
    vocab = dict()
    for text in D:
        text = text.split('\r')
        data = [p.replace('\n', '').split('.') for p in text]
        for dt in data:
            txt = str(dt).lower().strip()
            for char in special_chars:
                if char in txt:
                    txt = txt.replace(char, '')
            words = txt.split(' ')
            for word in words:
                if len(word) > 0:
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
    return vocab


def getVocab(text):
    vocab = dict()
    text = text.split('\r')
    data = [p.replace('\n', '').split('.') for p in text]
    for dt in data:
        txt = str(dt).lower().strip()
        for char in special_chars:
            if char in txt:
                txt = txt.replace(char, '')
        words = txt.split(' ')
        for word in words:
            if len(word) > 0:
                if word in vocab:
                    vocab[word] += 1
                else:
                    vocab[word] = 1
    return vocab


def tf(t, d):
    vocab = getVocab(d)
    return float(vocab[t]) / max(vocab.values())


def idf(t, D):
    count = 1
    for d in D:
        vocab = getVocab(d)
        if t in vocab:
            count += vocab[t]
    # print(math.log10(len(D)/(getAllVocab(D)[t]+1)))
    return math.log10(len(D) / count)


def tf_idf(t, d, D):
    VOCAB = getAllVocab(D)
    if t not in VOCAB:
        VOCAB[t] = 0
    return tf(t, d) * math.log10(len(D) / (VOCAB[t] + 1))


def tf_idf_all(D):
    tf_idf_values = []
    VOCAB = getAllVocab(D)
    index = 0
    for d in D:
        vocab_d = getVocab(d)
        max_d = max(vocab_d.values())
        tf_idf_values.append(dict())
        for t in vocab_d.keys():
            if len(t) > 0:
                tf_t = float(vocab_d[t]) / max_d
                idf_t = math.log10(len(D) / vocab_d[t])
                tf_idf_values[index][t] = tf_t * idf_t
        index += 1
    return tf_idf_values


def save_to_csv(label, data):
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(f"{save_path}/{label}.csv")


def read_data_from_dir(dir):
    data = []
    file_paths = os.listdir(dir)
    print(len(file_paths))
    if file_paths and len(file_paths):
        for file in file_paths:
            with open(f"{dir}/{file}", mode='rb') as f:
                text = f.read()
                data.append(text.decode("utf16").strip())
    return data


if __name__ == "__main__":
    df = read_data_from_dir(file_path)
    save_to_csv("TF_IDF", tf_idf_all(df))
