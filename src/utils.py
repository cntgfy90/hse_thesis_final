import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    hamming_loss,
)
import gensim.downloader as api
from sklearn.model_selection import train_test_split


def accuracy_ml_score(y_true, y_pred):
    total_obs, n_classes = y_true.shape
    correct_preds = np.sum(y_pred.T == y_true.T, axis=1)
    correct_preds = correct_preds / total_obs
    return np.sum(correct_preds) / n_classes


def get_accuracy(outputs, targets, full_results=False):
    if full_results:
        result = 0
        outputs
        return result

    correct_predictions = np.sum(outputs == targets)
    num_samples = targets.size
    return float(correct_predictions) / num_samples


def tfidf_vectorize(train_data, test_data):
    tfidf = TfidfVectorizer()
    tfidf_train_data = tfidf.fit_transform(train_data)
    tfidf_test_data = tfidf.transform(test_data)
    return tfidf_train_data, tfidf_test_data


def w2v_vectorize(data):
    def to_w2v_embedding(sentence):
        embeddings = []

        for word in sentence.split():
            if word in wv:
                embeddings.append(wv[word])
        embeddings = np.array(embeddings)
        return np.mean(embeddings, axis=0)

    wv = api.load("word2vec-google-news-300")

    return data.apply(to_w2v_embedding)


def display_metrics(y_true, y_pred):
    print(f"Accuracy (subset): {accuracy_score(y_true, y_pred)}")
    print(f"Accuracy (ML): {accuracy_ml_score(y_true, y_pred)}")
    print(f"Precision (macro): {precision_score(y_true, y_pred, average='macro')}")
    print(f"Precision (micro): {precision_score(y_true, y_pred, average='micro')}")
    print(f"Recall (macro): {recall_score(y_true, y_pred, average='macro')}")
    print(f"Recall (micro): {recall_score(y_true, y_pred, average='micro')}")
    print(f"Hamming loss: {hamming_loss(y_true, y_pred)}")

def get_splitted_dataset(data):
    categories = [column for column in data.columns if column != 'description']

    X_train, X_test, y_train, y_test = train_test_split(
        data['description'],
        data[categories],
        test_size=0.2,
        random_state=13
    )

    X_train = X_train.reset_index()
    X_train = X_train.drop(['index'], axis=1)

    X_test = X_test.reset_index()
    X_test = X_test.drop(['index'], axis=1)

    y_train = y_train.reset_index()
    y_train = y_train.drop(['index'], axis=1)

    y_test = y_test.reset_index()
    y_test = y_test.drop(['index'], axis=1)

    return X_train, X_test, y_train, y_test
