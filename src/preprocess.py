from preprocessing.utils import (
    remove_HTML,
    remove_punctuation,
    keep_alpha_numerical,
    remove_space,
    remove_stopwords,
    apply_lemmetization,
)


def preprocess(sentence):
    result = remove_HTML(sentence)
    result = remove_punctuation(result)
    result = keep_alpha_numerical(result)
    result = remove_space(result)
    result = result.lower()
    result = remove_stopwords(result)
    result = apply_lemmetization(result)
    return result


def with_category_features(data):
    data_categories = set()

    for category in data['category']:
        data_categories = data_categories.union(category)

    data_categories = sorted(map(lambda x: x.strip().lower(), list(data_categories)))

    data['category'] = data['category'].apply(lambda x: [i.strip().lower() for i in x])

    for category in data_categories:
        data[category] = data['category'].apply(lambda x: 1 if category in x else 0)

    data = data.drop(['category'], axis=1)

    return data