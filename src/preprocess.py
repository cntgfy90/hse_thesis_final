# import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
from preprocessing.utils import (
    remove_HTML,
    remove_punctuation,
    keep_alpha_numerical,
    remove_space,
    remove_stopwords,
    apply_lemmetization,
)
# lemmatizer = WordNetLemmatizer()

# STOPWORDS = set(stopwords.words('english'))

# First "preprocess" version is not needed
# TODO: remove
# def preprocess(text):
#     result = str(text)
#     result = result.replace(r'[^a-zA-Z0-9]', ' ')
#     result = result.replace(r'\s+', ' ')
#     result = result.lower()
#     result = result.split()
#     result = filter(lambda x: x not in STOPWORDS, result)
#     result = map(lambda x: lemmatizer.lemmatize(x), result)
#     return ' '.join(result)

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