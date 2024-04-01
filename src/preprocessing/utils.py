import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

lemmatizer = WordNetLemmatizer()

def remove_HTML(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def remove_punctuation(sentence):
    cleaned = re.sub(r'(\\)\w+', r'', sentence)
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',cleaned)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned


def keep_alpha_numerical(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-zA-Z0-9]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent

def remove_space(sentence):
    result = sentence.replace(r'\s+', ' ')
    return ' '.join(result.split())

def remove_stopwords(sentence):
    stop_words = set(stopwords.words('english'))
    re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
    return re_stop_words.sub(" ", sentence)

def apply_lemmetization(sentence):
    result = ""
    for word in sentence.split():
        lemma = lemmatizer.lemmatize(word)
        result += lemma
        result += " "
    result = result.strip()
    return result

def is_sentence_in_boundaries(sentence, min_tokens=10, max_tokens=512) -> bool:
    splitted =len(sentence.split()) 
    return splitted >= min_tokens and splitted <= max_tokens

def count_words(x):
    return len(x.split())

def get_most_common_words(data, n=15):
    words_counter = Counter()

    for tokens in data['description'].str.split():
        words_counter += Counter(tokens)

    words_counter_x, words_counter_y= [], []

    for word, count in words_counter.most_common(n):
        words_counter_x.append(word)
        words_counter_y.append(count)

    return words_counter_x, words_counter_y

def get_top_ngrams(data, n=None, k=15):
    vec = CountVectorizer(ngram_range=(n, n)).fit(data['description'])
    bag_of_words = vec.transform(data['description'])
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx])
                  for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:k]

# For testing
# if __name__ == '__main__':
#     sentence = 'A tokenization is the better best rocks test test test test asd asd asd cvbcvb cvbcvb xcxcbcvb'
#     result = remove_stopwords(sentence)
#     print(result)
#     result = apply_lemmetization(result)
#     print(result)
#     breakpoint()