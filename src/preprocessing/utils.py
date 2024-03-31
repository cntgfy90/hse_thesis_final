import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def remove_HTML(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext


def remove_punctuation(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned


def keep_alpha_numerical(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
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

def extract_sentence_boundaries(sentence, min_tokens=10, max_tokens=512):
    return setnce

# For testing
# if __name__ == '__main__':
#     sentence = 'A tokenization is the better best rocks'
#     result = remove_stopwords(sentence)
#     print(result)
#     result = apply_lemmetization(result)
#     print(result)