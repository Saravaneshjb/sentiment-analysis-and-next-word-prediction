import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import re
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.tokenize import regexp_tokenize
import contractions


# Download necessary resources
nltk.download('stopwords')
nltk.download('punkt')

## Methods for preprocessing
def expand_contractions(text):
    return contractions.fix(text)

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    # print("text after the user mentions removal : ",text)
    #Remove all the special characters except for apostrophes
    text = re.sub(r'[^a-zA-Z0-9\'\s]', '', text)
    # Expand contractions
    text = expand_contractions(text)
    # print("Text after contraction expansion :",text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # print("Text after URL removal : ",text)
    # Custom tokenization to keep contractions intact
    pattern = r"\b\w+'\w+|\b\w+|[^\w\s]"
    tokens = regexp_tokenize(text, pattern)
    # print("Tokens after custom tokenization:", tokens)
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    #Remove special characters
    tokens=[word for word in tokens if word.isalnum()]
    # print("text after stop word removal :", tokens)

    return ' '.join(tokens)



def stem_text(text,stemmer):
    tokens = word_tokenize(text)
    return ' '.join([stemmer.stem(token) for token in tokens])

def lemmatize_text(text,lemmatizer):
    tokens = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens])