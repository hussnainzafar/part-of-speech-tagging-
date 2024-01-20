import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# Specify the custom path for NLTK data
# Download NLTK data (if not already downloaded)
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


def nltk_pos_tagging(text):
    # Tokenize the text into words
    words = word_tokenize(text)

    # Perform POS tagging
    pos_tags = pos_tag(words)

    return pos_tags


