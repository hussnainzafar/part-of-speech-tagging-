import stanfordnlp

# Download the English language model (you only need to do this once)
stanfordnlp.download('en')


def stanfordnlp_pos_tagging(text):
    # Load the English language model
    nlp = stanfordnlp.Pipeline(processors='tokenize,pos', lang='en')

    # Process the text
    doc = nlp(text)

    # Extract POS tags
    pos_tags = [(word.text, word.pos) for sent in doc.sentences for word in sent.words]

    return pos_tags

