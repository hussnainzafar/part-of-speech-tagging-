import spacy

# Load the English language model
nlp = spacy.load('en_core_web_sm')

def spacy_pos_tagging(text):
    # Process the text
    doc = nlp(text)

    # Extract POS tags
    pos_tags = [(token.text, token.pos_) for token in doc]

    return pos_tags


