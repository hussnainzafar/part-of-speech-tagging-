
from NLTK_taging import nltk_pos_tagging
from Stanford_NLP_tagging import stanfordnlp_pos_tagging
from spacy_tagging import spacy_pos_tagging




def main():
    
    
    with open('./simple_text.txt', 'r') as file:
        text = file.read()
   
    print("\n\n\nstanford tagging : ", stanfordnlp_pos_tagging(text))
    print("\n\n\nspacy tagging : ", spacy_pos_tagging(text))
    print("\n\n\nNLTK tagging : ",nltk_pos_tagging(text))


if __name__ == "__main__":
    main()
