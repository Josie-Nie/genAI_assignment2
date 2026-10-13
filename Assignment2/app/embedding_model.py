import spacy

nlp = spacy.load("en_core_web_lg")


def calculate_embedding(input_word):
    """Return the spaCy word vector following Module 2 Practical 3."""
    word = nlp(input_word)
    return word.vector
