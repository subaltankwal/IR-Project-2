import pysolr
import numpy as np
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from exp2temp import getNTCforexp3

solr = pysolr.Solr('http://localhost:8983/solr/localDocs')
all_docs = set("MED-" + str(i) for i in range(1, 5372))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return tokens


def term_frequency(text):
    tokens = preprocess_text(text)
    term_freq = Counter(tokens)
    return term_freq


def get_document_vector_from_solr(doc_id):
    """
    Retrieves the document vector from Solr based on the document ID.

    Args:
        doc_id (str): The ID of the document.

    Returns:
        list: The document vector.
    """
    results = solr.search(f"id:{doc_id}")
    for result in results:
        title = result['Title']

    title = ' '.join(title)
    term_vector = term_frequency(title)
    return term_vector


def rocchio_algorithm(original_query_vector, relevant_docs, non_relevant_docs, alpha=1.0, beta=0.75, gamma=0.25):
    """
    Implements the Rocchio algorithm for pseudo feedback.

    Args:
        original_query_vector (dict): The original query vector (word frequencies).
        relevant_docs (list): List of relevant document IDs.
        non_relevant_docs (list): List of non-relevant document IDs.
        alpha (float): Weight of the original query vector.
        beta (float): Weight of the relevant documents.
        gamma (float): Weight of the non-relevant documents.

    Returns:
        dict: The updated query vector (word frequencies).
    """
    centroid_relevant = Counter()
    centroid_non_relevant = Counter()

    for doc_id in relevant_docs:
        doc_vector = get_document_vector_from_solr(doc_id)

        if doc_vector:
            centroid_relevant += doc_vector
    for doc_id in non_relevant_docs:
        doc_vector = get_document_vector_from_solr(doc_id)
        if doc_vector:
            centroid_non_relevant += doc_vector
    num_relevant_docs = len(relevant_docs)
    num_non_relevant_docs = len(non_relevant_docs)
    if num_relevant_docs > 0:
        centroid_relevant = {
            word: freq / num_relevant_docs for word, freq in centroid_relevant.items()}
    if num_non_relevant_docs > 0:
        centroid_non_relevant = {
            word: freq / num_non_relevant_docs for word, freq in centroid_non_relevant.items()}

    updated_query_vector = {}
    for word, freq in original_query_vector.items():
        updated_query_vector[word] = alpha*freq
    for word, freq in centroid_relevant.items():
        updated_query_vector[word] = updated_query_vector.get(
            word, 0) + beta*freq
    for word, freq in centroid_non_relevant.items():
        updated_query_vector[word] = updated_query_vector.get(
            word, 0) - gamma*freq

    updated_query_vector = {word: freq for word,
                            freq in updated_query_vector.items() if freq >= 0.05 and word != 'pubmed' and word != 'ncbi'}
    return updated_query_vector


with open('nfcorpus/train.titles.queries', 'r', encoding='utf-8') as file:
    for l in file:
        line = l.split('\t')
        relevant_docs = getNTCforexp3()[:79]
        non_relevant_docs = all_docs
        for doc in relevant_docs:
            if doc in non_relevant_docs:
                non_relevant_docs.remove(doc)
        updated_query_vector = rocchio_algorithm(
            term_frequency(line[1]), relevant_docs, non_relevant_docs)
        print(line[1])
        print(updated_query_vector)
        print()
        print()
        # break
