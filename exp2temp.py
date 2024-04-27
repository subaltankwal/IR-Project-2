import os
from math import log10
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk import FreqDist
import pysolr
import numpy as np
import re
from collections import Counter
from nltk.corpus import stopwords

solr = pysolr.Solr('http://localhost:8983/solr/localDocs')
all_docs = set("MED-" + str(i) for i in range(1, 5372))


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return tokens


def get_term_frequency(text):
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
        abstract = result['Abstract']

    abstract = ' '.join(abstract)
    term_vector = get_term_frequency(abstract)
    return term_vector


term_frequency = {}
document_frequencies = {}


def getNNN():
    for doc in all_docs:
        term_frequency[doc] = get_document_vector_from_solr(doc)
    allDocs = solr.search('*:*', rows=1000000)
    for document in allDocs:
        words = preprocess_text(' '.join(document['Abstract']))
        freq_dist = FreqDist(words)
        for word in freq_dist.keys():
            if (word != "abstract" and word != "background" and word != ""):
                document_frequencies[word] = document_frequencies.get(
                    word, 0) + 1

    with open('nfcorpus/dev.vid-desc.queries', 'r', encoding='utf-8') as file:
        for l in file:
            line = l.split('\t')
            query = line[1]
            queryVector = get_term_frequency(query)
            score = {}
            print(f"Query : {query}")
            for term, freq in queryVector.items():
                for doc, docVector in term_frequency.items():
                    if term in docVector.keys():
                        if doc in score.keys():
                            score[doc] += freq*docVector[term]
                        else:
                            score[doc] = freq*docVector[term]

            sortedScore = dict(
                sorted(score.items(), key=lambda item: item[1], reverse=True))
            count = 0
            for term, freq in sortedScore.items():
                print(term, freq)
                count += 1
                if count == 10:
                    break
            break


def getNTN():
    normalizedDocFreq = {}
    for term, freq in document_frequencies.items():
        normalizedDocFreq[term] = freq*log10(len(all_docs)/freq)
    with open('nfcorpus/dev.vid-desc.queries', 'r', encoding='utf-8') as file:
        for l in file:
            line = l.split('\t')
            query = line[1]
            queryVector = get_term_frequency(query)
            score = {}
            print(f"Query : {query}")
            for term, freq in queryVector.items():
                for doc, docVector in term_frequency.items():
                    if term in docVector.keys():
                        if doc in score.keys():
                            score[doc] += freq*normalizedDocFreq[term]
                        else:
                            score[doc] = freq*normalizedDocFreq[term]

            sortedScore = dict(
                sorted(score.items(), key=lambda item: item[1], reverse=True))
            count = 0
            for term, freq in sortedScore.items():
                print(term, freq)
                count += 1
                if count == 10:
                    break
            break


def getNTC():
    normalizedDocFreq = {}
    for term, freq in document_frequencies.items():
        normalizedDocFreq[term] = freq*log10(len(all_docs)/freq)
    with open('nfcorpus/dev.vid-desc.queries', 'r', encoding='utf-8') as file:
        for l in file:
            line = l.split('\t')
            query = line[1]
            queryVector = get_term_frequency(query)
            score = {}
            print(f"Query : {query}")
            for term, freq in queryVector.items():
                for doc, docVector in term_frequency.items():
                    docV = {}
                    for dc, fr in docVector.items():
                        if dc in normalizedDocFreq.keys():
                            docV[dc] = fr*normalizedDocFreq[dc]
                    sum_of_squares = sum(
                        x ** 2 for x in list(docV.values()))
                    euclidean_norm = sum_of_squares ** 0.5
                    term_f = {}
                    for t, f in docV.items():
                        term_f[t] = f/euclidean_norm
                    if term in docV.keys():
                        if doc in score.keys():
                            score[doc] += freq*term_f[term]
                        else:
                            score[doc] = freq*term_f[term]

            sortedScore = dict(
                sorted(score.items(), key=lambda item: item[1], reverse=True))
            count = 0
            for term, freq in sortedScore.items():
                print(term, freq)
                count += 1
                if count == 10:
                    break
            break

    return list(sortedScore.keys())


def getNTCforexp3():
    for doc in all_docs:
        term_frequency[doc] = get_document_vector_from_solr(doc)
    allDocs = solr.search('*:*', rows=1000000)
    for document in allDocs:
        words = preprocess_text(' '.join(document['Abstract']))
        freq_dist = FreqDist(words)
        for word in freq_dist.keys():
            if (word != "abstract" and word != "background" and word != ""):
                document_frequencies[word] = document_frequencies.get(
                    word, 0) + 1
    normalizedDocFreq = {}
    for term, freq in document_frequencies.items():
        normalizedDocFreq[term] = freq*log10(len(all_docs)/freq)
    with open('nfcorpus/dev.vid-desc.queries', 'r', encoding='utf-8') as file:
        for l in file:
            line = l.split('\t')
            query = line[1]
            queryVector = get_term_frequency(query)
            score = {}
            for term, freq in queryVector.items():
                for doc, docVector in term_frequency.items():
                    docV = {}
                    for dc, fr in docVector.items():
                        if dc in normalizedDocFreq.keys():
                            docV[dc] = fr*normalizedDocFreq[dc]
                    sum_of_squares = sum(
                        x ** 2 for x in list(docV.values()))
                    euclidean_norm = sum_of_squares ** 0.5
                    term_f = {}
                    for t, f in docV.items():
                        term_f[t] = f/euclidean_norm
                    if term in docV.keys():
                        if doc in score.keys():
                            score[doc] += freq*term_f[term]
                        else:
                            score[doc] = freq*term_f[term]

            sortedScore = dict(
                sorted(score.items(), key=lambda item: item[1], reverse=True))
            break
    return list(sortedScore.keys())


if __name__ == '__main__':
    getNNN()
    getNTN()
    getNTC()
