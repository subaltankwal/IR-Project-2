import pysolr
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
import math

solr = pysolr.Solr('http://localhost:8983/solr/localDocs', timeout=10)

def escape_special_characters(query):
    punctuations = '''!()-[]{};:'"\,<>./?#%^*_~'''
    for x in query:
        if x in punctuations:
            query = query.replace(x , "")
    return query

def preprocess_text(text):
    if isinstance(text, list):
        text = ' '.join(text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return tokens



def calculate_bm25(query, doc, doc_len, avg_doc_len, doc_freq, N):
    k1 = 1.5
    b = 0.75
    K = k1 * ((1 - b) + b * (doc_len / avg_doc_len))
    score = 0
    for term in query:
        if term in doc_freq:
            idf = math.log((N - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5) + 1)
            tf = doc.count(term)
            score += idf * ((tf * (k1 + 1)) / (tf + K))
    return score

def calculate_language_model(query, doc, doc_len, collection_freq, mu):
    score = 0
    for term in query:
        if term in collection_freq:
            tf = doc.count(term)
            cf = collection_freq[term]
            p_lm = (1 - mu) * (tf / doc_len) + mu * (cf / collection_len)
            if score == 0:
                score += p_lm
            else:
                score *= p_lm
                
    return score

def search_with_bm25(query):
    escaped_query = escape_special_characters(query)
    results = solr.search(f"Title:{escaped_query}" , rows= 200)
    
    if len(results) == 0:
        return []

    query_tokens = preprocess_text(query)

    avg_doc_len = sum(len(doc['Title']) for doc in results) / len(results)

    doc_freq = FreqDist([token for doc in results for token in preprocess_text(doc['Title'])])
    N = len(results)

    ranked_results = []
    for doc in results:
        doc_text = preprocess_text(doc['Title'])
        doc_len = len(doc_text)
        bm25_score = calculate_bm25(query_tokens, doc_text, doc_len, avg_doc_len, doc_freq, N)
        ranked_results.append((doc['id'], bm25_score))

    ranked_results.sort(key=lambda x: x[1], reverse=True)

    return ranked_results[:10]

def search_with_language_model(query):
    escaped_query = escape_special_characters(query)
    results = solr.search(f"Title:{escaped_query}" , rows= 200)

    query_tokens = preprocess_text(query)

    collection_freq = FreqDist(token for doc in results for token in preprocess_text(doc['Title']))
    global collection_len
    collection_len = sum(collection_freq.values())
    ranked_results = []
    for doc in results:
        doc_text = preprocess_text(doc['Title'])
        doc_len = len(doc_text)
        lm_score = calculate_language_model(query_tokens, doc_text, doc_len, collection_freq, mu=0.7)
        ranked_results.append((doc['id'], lm_score))

    ranked_results.sort(key=lambda x: x[1], reverse=True)

    return ranked_results[:10]

with open ('nfcorpus/train.titles.queries' , 'r' , encoding='utf-8') as file:
    for l in file:
        line = l.split('\t')
        query = line[1]
        print(line[0])
        print()
        bm25_results = search_with_bm25(query)
        print("BM25 Results:")
        for rank, (doc_id, score) in enumerate(bm25_results, start=1):
            print(f"Rank {rank}: Document ID: {doc_id}, BM25 Score: {score}")

        lm_results = search_with_language_model(query)
        print("\nLanguage Model Results:")
        for rank, (doc_id, score) in enumerate(lm_results, start=1):
            print(f"Rank {rank}: Document ID: {doc_id}, Language Model Score: {score}")