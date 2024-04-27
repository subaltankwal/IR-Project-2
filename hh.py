import pysolr
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from itertools import chain

# def calculate_cosine_similarity(query_vector, document_vectors):
#     # Calculate cosine similarity between query and documents
#     cosine_similarities = cosine_similarity(query_vector, document_vectors)

#     return cosine_similarities[0]
# def cosines(query , field):
#     solr = pysolr.Solr('http://localhost:8983/solr/localDocs', timeout=10)

#     results = solr.search(f'{field}:{query}', rows=100)
#     document_ids = [doc['id'] for doc in results]
#     document_texts = [doc[field] for doc in results]

#     vectorizer = TfidfVectorizer(stop_words='english')
#     document_texts = list(chain.from_iterable(document_texts))
#     query_text = query
#     all_texts = [query_text] + document_texts
    
#     tfidf_matrix = vectorizer.fit_transform(all_texts)

#     query_vector = tfidf_matrix[0:1]
#     document_vectors = tfidf_matrix[1:]

#     cosine_similarities = calculate_cosine_similarity(query_vector, document_vectors)

#     cosine_scores_with_ids = dict(zip(document_ids, cosine_similarities))
#     return cosine_scores_with_ids

# d = cosines("dark chocolate" , "Abstract")

# print(d)
# print(len(d))

# solr = pysolr.Solr('http://localhost:8983/solr/localDocs', timeout=10)
# query = 'dark chocolate '
# results = solr.search(f"Title:{query}", rows=100)
# print(len(results))
# for result in results:
#     print(result['id'])
