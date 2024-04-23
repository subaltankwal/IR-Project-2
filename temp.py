import pysolr
from collections import Counter

solr = pysolr.Solr('http://localhost:8983/solr/localDocs', timeout=10)


def perform_prf(train_queries, field):
    for query in train_queries:
        results = solr.search(field + ':"' + query + '"', rows=10)
        if results:
            relevant_docs = [doc['id'] for doc in results]
            non_relevant_docs = []  # Initialize list for non-relevant documents
            all_docs = []  # Initialize list for all documents
            query_vector = Counter()
            doc_freq = Counter()

            # Retrieve term frequencies from relevant documents
            for doc_id in relevant_docs:
                doc = solr.search(q="id:" + doc_id).docs[0]
                for word in doc.get(field, ""):
                    doc_freq[word] += 1
                all_docs.append(doc_id)

            # Retrieve term frequencies from non-relevant documents
            for doc_id in all_docs:
                if doc_id not in relevant_docs:
                    non_relevant_docs.append(doc_id)
                    doc = solr.search(q="id:" + doc_id).docs[0]
                    field_value = doc.get(field, "")
                    if isinstance(field_value, list):
                        field_value = " ".join(field_value)
                    for word in field_value.split():
                        doc_freq[word] += 1

            # Calculate centroid based on relevant and non-relevant documents
            centroid = {word: (doc_freq[word] if word in doc_freq else 0) / len(all_docs)
                        for word in doc_freq.keys()}

            # Retrieve term frequencies from all documents
            for doc_id in all_docs:
                doc = solr.search(q="id:" + doc_id).docs[0]
                field_value = doc.get(field, "")
                if isinstance(field_value, list):
                    field_value = " ".join(field_value)
                for word in field_value.split():
                    query_vector[word] += 1

            # Normalize the query vector
            max_tf = max(query_vector.values(), default=1)
            query_vector_normalized = {
                word: tf / max_tf for word, tf in query_vector.items()}

            # Apply the Rocchio feedback formula to compute the expanded query
            alpha = 1
            beta = 0.75
            gamma = 0.25
            expanded_query = {word: alpha * query_vector_normalized.get(word, 0) +
                              beta * centroid.get(word, 0) - gamma *
                              query_vector_normalized.get(word, 0)
                              for word in set(query_vector_normalized) | set(centroid)}

            # Construct the expanded query string
            expanded_query_str = " ".join(
                [f'"{word}"' for word, value in expanded_query.items() if value > 0.2])
            print(train_queries, ":", expanded_query_str)

        else:
            print("No results found. Unable to calculate centroid.")


train_queries = ['how contaminated']

perform_prf(train_queries, 'Title')
