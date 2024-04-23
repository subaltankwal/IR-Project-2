import pysolr
from collections import Counter
import re


# Initialize PySolr client
solr = pysolr.Solr('http://localhost:8983/solr/localDocs', always_commit=True)

# Function to perform initial retrieval


def initial_retrieval(train_query, search_params):
    results = solr.search(train_query, **search_params)
    return results.docs

# Function to select top-ranked documents


def select_top_documents(docs, N):
    return docs[:N]

# Function to extract relevant terms from documents


def extract_terms(documents):
    all_text = ' '.join(doc['Title'] for doc in documents)
    terms = re.findall(r'\b\w+\b', all_text.lower())
    term_counts = Counter(terms)
    return [term for term, _ in term_counts.most_common(5)]  # Top 5 terms

# Function to expand test query with relevant terms


def expand_query(test_query, relevant_terms):
    expanded_query = test_query + ' ' + ' '.join(relevant_terms)
    return expanded_query

# Function to perform retrieval with expanded query


def retrieval_with_expanded_query(expanded_query, search_params):
    expanded_results = solr.search(expanded_query, **search_params)
    return expanded_results.docs

# Function to evaluate performance


def evaluate_performance(results, relevant_ids):
    for doc in results:
        doc['relevance'] = 1 if doc['id'] in relevant_ids else 0
    return results

# Function to identify relevant documents


def identify_relevant_documents(results, N):
    relevant_ids = set(doc['id'] for doc in results[:N])
    return relevant_ids

# Function to identify non-relevant documents


def identify_non_relevant_documents(results, N):
    non_relevant_ids = set(doc['id'] for doc in results[N:])
    return non_relevant_ids

# Main function


def main():
    train_query = "example train query"
    test_query = "example test query"
    N = 10  # Number of top documents to select

    search_params = {
        'rows': 100,  # Number of rows to retrieve
        # Add any additional parameters as needed
    }

    # Initial retrieval
    initial_docs = initial_retrieval(train_query, search_params)

    # Select top-ranked documents
    selected_docs = select_top_documents(initial_docs, N)

    # Extract relevant terms
    relevant_terms = extract_terms(selected_docs)

    # Expand test query
    expanded_query = expand_query(test_query, relevant_terms)

    # Retrieval with expanded query
    expanded_results = retrieval_with_expanded_query(
        expanded_query, search_params)

    # Identify relevant documents
    relevant_ids = identify_relevant_documents(expanded_results, N)

    # Evaluate performance
    expanded_results = evaluate_performance(expanded_results, relevant_ids)

    # Identify non-relevant documents
    non_relevant_ids = identify_non_relevant_documents(expanded_results, N)

    print("Relevant documents:")
    for doc in expanded_results:
        if doc['id'] in relevant_ids:
            print(doc['id'])

    print("\nNon-relevant documents:")
    for doc in expanded_results:
        if doc['id'] in non_relevant_ids:
            print(doc['id'])


if __name__ == "__main__":
    main()
