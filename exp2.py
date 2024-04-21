import os
import math
from collections import defaultdict
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Function to read documents from .txt files

punctuations = ":(){}[],.';/"


def read_documents_from_directory(directory):
    documents = defaultdict(str)
    titles = defaultdict(str)
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith("doc_dump.txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                lines = file.readlines()
                for line in lines:
                    count += 1
                    # Split the line into document name and content, ignoring the website
                    parts = line.strip().split('\t')
                    if (parts[0] == 'MED-5371'):
                        doc_num, link, title, content = parts[0], parts[1], parts[2], parts[3]
                        documents[doc_num] += content
                        titles[doc_num] += title
                    else:
                        doc_num, link, title, content = parts[0], parts[1], parts[2], parts[3]
                        documents[doc_num] += content
                        titles[doc_num] += title

    return documents, titles, count

# Function to tokenize documents into terms


def tokenize_document(document):
    return word_tokenize(document.lower())

# Function to calculate term frequencies in documents


def calculate_term_frequencies(documents):
    term_frequencies = {}
    for doc_name, content in documents.items():
        terms = defaultdict(int)
        tokens = tokenize_document(content)
        for token in tokens:
            if token not in punctuations:
                terms[token] += 1
        term_frequencies[doc_name] = terms
    return term_frequencies
# Function to calculate Euclidean length of a vector


def norm(terms):
    return math.sqrt(sum(freq ** 2 for freq in terms.values()))

# Function to normalize term frequencies by Euclidean length (nnn)


def normalize_c(term_frequencies):
    normalized_documents = {}
    for doc_name, terms in term_frequencies.items():
        # print("Hemlo", terms)
        norm_length = norm(terms)
        normalized_terms = {term: freq /
                            norm_length for term, freq in terms.items()}
        normalized_documents[doc_name] = normalized_terms
    return normalized_documents


def normalize_ntn(term_frequencies, total_docs):
    with open('nfcorpus/raw/doc_dump.txt', 'r', encoding='utf-8') as file:
        for l in file:
            line = l.split('\t')
            for x in line[0]:
                if x in punctuations:
                    x.replace('')
            df[line[0]] = line[2]/total_docs


# Directory containing .txt files
directory = "./nfcorpus/raw"

# Read documents from .txt files in the specified directory
documents, titles, total_docs = read_documents_from_directory(directory)
df = {}

# Calculate term frequencies in documents
term_frequencies_docs = calculate_term_frequencies(documents)
term_frequencies_title = calculate_term_frequencies(titles)
print("NNN")
print()
for doc_name, normalized_terms in term_frequencies_docs.items():
    print(f"Document '{doc_name}' has this normalized vector:",
          normalized_terms)
    break

for doc_name, normalized_terms in term_frequencies_title.items():
    print(f"Title '{doc_name}' has this normalized vector:",
          normalized_terms)
    break

print("NTN")
print()
normalized_ntn = normalize_ntn(term_frequencies_docs)

# normalized_nnn = normalize_c(term_frequencies_docs)
# for doc_name, normalized_terms in normalized_nnn.items():
#     print(f"Document '{doc_name}' has this normalized vector:",
#           normalized_terms)
#     break

# for doc_name, normalized_terms in term_frequencies_title.items():
#     print(f"Title '{doc_name}' has this normalized vector:",
#           normalized_terms)
#     break
