import os
import math
from collections import defaultdict
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Function to read documents from .txt files

punctuations = '''!()-[]{};:'"\,<>./?#%^*_~'''


def read_documents_from_directory(directory):
    documents = {}
    titles = {}
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith("nfdump.txt"):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                lines = file.readlines()
                for line in lines:
                    count += 1
                    # Split the line into document name and content, ignoring the website
                    parts = line.split('\t')
                    doc_no, url, title, mainText, comments, topics_tags, description, doctors_note, article_links, question_links, topic_links, video_links, medarticle_links = [
                        parts[i] for i in range(0, 13)]
                    documents[doc_no] = description
                    titles[doc_no] = title

    return documents, titles, count

# Function to tokenize documents into terms


def getDocFreq():
    df = {}
    with open('nfcorpus/index/postings.tsv', 'r', encoding='utf-8') as file:
        for l in file:
            line = l.split('\t')
            temp = line[0]
            for x in temp:
                if x in punctuations:
                    temp = temp.replace(x, '')
            if len(temp) >= 1:
                df[temp] = line[2]
    return df


def tokenize_document(document):
    return word_tokenize(document.lower())

# Function to calculate term frequencies in documents


def calculate_term_frequencies(documents):
    term_frequencies = {}
    for doc_name, content in documents.items():
        terms = {}
        tokens = tokenize_document(content)
        for token in tokens:
            for x in token:
                if x in punctuations:
                    token = token.replace(x, '')
            if len(token) >= 1:
                if token not in terms:
                    terms[token] = 1
                else:
                    terms[token] += 1
        term_frequencies[doc_name] = terms
    return term_frequencies


def norm(terms):
    return math.sqrt(sum(freq ** 2 for freq in terms.values()))

# Function to normalize term frequencies by Euclidean length (nnn)


def normalize_c(term_frequencies):
    normalized_documents = {}
    for doc_name, terms in term_frequencies.items():
        norm_length = norm(terms)
        normalized_terms = {term: freq /
                            norm_length for term, freq in terms.items()}
        normalized_documents[doc_name] = normalized_terms
    return normalized_documents


def normalize_ntn(total_docs):
    t_df = {}
    with open('nfcorpus/index/postings.tsv', 'r', encoding='utf-8') as file:
        for l in file:
            line = l.split('\t')
            temp = line[0]
            for x in temp:
                if x in punctuations:
                    temp = temp.replace(x, '')
            if len(temp) >= 1:
                t_df[temp] = math.log10(total_docs/int(line[2]))
    return t_df


# Directory containing .txt files
directory = "./nfcorpus/raw"

# Read documents from .txt files in the specified directory
documents, titles, total_docs = read_documents_from_directory(directory)
df = getDocFreq()

# Calculate term frequencies in documents
term_frequencies_docs = calculate_term_frequencies(documents)
term_frequencies_title = calculate_term_frequencies(titles)
print("NNN")
print()
for term, doc_freq in df.items():
    print(f"term '{term}' has document frequency as '{doc_freq}'")
    break

for doc_name, normalized_terms in term_frequencies_docs.items():
    print(f"Document '{doc_name}' has this normalized vector:",
          normalized_terms)
    break

for doc_name, normalized_terms in term_frequencies_title.items():
    print(f"Title '{doc_name}' has this normalized vector:",
          normalized_terms)
    break

print("NTN")
print("Only the Document frequency changes for this:")
t_df = normalize_ntn(total_docs)
for term, doc_freq in t_df.items():
    print(f"term '{term}' has document frequency as '{doc_freq}'")
    break

print("NTC")
print("Only the vector will be cosine normalize")
normalized_ntc = normalize_c(term_frequencies_docs)
for doc_term, normalized_ter in normalized_ntc.items():
    print(f"'{doc_term}' has normalized vector as '{normalized_ter}'")
    break
