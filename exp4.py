punctuations = '''!()-[]{};:'"\,<>./?#%^*_~'''
def calculate_term_frequencies(documents):
    term_frequencies = {}
    for doc_name, content in documents.items():
        terms = {}
        tokens = content.lower()
        for token in tokens:
            if token not in punctuations:
                terms[token] += 1
        term_frequencies[doc_name] = terms
    return term_frequencies

def lang_model(query):
    dict = {}
    document_scores = {}
    freq = {}
    with open("nfcorpus/index/postings.tsv", encoding='utf-8') as file:
        for l in file:
            line = l.split('\t')
            word = line[0]
            for x in word:
                if x in punctuations:
                    word = word.replace(x, "")
            term_freq = line[1]
            doc_list = line[3:]
            if (word not in dict):
                freq[word] = term_freq
                dict[word] = set(doc_list)
            else:
                freq[word] += term_freq
                dict[word] = set(dict[word]).union(set(doc_list))
    words = query.lower().split()

    for word in words:
        if word in dict:
            for doc_id in dict[word]:
                if doc_id not in document_scores:
                    document_scores[doc_id] = 1
                