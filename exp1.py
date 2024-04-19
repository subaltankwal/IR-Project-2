import os

def read_data(data_path):
    with open(data_path , 'r' ,encoding = 'utf-8') as file:
        if not os.path.exists('nfcorpus' + "/index/"):
            os.mkdir('nfcorpus' + "/index/")
        o = open('nfcorpus' + "/index/output.tsv", "w", encoding="utf-8")
        for line in file:
            parts = line.strip().split('\t')
            docno, url, title, content = parts
            # documents.append({'docno': docno, 'url': url, 'title': title, 'content': content})
            tokens = title.split(" ")
            for t in tokens:
                o.write(t.lower() + "\t" + str(docno) + "\n")
            tokens = content.split(" ")
            for t in tokens:
                o.write(t.lower() + "\t" + str(docno) + "\n")
        o.close()

def sort(dir):
    f = open(dir + "/index/output.tsv", encoding="utf-8")
    o = open(dir + "/index/output_sorted.tsv", "w", encoding="utf-8")

    # initialize an empty list of pairs of
    # tokens and their doc_ids
    pairs = []

    for line in f:
        line = line[:-1]
        split_line = line.split("\t")
        if len(split_line) == 2:
            pair = (split_line[0], split_line[1])
            pairs.append(pair)

    # sort (token, doc_id) pairs by token first and then doc_id
    sorted_pairs = sorted(pairs, key=lambda x: (x[0], x[1]))

    # write sorted pairs to file
    for sp in sorted_pairs:
        o.write(sp[0] + "\t" + sp[1] + "\n")
    o.close()

read_data('nfcorpus/raw/doc_dump.txt')
sort('nfcorpus')

def construct_postings(dir):
    # open file to write postings
    o1 = open(dir + "/index/postings.tsv", "w", encoding="utf-8")

    postings = {}  # initialize our dictionary of terms
    doc_freq = {}  # document frequency for each term
    term_freq = {} # term frequency

    # read the file containing the sorted pairs
    f = open(dir + "/index/output_sorted.tsv", encoding="utf-8")

    # initialize sorted pairs
    sorted_pairs = []

    # read sorted pairs
    for line in f:
        line = line[:-1]
        split_line = line.split("\t")
        pairs = (split_line[0], split_line[1])
        sorted_pairs.append(pairs)

    # construct postings from sorted pairs
    for pairs in sorted_pairs:
        if pairs[0] not in postings:
            postings[pairs[0]] = []
            postings[pairs[0]].append(pairs[1])
            term_freq[pairs[0]] = 1
        else:
            term_freq[pairs[0]] += 1
            len_postings = len(postings[pairs[0]])
            if len_postings >= 1:
                # check for duplicates
                # assuming the doc_ids are sorted
                # the same doc_ids will appear
                # one after another and detected by
                # checking the last element of the postings
                if pairs[1] != postings[pairs[0]][len_postings - 1]:
                    postings[pairs[0]].append(pairs[1])

    # update doc_freq which is the size of postings list
    for token in postings:
        doc_freq[token] = len(postings[token])

    # print("postings: " + str(postings))
    # print("doc freq: " + str(doc_freq))
    print("Dictionary size: " + str(len(postings)))

    # write postings and document frequency to file

    for token in postings:
        o1.write(token + "\t" + str(term_freq[token]) + "\t" +str(doc_freq[token]))
        for l in postings[token]:
            o1.write("\t" + l)
        o1.write("\n")
    o1.close()

read_data('nfcorpus/raw/doc_dump.txt')
sort('nfcorpus')
construct_postings('nfcorpus')
