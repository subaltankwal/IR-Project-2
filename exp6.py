from math import log
import pysolr
import csv
from nltk import FreqDist
from nltk import word_tokenize


def getAllEntity():
    entityList = []
    entityRelation = {}
    csv_file_path = "gena_data_final_triples (1).csv"
    with open(csv_file_path, 'r', encoding='utf-8', newline="") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            entityList.append(row[0])
            if row[2] != "No synonyms":
                if row[0] not in entityRelation.keys():
                    entityRelation[row[0]] = row[2]
                else:
                    entityRelation[row[0]] += row[2]
        entityRelation.pop("Subject")
        entityList.remove("Subject")
    entity_set = set(entityList)
    return entity_set, entityRelation


solr = pysolr.Solr('http://localhost:8983/solr/localDocs', timeout=10)
F = 5371
entityInfo = getAllEntity()
objects = list(entityInfo[0])
entityRelation = entityInfo[1]

allDocs = solr.search('*:*', rows=1000000)
document_frequencies = {}
for document in allDocs:
    words = word_tokenize(' '.join(document['Abstract']).lower())
    freq_dist = FreqDist(words)
    for word in freq_dist.keys():
        document_frequencies[word] = document_frequencies.get(word, 0) + 1

rankObjects = {}
for object in objects:
    if object in document_frequencies.keys():
        rankObjects[object] = document_frequencies[object] * \
            log(F/document_frequencies[object])

E = len(rankObjects)
with open('nfcorpus/dev.vid-desc.queries', 'r', encoding='utf-8') as file:
    for l in file:
        line = l.split('\t')
        query = line[1]
