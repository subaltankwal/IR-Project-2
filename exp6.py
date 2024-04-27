from math import log
import pysolr
import csv
from nltk import FreqDist
from nltk import word_tokenize
import re


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
# document_frequencies = {}
# for document in allDocs:
#     words = word_tokenize(' '.join(document['Abstract']).lower())
#     freq_dist = FreqDist(words)
#     for word in freq_dist.keys():
#         document_frequencies[word] = document_frequencies.get(word, 0) + 1

# step-1
# rankObjects = {}

# for object in objects:
#     for doc in allDocs:
#         term_fre = len(re.findall(object, ' '.join(doc['Abstract'])))
#         if object in rankObjects.keys():
#             if object in document_frequencies.keys():
#                 rankObjects[object] += term_fre * \
#                     log(F/document_frequencies[object])
#             else:
#                 rankObjects[object] = term_fre * \
#                     log(F/document_frequencies[object])

# step-2


# E = len(rankObjects)

document_entity_vector = {}
for document in allDocs:
    curr_doc_vector = {}
    for object in objects:
        if object in document['Abstract']:
            if object in curr_doc_vector.keys():
                curr_doc_vector[object] += len(re.findall(object,
                                               document['Abstract']))
            else:
                curr_doc_vector[object] = len(
                    re.findall(object, document['Abstract']))

            objRelation = entityRelation[object]
            for info in objRelation:
                for object2 in objects:
                    if object2 in info:
                        if object2 in curr_doc_vector.keys():
                            curr_doc_vector[object2] += len(
                                re.findall(object2, info))
                        else:
                            curr_doc_vector[object2] = len(
                                re.findall(object2, info))
    document_entity_vector[document['id']] = curr_doc_vector


with open('nfcorpus/dev.vid-desc.queries', 'r', encoding='utf-8') as file:
    for l in file:
        query_entity_freq = {}
        line = l.split('\t')
        query = line[1]
        for object in objects:
            if object in query:
                if object in query_entity_freq.keys():
                    query_entity_freq[object] += len(re.findall(object, query))
                else:
                    query_entity_freq[object] = len(re.findall(object, query))

                objRelation = entityRelation[object]
                for info in objRelation:
                    for object2 in objects:
                        if object2 in info:
                            if object2 in query_entity_freq.keys():
                                query_entity_freq[object2] += len(
                                    re.findall(object2, info))
                            else:
                                query_entity_freq[object2] = len(
                                    re.findall(object2, info))

        for doc, doc_vector in document_entity_vector.items():
