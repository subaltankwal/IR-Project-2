from math import log
import pysolr
import csv
from nltk import FreqDist
from nltk import word_tokenize
import re
import exp3Rocchio


def getAllEntity():
    entityList = []
    entityRelation = {}
    csv_file_path = "gena_data_final_triples (1).csv"
    with open(csv_file_path, 'r', encoding='utf-8', newline="") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            if len(row[0]) >= 5:
                entityList.append(row[0])
            if row[2] != "No synonyms":
                if row[0] not in entityRelation.keys():
                    entityRelation[row[0]] = [row[2]]
                else:
                    entityRelation[row[0]].append(row[2])
        entityRelation.pop("Subject")
        entityList.remove("Subject")
    entity_set = set(entityList)
    return entity_set, entityRelation


solr = pysolr.Solr('http://localhost:8983/solr/localDocs', timeout=10)
entityInfo = getAllEntity()
objects = list(entityInfo[0])
entityRelation = entityInfo[1]

allDocs = solr.search('*:*', rows=1000000)
document_entity_vector = {}
with open('nfcorpus/train.titles.queries', 'r', encoding='utf-8') as file:
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
                                num = len(re.findall(object2, info))
                                if num != 0:
                                    query_entity_freq[object2] += 1/num
                            else:
                                num = len(re.findall(object2, info))
                                if num != 0:
                                    query_entity_freq[object2] = 1/num
        if query_entity_freq == {}:
            print("No entity for :", query)
        else:
            score = {}
            count = 0
            for doc in allDocs:
                doc_freq = {}
                for object in objects:
                    if object in ' '.join(doc['Title']):
                        if object in doc_freq.keys():
                            doc_freq[object] += len(re.findall(object,
                                                    ' '.join(doc['Title'])))
                        else:
                            doc_freq[object] = len(re.findall(
                                object, ' '.join(doc['Title'])))
                for key in query_entity_freq.keys():
                    if key in doc_freq.keys() and key != 'y':
                        if doc['id'] in score.keys():
                            score[doc['id']] += query_entity_freq[key] * \
                                doc_freq[key]
                        else:
                            score[doc['id']] = query_entity_freq[key] * \
                                doc_freq[key]
            sortedScore = dict(
                sorted(score.items(), key=lambda key_val: key_val[1], reverse=True))
            relevant_docs = list(sortedScore.keys())[:100]
            # for key, value in sortedScore.items():
            #     if value == 0.0:
            #         break
            #     else:
            #         relevant_docs.append(key)
            non_relevant_docs = set("MED-" + str(i) for i in range(1, 5372))
            for doc in relevant_docs:
                if doc in non_relevant_docs:
                    non_relevant_docs.remove(doc)
            updated_query_vector = exp3Rocchio.rocchio_algorithm(
                exp3Rocchio.term_frequency(query), relevant_docs, non_relevant_docs)
            print(query, updated_query_vector)
