import spacy
import csv
nlp = spacy.load("en_core_web_sm")


def getAllEntity():
    entityList = []
    csv_file_path = "gena_data_final_triples (1).csv"
    with open(csv_file_path, 'r', encoding='utf-8', newline="") as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            entityList.append(row[0])
        entityList.remove("Subject")
    entity_set = set(entityList)
    return entity_set


def bag_of_entities(document, entity_list):
    entity_vector = {}
    n = len(document)
    for i in range(n):
        for j in range(i+1, n+1):
            substring = document[i:j]
            if substring in entity_list:
                if substring in entity_vector:
                    entity_vector[substring] += 1
                else:
                    entity_vector[substring] = 1
    return entity_vector


def getRankedList(query, entity_list):
    queryEntities = bag_of_entities(query, entity_list)
    score = {}
    if queryEntities == {}:
        print("No entities for:", query)
    else:
        with open('nfcorpus/raw/doc_dump.txt', 'r', encoding='utf-8') as file:
            for l in file:
                line = l.split('\t')
                id, link, title, abstract = line
                docEntities = bag_of_entities(title, entity_list)
                docScore = 0
                for term, freq in queryEntities.items():
                    if term in docEntities.keys():
                        docScore += (freq*docEntities[term])
                if docScore != 0:
                    score[id] = docScore
        sortedScore = dict(
            sorted(score.items(), key=lambda key_val: key_val[1]))
        retrieved_docs = list(sortedScore.keys())
        if len(retrieved_docs)-10 > 0:
            topDocs = retrieved_docs[len(retrieved_docs)-10:]
            print("Retrieved Docs for", query, ":", set(topDocs))
            print()
        else:
            print("Retrieved Docs for", query, ":", set(retrieved_docs))
            print()


entity_list = getAllEntity()
with open('nfcorpus/train.titles.queries', 'r', encoding='utf-8') as file:
    for l in file:
        line = l.split('\t')
        getRankedList(
            line[1], entity_list)
