from math import log
import pysolr
import csv


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


solr = pysolr.Solr('http://localhost:8983/solr/localDocs', timeout=10)
F = 5371
objects = list(getAllEntity())
params = {
    'terms': 'true',
    'terms.fl': 'Title',
    'terms.limit': -1,
    'terms.sort': 'count',
}
df = {}
for object in objects:
    # finalObject = pysolr.escape_query_chars(object)
    object.replace('+', r'\+').replace('(', r'\(').replace(')', r'\)')
    response = solr.search(object, **params)
    for res in response:
        print(res)
    # print(response)
    # if 'terms' in response.raw_response:
    #     terms = response.raw_response['terms']['Title']
    #     for term_info in terms:
    #         term = term_info['term']
    #         frequency = term_info['count']
    #         print(f'Term: {term}, Frequency: {frequency}')
    # else:
    #     print("No term frequency information found in the response.")

# print(df)

# Frequency of each object in each document
# tf = {'d1': {'o1': 5, 'o2': 7, 'o3': 10},
#       'd2': {'o1': 8, 'o2': 6, 'o3': 12}}  # Replace with actual term frequencies


# def rank_objects(objects, tf, df, F):
#     rankings = {}
#     for o in objects:
#         # Calculate ranking score for each object
#         score = sum(tf[d].get(o, 0) * log(F / df[o], 10) for d in tf)
#         rankings[o] = score
#     # Sort objects by score in descending order
#     ranked_objects = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
#     return ranked_objects


# Get the ranked list of objects with their scores
# ranked_list = rank_objects(objects, tf, df, F)
# print(ranked_list)
