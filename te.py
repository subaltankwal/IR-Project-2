import pysolr
solr = pysolr.Solr('http://localhost:8983/solr/localDocs')
# with open('nfcorpus/raw/doc_dump.txt', 'r', encoding='utf-8') as file:
#     for l in file:
#         line = l.split('\t')
#         ID, URL, TITLE, ABSTRACT = [line[i] for i in range(0, 4)]
#         doc = {
#             'id': ID,
#             'URL': URL,
#             'Title': TITLE,
#             'Abstract': ABSTRACT
#         }
#         solr.add([doc])
