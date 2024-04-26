import pysolr
# solr = pysolr.Solr('http://localhost:8983/solr/nfdump')

# with open('nfcorpus/raw/nfdump.txt', 'r', encoding='utf-8') as file:
#     for l in file:
#         line = l.split('\t')
#         ID, URL, TITLE, MAIN_TEXT, COMMENTS, TOPICS_TAGS, DESCRIPTION, DOCTORS_NOTE, ARTICLE_LINKS, QUESTION_LINKS, TOPIC_LINKS, VIDEO_LINKS, MEDARTICLE_LINKS = line
#         solr.add([
#             {"id": "ID"},
#             {"url": "URL"},
#             {"title": "TITLE"},
#             {"maintext": "MAIN_TEXT"},
#             {"comments": "COMMENTS"},
#             {"topics_tags": "TOPICS_TAGS"},
#             {"description": "DESCRIPTION"},
#             {"doctors_note": "DOCTORS_NOTE"},
#             {"article_links": "ARTICLE_LINKS"},
#             {"question_links": "QUESTION_LINKS"},
#             {"topic_links": "TOPIC_LINKS"},
#             {"video_links": "VIDEO_LINKS"},
#             {"medarticle_links": "MEDARTICLE_LINKS"}
#         ])


solr = pysolr.Solr('http://localhost:8983/solr/localDocs')

with open('nfcorpus/raw/doc_dump.txt', 'r', encoding='utf-8') as file:
    for l in file:
        line = l.split('\t')
        ID, URL, TITLE, ABSTRACT = line
        solr.add([
            {
                "id": ID,
                "url": URL,
                "Title": TITLE,
                "Abstract": ABSTRACT,
            }
        ])
