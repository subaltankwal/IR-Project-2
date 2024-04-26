import spacy

# Load the English language model
nlp = spacy.load("en_core_web_sm")

# Sample documents with associated entities
documents = [
    {"text": "Barack Obama was the 44th President of the United States.",
        "entities": ["Barack Obama", "United States"]},
    {"text": "Steve Jobs co-founded Apple Inc.",
        "entities": ["Steve Jobs", "Apple Inc."]},
    {"text": "Albert Einstein was a theoretical physicist.",
        "entities": ["Albert Einstein"]}
]

# Index entities
entity_index = {}
for i, doc in enumerate(documents):
    for entity in doc['entities']:
        if entity not in entity_index:
            entity_index[entity] = []
        entity_index[entity].append(i)

# Function to retrieve documents based on entities in the query


def retrieve_documents(query):
    query_entities = set()
    query_doc = nlp(query)
    for ent in query_doc.ents:
        query_entities.add(ent.text)

    relevant_documents = set()
    for entity in query_entities:
        if entity in entity_index:
            relevant_documents.update(entity_index[entity])

    return [documents[i]['text'] for i in relevant_documents]


# Example query
query = "Who was the president of the United States?"
relevant_documents = retrieve_documents(query)
print("Relevant Documents:")
for doc in relevant_documents:
    print("-", doc)
