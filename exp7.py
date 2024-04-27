import pysolr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

def escape_special_characters(query):
    punctuations = '''!()-[]{};:'"\,<>./?#%^*_~'''
    for x in query:
        if x in punctuations:
            query = query.replace(x , "")
    return query

def calculate_cosine_similarity(query_vector, document_vectors):
    cosine_similarities = cosine_similarity(query_vector, document_vectors)

    return cosine_similarities[0]
def cosines(query , field):
    solr = pysolr.Solr('http://localhost:8983/solr/localDocs', timeout=10)

    results = solr.search(f'{field}:{query}', rows=100)
    if not results:
        return {}
    document_ids = [doc['id'] for doc in results]
    document_texts = [doc[field] for doc in results]

    vectorizer = TfidfVectorizer(stop_words='english')
    document_texts = list(chain.from_iterable(document_texts))
    all_texts = [query] + document_texts
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    query_vector = tfidf_matrix[0:1]
    document_vectors = tfidf_matrix[1:]

    cosine_similarities = calculate_cosine_similarity(query_vector, document_vectors)

    cosine_scores_with_ids = dict(zip(document_ids, cosine_similarities))
    # print(f"the dict with field {field} is {cosine_scores_with_ids}")
    return cosine_scores_with_ids

def features_and_labels(query):
    qid , qtext = query.split('\t')
    esc_qtext = escape_special_characters(qtext)
    cosine_score_with_titles = cosines(esc_qtext , "Title")
    cosine_score_with_abstract = cosines(esc_qtext , "Abstract")
    # print(f" {qid} : {cosine_score_with_abstract}")

    features = []
    for key in set(cosine_score_with_titles) | set(cosine_score_with_abstract):
        feature = []
        value1 = cosine_score_with_titles.get(key , 0)
        value2 = cosine_score_with_abstract.get(key , 0)
        feature = [qid , key , value1 , value2 ]
        features.append(feature)

    given_relevance_labels = {}
    with open('nfcorpus/merged.qrel' , 'r' , encoding='utf-8') as file:
        for line in file:
            # print(line)
            query_id , ignore , doc_id , rel_score = line.split('\t')
            given_relevance_labels[(query_id , doc_id)] = int(rel_score)

    ordered_features = []
    ordered_relevance_labels = []
    for feature in features:
        doc_id = feature[1]
        qid = feature[0]
        # print(feature[2:])
        if (qid, doc_id) in given_relevance_labels:
            ordered_features.append(feature)
            ordered_relevance_labels.append(given_relevance_labels[(qid, doc_id)])
        else:
            ordered_features.append(feature)
            ordered_relevance_labels.append(0)
    # print(features)
    # print(ordered_features)
    # print(f"rel : {ordered_relevance_labels}")

    return ordered_features, ordered_relevance_labels

def write_predictions_to_file(predictions, qid, doc_ids, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for i in range(len(predictions)):
            file.write(f"{qid[i]}\t{doc_ids[i]}\t{predictions[i]}\n")

def nn_model(train_features , train_relevance_labels, dev_features, dev_relevance_labels,output_file):
    # if not train_features:
    #     X_train = np.array([0,0])
    X_train = np.array([feature[2:] for feature in train_features])
    X_dev = np.array([feature[2:] for feature in dev_features])
    Y_train = np.array(train_relevance_labels)
    Y_dev = np.array(dev_relevance_labels)
    dev_doc_ids = np.array([feature[1] for feature in dev_features])
    dev_qids = np.array([feature[0] for feature in dev_features])
    # print(train_features)
    # print(f"train:  {X_train}")
    # print(Y_train)

    model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1) ])
    model.compile(optimizer='adam', loss='mean_squared_error')

    print("Shape of X_train:", X_train.shape)
    print("Shape of Y_train:", Y_train.shape)

    model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_data=(X_dev, Y_dev))
    predicted_rel = model.predict(X_dev).flatten()
    print("Length of dev_doc_ids:", len(dev_doc_ids))
    print("Length of predictions:", len(predicted_rel))
    write_predictions_to_file(predicted_rel, dev_qids, dev_doc_ids, output_file)

    val_loss = model.evaluate(X_dev, Y_dev)
    print(f"Validation Loss: {val_loss:.4f}")

train_feat = []
train_rel = []
with open('nfcorpus/train.titles.queries', 'r' , encoding='utf-8') as file:
    for line in file:
        temp_features , temp_rel = features_and_labels(line)
        train_feat+= temp_features
        train_rel += temp_rel

dev_feat = []
dev_rel = []
with open('nfcorpus/dev.titles.queries', 'r' , encoding='utf-8') as file:
    for line in file:
        temp_features , temp_rel = features_and_labels(line)
        dev_feat += temp_features
        dev_rel += temp_rel
        
nn_model(train_feat , train_rel , dev_feat , dev_rel ,'rel_out.txt')

# d = cosines("dark chocolate" , "Title")
# print(d)