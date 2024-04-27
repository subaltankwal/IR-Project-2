import pysolr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from itertools import chain
import numpy as np
import tensorflow as tf
from sklearn.metrics import ndcg_score

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
    return cosine_scores_with_ids

def features_and_labels(query):
    qid , qtext = query.split('\t')
    esc_qtext = escape_special_characters(qtext)
    cosine_score_with_titles = cosines(esc_qtext , "Title")
    cosine_score_with_abstract = cosines(esc_qtext , "Abstract")

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
            query_id , ignore , doc_id , rel_score = line.split('\t')
            given_relevance_labels[(query_id , doc_id)] = int(rel_score)

    ordered_features = []
    ordered_relevance_labels = []
    for feature in features:
        doc_id = feature[1]
        qid = feature[0]
        if (qid, doc_id) in given_relevance_labels:
            ordered_features.append(feature)
            ordered_relevance_labels.append(given_relevance_labels[(qid, doc_id)])
        else:
            ordered_features.append(feature)
            ordered_relevance_labels.append(0)

    return ordered_features, ordered_relevance_labels

def pair_documents(features, relevance_labels):
    paired_features = []
    paired_labels = []
    num_docs = len(features)
    for i in range(num_docs):
        for j in range(i + 1, num_docs):
            if relevance_labels[i] != relevance_labels[j]:
                feat_i = np.array(features[i][2:])
                feat_j = np.array(features[j][2:])
                paired_features.append(feat_i - feat_j)
                paired_labels.append(1 if relevance_labels[i] > relevance_labels[j] else 0)
    return np.array(paired_features), np.array(paired_labels)


def write_predictions_to_file(predictions, qid, doc_ids, output_file):
    combined = list(zip(predictions, qid, doc_ids))
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
    with open(output_file, 'w', encoding='utf-8') as file:
        count = 1
        for item in sorted_combined:
            file.write(f"{item[1]}\t0\t{item[2]}\t{count}\t{item[0]}\tExp-7\n")
            count += 1

def nn_model(train_features , train_relevance_labels, dev_features, dev_relevance_labels,output_file):
    train_paired_features, train_paired_labels = pair_documents(train_features, train_relevance_labels)
    dev_paired_features, dev_paired_labels = pair_documents(dev_features, dev_relevance_labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')

    model.fit(train_paired_features, train_paired_labels, epochs=10, batch_size=32, validation_data=(dev_paired_features, dev_paired_labels))

    predicted_rel = model.predict(dev_paired_features).flatten()
    probabilities = np.where(predicted_rel > 0.5, predicted_rel, 1 - predicted_rel)
    write_predictions_to_file(probabilities, dev_qids, dev_doc_ids, output_file)

    val_loss = model.evaluate(dev_paired_features, dev_paired_labels)
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

# Storing dev qids and doc ids
dev_doc_ids = np.array([feature[1] for feature in dev_feat])
dev_qids = np.array([feature[0] for feature in dev_feat])

nn_model(train_feat , train_rel , dev_feat , dev_rel ,'rel2_out.txt')

def load_ground_truth(file_path):
    gt_relevance = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            query_id, _, doc_id, rel_score = line.strip().split('\t')
            gt_relevance.setdefault(query_id, {})[doc_id] = int(rel_score)
    return gt_relevance

def load_predicted_rankings(file_path):
    predicted_rankings = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            query_id, _, doc_id, rank, score, _ = line.strip().split('\t')
            predicted_rankings.setdefault(query_id, []).append((doc_id, float(score)))
    return predicted_rankings

def calculate_ndcg(gt_relevance, predicted_rankings):
    ndcg_scores = []
    for query_id, ranked_list in predicted_rankings.items():
        ground_truth = [gt_relevance[query_id].get(doc_id, 0) for doc_id, _ in ranked_list]
        predicted_scores = [score for _, score in ranked_list]
        ndcg = ndcg_score([ground_truth], [predicted_scores], k=len(ranked_list))
        ndcg_scores.append(ndcg)
    avg_ndcg = sum(ndcg_scores) / len(ndcg_scores)
    return avg_ndcg

gt_relevance = load_ground_truth('nfcorpus/merged.qrel')
predicted_rankings = load_predicted_rankings('rel2_out.txt')
avg_ndcg = calculate_ndcg(gt_relevance, predicted_rankings)

print("Average NDCG:", avg_ndcg)