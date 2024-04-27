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
            query = query.replace(x, "")
    return query

def cosines(query, field):
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

    cosine_similarities = np.squeeze(np.asarray(np.dot(query_vector, document_vectors.T)))

    cosine_scores_with_ids = dict(zip(document_ids, cosine_similarities))
    return cosine_scores_with_ids
def write_predictions_to_file(predictions, qid, doc_ids, output_file):
    combined = list(zip(predictions, qid, doc_ids))
    sorted_combined = sorted(combined, key=lambda x: x[0], reverse=True)
    with open(output_file, 'w', encoding='utf-8') as file:
        count = 1
        for item in sorted_combined:
            file.write(f"{item[1]}\t0\t{item[2]}\t{count}\t{item[0]}\tExp-7\n")
            count += 1
def features_and_labels(query):
    qid, qtext = query.split('\t')
    esc_qtext = escape_special_characters(qtext)
    cosine_score_with_titles = cosines(esc_qtext, "Title")
    cosine_score_with_abstract = cosines(esc_qtext, "Abstract")

    features = []
    for key in set(cosine_score_with_titles) | set(cosine_score_with_abstract):
        value1 = cosine_score_with_titles.get(key, 0)
        value2 = cosine_score_with_abstract.get(key, 0)
        features.append([key, value1, value2])

    given_relevance_labels = {}
    with open('nfcorpus/merged.qrel', 'r', encoding='utf-8') as file:
        for line in file:
            query_id, _, doc_id, rel_score = line.split('\t')
            given_relevance_labels[(query_id, doc_id)] = int(rel_score)

    ordered_features = []
    ordered_relevance_labels = []
    for feature in features:
        doc_id = feature[0]
        if (qid, doc_id) in given_relevance_labels:
            ordered_features.append(feature[1:])
            ordered_relevance_labels.append(given_relevance_labels[(qid, doc_id)])
        else:
            ordered_features.append(feature[1:])
            ordered_relevance_labels.append(0)

    return ordered_features, ordered_relevance_labels

def listnet_model(train_features, train_relevance_labels, dev_features, dev_relevance_labels, output_file):
    X_train = np.array(train_features)
    X_dev = np.array(dev_features)
    Y_train = np.array(train_relevance_labels)
    Y_dev = np.array(dev_relevance_labels)

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 classes for relevance levels
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Shape of X_train:", X_train.shape)
    print("Shape of Y_train:", Y_train.shape)

    model.fit(X_train, Y_train, epochs=10, batch_size=32, validation_split=0.2)

    predicted_rel = model.predict(X_dev)
    predicted_rel_indices = np.argmax(predicted_rel, axis=1)  # Convert probabilities to class indices
    write_predictions_to_file(predicted_rel_indices, dev_qids, dev_doc_ids, output_file)

    val_loss, val_accuracy = model.evaluate(X_dev, Y_dev)
    print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

train_feat = []
train_rel = []
with open('nfcorpus/train.titles.queries', 'r', encoding='utf-8') as file:
    for line in file:
        temp_features, temp_rel = features_and_labels(line)
        train_feat += temp_features
        train_rel += temp_rel

dev_feat = []
dev_rel = []
with open('nfcorpus/dev.titles.queries', 'r', encoding='utf-8') as file:
    for line in file:
        temp_features, temp_rel = features_and_labels(line)
        dev_feat += temp_features
        dev_rel += temp_rel

dev_doc_ids = np.array([feature[0] for feature in dev_feat])
dev_qids = np.array([i for i in range(len(dev_feat))])  # Just use an incremental ID for queries

listnet_model(train_feat, train_rel, dev_feat, dev_rel, 'listnet_out.txt')

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
