import lightgbm

from document_preprocessor import Tokenizer
from indexing import InvertedIndex, BasicInvertedIndex
from ranker import *
# from vector_ranker import *
from collections import defaultdict
import csv
import numpy as np


class L2RRanker:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 document_preprocessor: Tokenizer, stopwords: set[str], ranker: Ranker,
                 feature_extractor: 'L2RFeatureExtractor') -> None:
        """
        Initializes a L2RRanker model.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            ranker: The Ranker object ** hw3 modified **
            feature_extractor: The L2RFeatureExtractor object
        """
        # TODO: Save any arguments that are needed as fields of this class
        # self.scorer = scorer
        self.document_index = document_index
        self.title_index = title_index
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.feature_extractor = feature_extractor#(document_index, title_index, 1, document_preprocessor, stopwords, 2, 3)
        self.ranker = ranker
        self.model = None

        # TODO: Initialize the LambdaMART model (but don't train it yet)
        self.lMart = LambdaMART()






        # TODO: Initialize the LambdaMART model (but don't train it yet)
        # self.model = None # This should a LambdaMART object
        # pass

    def prepare_training_data(self, query_to_document_relevance_scores: dict[str, list[tuple[int, int]]]):
        """
        Prepares the training data for the learning-to-rank algorithm.

        Args:
            query_to_document_relevance_scores (dict): A dictionary of queries mapped to a list of 
                documents and their relevance scores for that query
                The dictionary has the following structure:
                    query_1_text: [(docid_1, relance_to_query_1), (docid_2, relance_to_query_2), ...]

        Returns:
            tuple: A tuple containing the training data in the form of three lists: x, y, and qgroups
                X (list): A list of feature vectors for each query-document pair
                y (list): A list of relevance scores for each query-document pair
                qgroups (list): A list of the number of documents retrieved for each query
        """
        # NOTE: qgroups is not the same length as X or y.
        # This is for LightGBM to know how many relevance scores we have per query.
        X = []
        y = []
        qgroups = []

        # TODO: for each query and the documents that have been rated for relevance to that query,
        # process these query-document pairs into features
        for query, doc_scores in query_to_document_relevance_scores.items():
            query_tokens = self.document_preprocessor.tokenize(query)
            query_word_counts = defaultdict(int)
            for token in query_tokens:
                query_word_counts[token] += 1

        # TODO: Accumulate the token counts for each document's title and content here
            doc_term_counts = self.accumulate_doc_term_counts(self.document_index, query_tokens)
            title_term_counts = self.accumulate_doc_term_counts(self.title_index, query_tokens)



        # TODO: For each of the documents, generate its features, then append
        # the features and relevance score to the lists to be returned
            for doc_id, rel_score in doc_scores:
                features = self.feature_extractor.generate_features(doc_id, doc_term_counts[doc_id], title_term_counts[doc_id], query_tokens)
                X.append(features)
                y.append(rel_score)

        # TODO: Make sure to keep track of how many scores we have for this query in qrels
            qgroups.append(len(doc_scores))

        return X, y, qgroups

    @staticmethod
    def accumulate_doc_term_counts(index: InvertedIndex, query_parts: list[str]) -> dict[int, dict[str, int]]:
        """
        A helper function that for a given query, retrieves all documents that have any
        of these words in the provided index and returns a dictionary mapping each document id to
        the counts of how many times each of the query words occurred in the document

        Args:
            index: An inverted index to search
            query_parts: A list of tokenized query tokens

        Returns:
            A dictionary mapping each document containing at least one of the query tokens to
            a dictionary with how many times each of the query words appears in that document
        """
        # TODO: Retrieve the set of documents that have each query word (i.e., the postings) and
        # create a dictionary that keeps track of their counts for the query word
        accum = defaultdict(lambda: defaultdict(int))
        for tok in query_parts:
            postings = index.get_postings(tok)

            for doc_id, count in postings:
                accum[doc_id][tok] += count

        return accum

    def train(self, training_data_filename: str) -> None:
        """
        Trains a LambdaMART pair-wise learning to rank model using the documents and relevance scores provided 
        in the training data file.

        Args:
            training_data_filename (str): a filename for a file containing documents and relevance scores
        """
        # TODO: Convert the relevance data into the right format for training data preparation

        query_to_document_relevance_scores = defaultdict(list)
        with open(training_data_filename, 'r', encoding='latin1') as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                query = row['query']
                docid = row["docid"]
                rel = row["rel"]
                query_to_document_relevance_scores[query].append((int(docid), int(float(rel))))

        # TODO: prepare the training data by featurizing the query-doc pairs and
        # getting the necessary datastructures
        X_train, y_train, qgroups_train = self.prepare_training_data(query_to_document_relevance_scores)

        # TODO: Train the model
        self.lMart.fit(X_train, y_train, qgroups_train)
        self.model = self.lMart

    def predict(self, X):
        """
        Predicts the ranks for featurized doc-query pairs using the trained model.

        Args:
            X (array-like): Input data to be predicted
                This is already featurized doc-query pairs.

        Returns:
            array-like: The predicted rank of each document

        Raises:
            ValueError: If the model has not been trained yet.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")

        # TODO: Return a prediction made using the LambdaMART model
        return self.lMart.predict(X)

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Retrieves potentially-relevant documents, constructs feature vectors for each query-document pair,
        uses the L2R model to rank these documents, and returns the ranked documents.

        Args:
            query: A string representing the query to be used for ranking

        Returns:
            A list containing tuples of the ranked documents and their scores, sorted by score in descending order
                The list has the following structure: [(doc_id_1, score_1), (doc_id_2, score_2), ...]
        """
        # TODO: Retrieve potentially-relevant documents
        scores = self.ranker.query(query)

        # TODO: Fetch a list of possible documents from the index and create a mapping from
        # a document ID to a dictionary of the counts of the query terms in that document.
        # You will pass the dictionary to the RelevanceScorer as input.
        #
        # NOTE: we collect these here (rather than calling a Ranker instance) because we'll
        # pass these doc-term-counts to functions later, so we need the accumulated representations
        query_tokens = self.document_preprocessor.tokenize(query)
        query_word_counts = defaultdict(int)
        for token in query_tokens:
            query_word_counts[token] += 1

        # TODO: Accumulate the documents word frequencies for the title and the main body
        doc_term_counts = self.accumulate_doc_term_counts(self.document_index, query_tokens)
        title_term_counts = self.accumulate_doc_term_counts(self.title_index, query_tokens)

        # TODO: Score and sort the documents by the provided scrorer for just the document's main text (not the title)
        # This ordering determines which documents we will try to *re-rank* using our L2R model
        # scores = [(doc_id, self.scorer.score(doc_id, doc_term_counts[doc_id], query_word_counts)) for doc_id, rel in rel_docs]
        # scores.sort(key=lambda x: x[1], reverse=True)

        # TODO: Filter to just the top 100 documents for the L2R part for re-ranking
        top_docs = scores[:100]

        # TODO: Construct the feature vectors for each query-document pair in the top 100
        feature_vectors = []
        for doc_id, score in top_docs:
            features = self.feature_extractor.generate_features(
                doc_id, doc_term_counts[doc_id], title_term_counts[doc_id], query_tokens
            )
            feature_vectors.append((doc_id, features))

        featurized_docs = [fv[1] for fv in feature_vectors]

        # TODO: Use your L2R model to rank these top 100 documents
        ranked_scores = self.lMart.predict(featurized_docs)

        # TODO: Sort posting_lists based on scores
        ranked_top_docs = [(feature_vectors[i][0], ranked_scores[i]) for i in range(len(ranked_scores))]
        ranked_top_docs.sort(key=lambda x: x[1], reverse=True)


        # TODO: Make sure to add back the other non-top-100 documents that weren't re-ranked
        final_ranking = ranked_top_docs + scores[100:]
        final_ranking.sort(key=lambda x: x[1], reverse=True)

        # TODO: Return the ranked documents
        return final_ranking


class L2RFeatureExtractor:
    def __init__(self, document_index: InvertedIndex, title_index: InvertedIndex,
                 doc_category_info: dict[int, list[str]],
                 document_preprocessor: Tokenizer, stopwords: set[str],
                 recognized_categories: set[str], docid_to_network_features: dict[int, dict[str, float]],
                 ce_scorer: CrossEncoderScorer) -> None:
        """
        Initializes a L2RFeatureExtractor object.

        Args:
            document_index: The inverted index for the contents of the document's main text body
            title_index: The inverted index for the contents of the document's title
            doc_category_info: A dictionary where the document id is mapped to a list of categories
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            recognized_categories: The set of categories to be recognized as binary features
                (whether the document has each one)
            docid_to_network_features: A dictionary where the document id is mapped to a dictionary
                with keys for network feature names "page_rank", "hub_score", and "authority_score"
                and values with the scores for those features
            ce_scorer: The CrossEncoderScorer object
        """
        # TODO: Set the initial state using the arguments
        self.doc_index = document_index
        self.title_index = title_index
        self.doc_category_info = doc_category_info
        self.document_preprocessor = document_preprocessor
        self.stopwords = stopwords
        self.docid_to_network_features = docid_to_network_features
        self.ce_scorer = ce_scorer

        # TODO: For the recognized categories (i.e,. those that are going to be features), considering
        # how you want to store them here for faster featurizing
        self.recognized_categories = recognized_categories

        # TODO (HW2): Initialize any RelevanceScorer objects you need to support the methods below.
        #             Be sure to use the right InvertedIndex object when scoring.


    # TODO: Article Length
    def get_article_length(self, docid: int) -> int:
        """
        Gets the length of a document (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document
        """
        return self.doc_index.get_doc_metadata(docid)["length"]

    # TODO: Title Length
    def get_title_length(self, docid: int) -> int:
        """
        Gets the length of a document's title (including stopwords).

        Args:
            docid: The id of the document

        Returns:
            The length of a document's title
        """
        return self.title_index.get_doc_metadata(docid)["length"]

    # TODO: TF
    def get_tf(self, index: InvertedIndex, docid: int, word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF score
        """

        tf_score = 0.0
        doc_total = sum(word_counts.values())

        if doc_total == 0:
            return tf_score

        for term in query_parts:
            if term in word_counts:
                term_total = word_counts[term] + 1
                tf_score += np.log(term_total)

        return tf_score

    # TODO: TF-IDF
    def get_tf_idf(self, index: InvertedIndex, docid: int,
                   word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Calculates the TF-IDF score.

        Args:
            index: An inverted index to use for calculating the statistics
            docid: The id of the document
            word_counts: The words in some part of a document mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The TF-IDF score
        """
        query_word_counts = defaultdict(lambda: 0)
        for tok in query_parts:
            query_word_counts[tok] += 1

        return TF_IDF(index).score(docid, word_counts, query_word_counts)

    def get_BM25_score(self, docid: int, doc_word_counts: dict[str, int],
                       query_parts: list[str]) -> float:
        """
        Calculates the BM25 score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The BM25 score
        """
        # TODO: Calculate the BM25 score and return it
        query_word_counts = defaultdict(lambda: 0)
        for tok in query_parts:
            query_word_counts[tok] += 1

        return BM25(self.doc_index).score(docid, doc_word_counts, query_word_counts)

    # TODO: Pivoted Normalization
    def get_pivoted_normalization_score(self, docid: int, doc_word_counts: dict[str, int],
                                        query_parts: list[str]) -> float:
        """
        Calculates the pivoted normalization score.

        Args:
            docid: The id of the document
            doc_word_counts: The words in the document's main text mapped to their frequencies
            query_parts: A list of tokenized query tokens

        Returns:
            The pivoted normalization score
        """
        # TODO: Calculate the pivoted normalization score and return it
        query_word_counts = defaultdict(lambda: 0)
        for tok in query_parts:
            query_word_counts[tok] += 1
        return PivotedNormalization(self.doc_index).score(docid, doc_word_counts, query_word_counts)

    # TODO: Document Categories
    def get_document_categories(self, docid: int) -> list:
        """
        Generates a list of binary features indicating which of the recognized categories that the document has.
        Category features should be deterministically ordered so list[0] should always correspond to the same
        category. For example, if a document has one of the three categories, and that category is mapped to
        index 1, then the binary feature vector would look like [0, 1, 0].

        Args:
            docid: The id of the document

        Returns:
            A list containing binary list of which recognized categories that the given document has.
        """
        accum = []
        for cat in self.recognized_categories:
            if cat in self.doc_category_info[docid]:
                accum.append(1)
            else:
                accum.append(0)

        return accum

    # TODO Pagerank score
    def get_pagerank_score(self, docid: int) -> float:
        """
        Gets the PageRank score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The PageRank score
        """
        return self.docid_to_network_features[docid]["pagerank"]

    # TODO HITS Hub score
    def get_hits_hub_score(self, docid: int) -> float:
        """
        Gets the HITS hub score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS hub score
        """
        return self.docid_to_network_features[docid]["hub_score"]

    # TODO HITS Authority score
    def get_hits_authority_score(self, docid: int) -> float:
        """
        Gets the HITS authority score for the given document.

        Args:
            docid: The id of the document

        Returns:
            The HITS authority score
        """
        return self.docid_to_network_features[docid]["authority_score"]

    # TODO (HW3): Cross-Encoder Score
    def get_cross_encoder_score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The Cross-Encoder score
        """
        return self.ce_scorer.score(docid, query)

    # TODO: Add at least one new feature to be used with your L2R model

    def generate_features(self, docid: int, doc_word_counts: dict[str, int],
                          title_word_counts: dict[str, int], query_parts: list[str]) -> list:
        """
        Generates a dictionary of features for a given document and query.

        Args:
            docid: The id of the document to generate features for
            doc_word_counts: The words in the document's main text mapped to their frequencies
            title_word_counts: The words in the document's title mapped to their frequencies
            query_parts : A list of tokenized query terms to generate features for

        Returns:
            A vector (list) of the features for this document
                Feature order should be stable between calls to the function
                (the order of features in the vector should not change).
        """

        feature_vector = []

        # TODO: Document Length

        # TODO: Title Length

        # TODO Query Length

        # TODO: TF (document)

        # TODO: TF-IDF (document)

        # TODO: TF (title)

        # TODO: TF-IDF (title)

        # TODO: BM25

        # TODO: Pivoted Normalization

        # TODO: Pagerank

        # TODO: HITS Hub

        # TODO: HITS Authority

        # TODO: (HW3) Cross-Encoder Score

        # TODO: Add at least one new feature to be used with your L2R model.

        # TODO: Calculate the Document Categories features.
        # NOTE: This should be a list of binary values indicating which categories are present.

        # return feature_vector

        feature_vector = []

        # TODO: Document Length
        feature_vector.append(self.get_article_length(docid))

        # TODO: Title Length
        feature_vector.append(self.get_title_length(docid))

        # TODO Query Length
        feature_vector.append(len(query_parts))

        # TODO: TF (document)
        feature_vector.append(self.get_tf(self.doc_index, docid, doc_word_counts, query_parts))

        # TODO: TF-IDF (document)
        feature_vector.append(self.get_tf_idf(self.doc_index, docid, doc_word_counts, query_parts))

        # TODO: TF (title)
        feature_vector.append(self.get_tf(self.title_index, docid, title_word_counts, query_parts))

        # TODO: TF-IDF (title)
        feature_vector.append(self.get_tf_idf(self.title_index, docid, title_word_counts, query_parts))

        # TODO: BM25
        feature_vector.append(self.get_BM25_score(docid, doc_word_counts, query_parts))

        # TODO: Pivoted Normalization
        feature_vector.append(self.get_pivoted_normalization_score(docid, doc_word_counts, query_parts))

        # TODO: Pagerank
        feature_vector.append(self.get_pagerank_score(docid))

        # TODO: HITS Hub
        feature_vector.append(self.get_hits_hub_score(docid))

        # TODO: HITS Authority
        feature_vector.append(self.get_hits_authority_score(docid))

        # TODO: (HW3) Cross-Encoder Score
        feature_vector.append(self.get_cross_encoder_score(docid, " ".join(query_parts)))

        # TODO: Calculate the Document Categories features.
        # NOTE: This should be a list of binary values indicating which categories are present.
        feature_vector.extend(self.get_document_categories(docid))

        return feature_vector


class LambdaMART:
    def __init__(self, params=None) -> None:
        """
        Initializes a LambdaMART (LGBRanker) model using the lightgbm library.

        Args:
            params (dict, optional): Parameters for the LGBMRanker model. Defaults to None.
        """
        default_params = {
            'objective': "lambdarank",
            'boosting_type': "gbdt",
            'n_estimators': 20,
            'importance_type': "gain",
            'metric': "ndcg",
            'num_leaves': 20,
            'learning_rate': 0.005,
            'max_depth': -1,
            # NOTE: You might consider setting this parameter to a higher value equal to
            # the number of CPUs on your machine for faster training
            "n_jobs": 1,
            # "verbosity": 1,
        }

        if params:
            default_params.update(params)

        # TODO: initialize the LGBMRanker with the provided parameters and assign as a field of this class
        self.ranker = lightgbm.LGBMRanker(default_params["boosting_type"], default_params["num_leaves"], default_params["max_depth"], default_params["learning_rate"], default_params["n_estimators"], objective=default_params["objective"], metric=default_params["metric"], importance_type=default_params["importance_type"], n_jobs=default_params["n_jobs"])

    def fit(self,  X_train, y_train, qgroups_train):
        """
        Trains the LGBMRanker model.

        Args:
            X_train (array-like): Training input samples.
            y_train (array-like): Target values.
            qgroups_train (array-like): Query group sizes for training data.

        Returns:
            self: Returns the instance itself.
        """

        # TODO: fit the LGBMRanker's parameters using the provided features and labels
        self.ranker.fit(X_train, y_train, group=qgroups_train)
        return self

    def predict(self, featurized_docs):
        """
        Predicts the target values for the given test data.

        Args:
            featurized_docs (array-like): 
                A list of featurized documents where each document is a list of its features
                All documents should have the same length.

        Returns:
            array-like: The estimated ranking for each document (unsorted)
        """

        # TODO: Generating the predicted values using the LGBMRanker
        return self.ranker.predict(featurized_docs)
