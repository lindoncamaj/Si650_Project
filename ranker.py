"""
This is the template for implementing the rankers for your search engine.
You will be implementing WordCountCosineSimilarity, DirichletLM, TF-IDF, BM25, Pivoted Normalization, and your own ranker.
"""
from indexing import InvertedIndex, BasicInvertedIndex, Indexer
from document_preprocessor import RegexTokenizer
import numpy as np
from sentence_transformers.cross_encoder import CrossEncoder
import csv
import time
from collections import Counter, defaultdict


class Ranker:
    """
    The ranker class is responsible for generating a list of documents for a given query, ordered by their scores
    using a particular relevance function (e.g., BM25).
    A Ranker can be configured with any RelevanceScorer.
    """

    def __init__(self, index: InvertedIndex, document_preprocessor, stopwords: set[str],
                 scorer: 'RelevanceScorer', raw_text_dict: dict[int, str] = None) -> None:
        """
        Initializes the state of the Ranker object.

        Args:
            index: An inverted index
            document_preprocessor: The DocumentPreprocessor to use for turning strings into tokens
            stopwords: The set of stopwords to use or None if no stopword filtering is to be done
            scorer: The RelevanceScorer object
        """
        self.index = index
        self.tokenize = document_preprocessor.tokenize
        self.scorer = scorer
        self.stopwords = stopwords
        self.raw_text_dict = raw_text_dict

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Searches the collection for relevant documents to the query and
        returns a list of documents ordered by their relevance (most relevant first).

        Args:
            query: The query to search for

        Returns:
            A sorted list containing tuples of the document id and its relevance score

        TODO (HW3): We are standardizing the query output of Ranker to match with L2RRanker.query and VectorRanker.query
        The query function should return a sorted list of tuples where each tuple has the first element as the document ID
        and the second element as the score of the document after the ranking process.

        """
        # 1. Tokenize query

        # 2. Fetch a list of possible documents from the index

        # 3. Run RelevanceScorer (like BM25 from below classes) (implemented as relevance classes)

        # 4. Return **sorted** results as format [(100, 0.5), (10, 0.2), ...]

        tokenized_query = self.tokenize(query)

        if not tokenized_query:
            return []


        docs = [self.index.get_postings(tok) for tok in tokenized_query]
        doc_ids = list(set([i[0] for doc in docs for i in doc]))
        doc_word_counts = {}
        query_word_counts = {}

        for i in range(0, len(docs)):
            if docs[i] == []:
                continue
            else:
                for j in docs[i]:
                    if j[0] not in doc_word_counts.keys():
                        doc_word_counts[j[0]] = {tokenized_query[i]: j[1]}
                    else:
                        if tokenized_query[i] in doc_word_counts[j[0]].keys():
                            continue
                        else:
                            doc_word_counts[j[0]][tokenized_query[i]] = j[1]

        for doc in doc_word_counts.keys():
            for token in tokenized_query:
                if token in doc_word_counts[doc].keys():
                            continue
                else:
                    doc_word_counts[doc][token] = 0


        for word in tokenized_query:
            if word in query_word_counts.keys():
                query_word_counts[word] += 1
            else:
                query_word_counts[word] = 1


        rel_score = []
        for doc in doc_ids:
            rel_score.append((doc, self.scorer.score(doc, doc_word_counts[doc], query_word_counts)))


        return sorted(rel_score, key=lambda tup: tup[1], reverse=True)




class RelevanceScorer:
    '''
    This is the base interface for all the relevance scoring algorithm.
    It will take a document and attempt to assign a score to it.
    '''
    # Implement the functions in the child classes (WordCountCosineSimilarity, DirichletLM, BM25, PivotedNormalization, TF_IDF) and not in this one

    def __init__(self, index, parameters) -> None:
        raise NotImplementedError

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        """
        Returns a score for how relevance is the document for the provided query.

        Args:
            docid: The ID of the document
            doc_word_counts: A dictionary containing all words in the document and their frequencies.
                Words that have been filtered will be None.
            query_word_counts: A dictionary containing all words in the query and their frequencies.
                Words that have been filtered will be None.

        Returns:
            A score for how relevant the document is (Higher scores are more relevant.)

        """
        raise NotImplementedError


class SampleScorer(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters) -> None:
        pass

    def score(self, docid: int, doc_word_counts: dict[str, int], query_parts: list[str]) -> float:
        """
        Scores all documents as 10.
        """
        return 10


# TODO Implement unnormalized cosine similarity on word count vectors
class WordCountCosineSimilarity(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Find the dot product of the word count vector of the document and the word count vector of the query

        # 2. Return the score
        score = 0

        for word, query_count in query_word_counts.items():
            if word in doc_word_counts:
                score += query_count * doc_word_counts[word]

        return score


# TODO Implement DirichletLM
class DirichletLM(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'mu': 2000}) -> None:
        self.index = index
        self.parameters = parameters

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query_parts, compute score

        # 4. Return the score
        doc_len = self.index.get_doc_metadata(docid)["length"]
        mu = self.parameters["mu"]

        score = 0
        for q_term in query_word_counts:
            if q_term and q_term in self.index.index:
                postings = self.index.get_postings(q_term)
                doc_tf = doc_word_counts[q_term]

                if doc_tf > 0:
                    query_tf = query_word_counts[q_term]
                    p_wc = sum([doc[1] for doc in postings]) / \
                        self.index.get_statistics()["total_token_count"]
                    tfidf = np.log(1 + (doc_tf / (mu * p_wc)))

                    score += (query_tf * tfidf)

        score += len(query_word_counts) * np.log(mu / (doc_len + mu))

        return score


# TODO Implement BM25
class BM25(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2, 'k3': 8}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.k3 = parameters['k3']
        self.N = len(self.index.document_metadata)
        self.avdl = (1 / self.N) * self.index.statistics["total_token_count"]

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return score
        doc_len = self.index.get_doc_metadata(docid)["length"]

        score = 0

        for q_term in query_word_counts:
            if q_term and q_term in self.index.index:
                doc_tf = doc_word_counts[q_term]
                query_tf = query_word_counts[q_term]
                n_docs_term = len(self.index.get_postings(q_term))

                if n_docs_term == 0 or doc_tf == 0:
                    continue

                idf = np.log((self.N - n_docs_term + 0.5) / (n_docs_term + 0.5))

                tf = ((self.k1 + 1) * doc_tf) / (self.k1 * (1 - self.b + self.b * (doc_len/self.avdl)) + doc_tf)

                qtf = ((self.k3 + 1) * query_tf) / (self.k3 + query_tf)

                score += (idf * tf * qtf)

        return score


# TODO Implement Pivoted Normalization
class PivotedNormalization(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.2}) -> None:
        self.index = index
        self.b = parameters['b']
        self.num_docs = len(self.index.document_metadata)
        self.avdl = (1 / self.num_docs) * self.index.statistics["total_token_count"]

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF, IDF, and QTF values to get a score

        # 4. Return the score
        doc_len = self.index.get_doc_metadata(docid)["length"]
        score = 0

        for q_term in query_word_counts:
            if q_term in self.index.index:
                doc_tf = doc_word_counts[q_term]
                query_tf = query_word_counts[q_term]
                n_docs_term = len(self.index.get_postings(q_term))

                if n_docs_term == 0 or doc_tf == 0:
                    continue

                tf = (1 + np.log(1 + np.log(doc_tf))) / (1 - self.b + self.b * (doc_len / self.avdl))
                idf = np.log((self.num_docs + 1) / n_docs_term)


                score += (idf * tf * query_tf)

        return score


# TODO Implement TF-IDF
class TF_IDF(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {}) -> None:
        self.index = index
        self.parameters = parameters
        self.num_docs = len(self.index.document_metadata)

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int]) -> float:
        # 1. Get necessary information from index

        # 2. Compute additional terms to use in algorithm

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return the score
        doc_len = self.index.get_doc_metadata(docid)["length"]
        score = 0

        for q_term in query_word_counts:
            if q_term in self.index.index:
                doc_tf = doc_word_counts[q_term]
                query_tf = query_word_counts[q_term]
                n_docs_term = len(self.index.get_postings(q_term))

                if n_docs_term == 0 or doc_tf == 0:
                    continue

                tf = np.log(doc_tf + 1)
                idf = np.log((self.num_docs / n_docs_term)) + 1


                score += (idf * tf * query_tf)

        return score


# TODO Implement your own ranker with proper heuristics
class YourRanker(RelevanceScorer):
    def __init__(self, index: InvertedIndex, parameters: dict = {'b': 0.75, 'k1': 1.2}) -> None:
        self.index = index
        self.b = parameters['b']
        self.k1 = parameters['k1']
        self.N = len(self.index.document_metadata)
        self.avdl = (1 / self.N) * self.index.statistics["total_token_count"]

    def score(self, docid: int, doc_word_counts: dict[str, int], query_word_counts: dict[str, int])-> float:
        # 1. Get necessary information from index

        # 2. Find the dot product of the word count vector of the document and the word count vector of the query

        # 3. For all query parts, compute the TF and IDF to get a score

        # 4. Return score
        doc_len = self.index.get_doc_metadata(docid)["length"]
        score = 0

        for q_term in query_word_counts:
            if q_term and q_term in self.index.index:
                doc_tf = doc_word_counts[q_term]
                query_tf = query_word_counts[q_term]
                n_docs_term = len(self.index.get_postings(q_term))

                if n_docs_term == 0 or doc_tf == 0:
                    continue

                idf = np.log((self.N - n_docs_term + 0.5) / (n_docs_term + 0.5))

                tf = np.log(((self.k1 + 1) * doc_tf) / (self.k1 * (1 - self.b + self.b * (doc_len/self.avdl)) + doc_tf))

                score += (idf * tf * query_tf)

        return score


# TODO (HW3): The CrossEncoderScorer class uses a pre-trained cross-encoder model from the Sentence Transformers package
#             to score a given query-document pair; check README for details
#
# NOTE: This is not a RelevanceScorer object because the method signature for score() does not match, but it
# has the same intent, in practice
class CrossEncoderScorer:
    '''
    A scoring object that uses cross-encoder to compute the relevance of a document for a query.
    '''
    def __init__(self, raw_text_dict: dict[int, str], 
                 cross_encoder_model_name: str = 'cross-encoder/msmarco-MiniLM-L6-en-de-v1') -> None:
        """
        Initializes a CrossEncoderScorer object.

        Args:
            raw_text_dict: A dictionary where the document id is mapped to a string with the first 500 words
                in the document
            cross_encoder_model_name: The name of a cross-encoder model
        """
        # TODO: Save any new arguments that are needed as fields of this class
        self.raw_text_dict = raw_text_dict
        self.model = CrossEncoder(cross_encoder_model_name)

    def score(self, docid: int, query: str) -> float:
        """
        Gets the cross-encoder score for the given document.

        Args:
            docid: The id of the document
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            The score returned by the cross-encoder model
        """
        # NOTE: Do not forget to handle an edge case
        # (e.g., docid does not exist in raw_text_dict or empty query, both should lead to 0 score)
        if docid not in self.raw_text_dict or not query:
            return 0

        # NOTE: unlike the other scorers like BM25, this method takes in the query string itself,
        # not the tokens!

        # TODO (HW3): Get a score from the cross-encoder model
        #             Refer to IR_Encoder_Examples.ipynb in Demos folder on Canvas if needed
        document = self.raw_text_dict[docid]
        score = self.model.predict([(query, document)])

        return score[0]



if __name__ == "__main__":
    document_preprocessor = RegexTokenizer()
    stopwords = set()
    print("start creating indexs")
    index = Indexer.create_index(BasicInvertedIndex, '../data/wikipedia_10k_dataset.jsonl', document_preprocessor, stopwords, 0)

    start_time = time.time()
    prev_time = time.time()
    time_to_process_split = []

    relevance_data = {}
    queries = []
    with open("../relevance.test.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        headers = next(csv_reader)
        for row in csv_reader:
            query = row[0]
            doc_id = int(row[2])
            score = int(row[4])
            if query not in relevance_data:
                relevance_data[query] = []
            if query not in queries:
                queries.append(query)
            relevance_data[query].append((doc_id, score))


    scorers = [{"name": "WordCountCosineSimilarity", "scorer": WordCountCosineSimilarity(index)}, {"name": "BM25", "scorer": BM25(index)}, {"name": "PivotedNormalization", "scorer": PivotedNormalization(index)}, {"name": "DirchletLM", "scorer": DirichletLM(index)}, {"name": "TF_IDF", "scorer": TF_IDF(index)}, {"name": "MyRanker", "scorer": YourRanker(index)}]
    # results = []
    for scorer in scorers:
        ranker = Ranker(index=index, document_preprocessor=document_preprocessor, stopwords=stopwords, scorer=scorer['scorer'])

        print("New Test")

        for q in queries:
            ranker.query(q)


        time_to_process_split.append(time.time() - prev_time)
        prev_time = time.time()
        # print(result)
    final_time = time.time()

    print(time_to_process_split)