from sentence_transformers import SentenceTransformer, util
from numpy import ndarray
from ranker import Ranker
import numpy as np


class VectorRanker(Ranker):
    def __init__(self, bi_encoder_model_name: str, encoded_docs: ndarray,
                 row_to_docid: list[int]) -> None:
        """
        Initializes a VectorRanker object.

        Args:
            bi_encoder_model_name: The name of a huggingface model to use for initializing a 'SentenceTransformer'
            encoded_docs: A matrix where each row is an already-encoded document, encoded using the same encoded
                as specified with bi_encoded_model_name
            row_to_docid: A list that is a mapping from the row number to the document id that row corresponds to
                the embedding

        Using zip(encoded_docs, row_to_docid) should give you the mapping between the docid and the embedding.
        """
        # TODO: Instantiate the bi-encoder model here

        # NOTE: we're going to use the cpu for everything here so if you decide to use a GPU, do not
        # submit that code to the autograder
        self.biencoder_model = SentenceTransformer(bi_encoder_model_name) # Initialize the bi-encoder model here

        # TODO: Also include other necessary initialization code
        self.encoded_docs = encoded_docs
        self.row_to_docid = row_to_docid

    def query(self, query: str) -> list[tuple[int, float]]:
        """
        Encodes the query and then scores the relevance of the query with all the documents.

        Args:
            query: The query in its original form (no stopword filtering/tokenization)

        Returns:
            A sorted list of tuples containing the document id and its relevance to the query,
            with most relevant documents first or an empty list if a query cannot be encoded
            or no results are return
        """
        # NOTE: Do not forget to handle edge cases on the inputs
        if not query:
            return []

        # TODO: Encode the query using the bi-encoder
        embeddings = self.biencoder_model.encode(query)
        print(f"Document embeddings shape: {self.encoded_docs.shape}")
        print(f"Query embedding shape: {embeddings.shape}")
        print(query)

        # TODO: Score the similarity of the query vector and document vectors for relevance
        # Calculate the dot products between the query embedding and all document embeddings
        similarities = np.dot(self.encoded_docs, embeddings)

        # TODO: Generate the ordered list of (document id, score) tuples
        rel_docs = [(self.row_to_docid[idx], score) for idx, score in enumerate(similarities)]

        # TODO: Sort the list so most relevant are first
        return sorted(rel_docs, key=lambda tup: tup[1], reverse=True)

