'''
Here you will be implemeting the indexing strategies for your search engine. You will need to create, persist and load the index.
This will require some amount of file handling.
DO NOT use the pickle module.
'''

from enum import Enum
from document_preprocessor import Tokenizer, RegexTokenizer
from collections import Counter, defaultdict
import json
import sys
import time
from tqdm import tqdm
import os
import gzip


class IndexType(Enum):
    # the two types of index currently supported are BasicInvertedIndex, PositionalIndex
    PositionalIndex = 'PositionalIndex'
    BasicInvertedIndex = 'BasicInvertedIndex'
    SampleIndex = 'SampleIndex'


class InvertedIndex:
    """
    This class is the basic implementation of an in-memory inverted index. This class will hold the mapping of terms to their postings.
    The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
    and documents in the index. These metadata will be necessary when computing your relevance functions.
    """

    def __init__(self) -> None:
        """
        An inverted index implementation where everything is kept in memory
        """
        self.statistics = {}   # the central statistics of the index
        self.statistics['vocab'] = Counter()  # token count
        self.statistics["total_token_count"] = 0
        self.statistics["mean_document_length"] = 0
        self.statistics["number_of_documents"] = 0
        self.statistics["unique_token_count"] = 0
        self.vocabulary = set()  # the vocabulary of the collection
        # metadata like length, number of unique tokens of the documents
        self.document_metadata = {}

        self.index = defaultdict(list)  # the index

    # NOTE: The following functions have to be implemented in the two inherited classes and not in this class

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        raise NotImplementedError

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        raise NotImplementedError

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        raise NotImplementedError

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        raise NotImplementedError

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        raise NotImplementedError

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        raise NotImplementedError

    def save(self, index_directory_name: str) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        raise NotImplementedError

    def load(self, index_directory_name: str) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        raise NotImplementedError


class BasicInvertedIndex(InvertedIndex):
    def __init__(self) -> None:
        """
        This is the typical inverted index where each term keeps track of documents and the term count per document.
        This class will hold the mapping of terms to their postings.
        The class also has functions to save and load the index to/from disk and to access metadata about the index and the terms
        and documents in the index. These metadata will be necessary when computing your ranker functions.
        """
        super().__init__()
        self.statistics['index_type'] = 'BasicInvertedIndex'


    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        for token in self.index:
            if not any(docid == t[0] for t in token):
                continue
            else:
                ind = 0
                for i in range(0, len(self.index[token])):
                        if self.index[token][i][0] == docid:
                            ind = i

                self.index[token].pop(ind)
        self.document_metadata.pop(docid)
        self.statistics["number_of_documents"] -= 1

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        # # [a,b,c,d,a,b]
        # # {a:3, b:2, c:1, d:1}
        token_counts = Counter(token for token in tokens if token)

        for token, token_count in token_counts.items():
            if token not in self.index:
                self.index[token] = {}

            self.index[token][docid] = token_count

            self.statistics['vocab'][token] += token_count

        unique_token_count = len(token_counts)
        self.document_metadata[docid] = {"unique_tokens": unique_token_count, "length": len(tokens)}

        self.statistics["total_token_count"] = self.statistics.get("total_token_count", 0) + len(tokens)
        self.statistics["stored_total_token_count"] = self.statistics.get("stored_total_token_count", 0) + unique_token_count

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        if term in self.index:
            return list(self.index[term].items())
        else:
            return []


    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        return self.document_metadata[doc_id]

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        return {"term_count": self.statistics["vocab"][term], "doc_frequency": len(self.index[term])}

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        self.statistics["unique_token_count"] = len(self.index)
        self.statistics["number_of_documents"] = len(self.document_metadata)
        if self.statistics["number_of_documents"]:
            self.statistics["mean_document_length"] = self.statistics["total_token_count"] / self.statistics["number_of_documents"]
        else:
            self.statistics["mean_document_length"] = 0
        return self.statistics

    def save(self, directory) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        os.makedirs(directory, exist_ok=True)
        with open(f"{directory}/inverted_index.json", 'w') as file:
            json.dump(self.index, file)

        with open(f"{directory}/inverted_index_metadata.json", 'w') as file:
            json.dump(self.document_metadata, file)

    def load(self, directory) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        os.makedirs(directory, exist_ok=True)
        with open(f"{directory}/inverted_index.json", 'r') as file:
            self.index = json.load(file)

        with open(f"{directory}/inverted_index_metadata.json", 'r') as file:
            metadata = json.load(file)

        self.document_metadata = {int(k):v for k,v in metadata.items()}


class PositionalInvertedIndex(BasicInvertedIndex):
    def __init__(self) -> None:
        """
        This is the positional index where each term keeps track of documents and positions of the terms
        occurring in the document.
        """
        super().__init__()

    def remove_doc(self, docid: int) -> None:
        """
        Removes a document from the index and updates the index's metadata on the basis of this
        document's deletion.

        Args:
            docid: The id of the document
        """
        for token in self.index:
            if not any(docid == t[0] for t in token):
                continue
            else:
                ind = 0
                for i in range(0, len(self.index[token])):
                        if self.index[token][i][0] == docid:
                            ind = i

                self.index[token].pop(ind)
        self.document_metadata.pop(docid)
        self.statistics["number_of_documents"] -= 1

    def add_doc(self, docid: int, tokens: list[str]) -> None:
        """
        Add a document to the index and update the index's metadata on the basis of this
        document's condition (e.g., collection size, average document length).

        Args:
            docid: The id of the document
            tokens: The tokens of the document
                Tokens that should not be indexed will have been replaced with None in this list.
                The length of the list should be equal to the number of tokens prior to any token removal.
        """
        token_counts = Counter(token for token in tokens if token)

        token_positions = defaultdict(list)
        for index, token in enumerate(tokens):
            if token:
                token_positions[token].append(index)


        for token, token_count in token_counts.items():
            if token not in self.index:
                self.index[token] = {}

            self.index[token][docid] = (token_count, token_positions[token])

            self.statistics['vocab'][token] += token_count

        unique_token_count = len(token_counts)
        self.document_metadata[docid] = {"unique_tokens": unique_token_count, "length": len(tokens)}

        self.statistics["total_token_count"] = self.statistics.get("total_token_count", 0) + len(tokens)
        self.statistics["stored_total_token_count"] = self.statistics.get("stored_total_token_count", 0) + unique_token_count

    def get_postings(self, term: str) -> list:
        """
        Returns the list of postings, which contains (at least) all the documents that have that term.
        In most implementation, this information is represented as list of tuples where each tuple
        contains the docid and the term's frequency in that document.

        Args:
            term: The term to be searched for

        Returns:
            A list of tuples containing a document id for a document
            that had that search term and an int value indicating the term's frequency in
            the document
        """
        if term in self.index:
            return [(docid, count, positions) for docid, (count, positions) in self.index[term].items()]
        else:
            return []

    def get_doc_metadata(self, doc_id: int) -> dict[str, int]:
        """
        For the given document id, returns a dictionary with metadata about that document.
        Metadata should include keys such as the following:
            "unique_tokens": How many unique tokens are in the document (among those not-filtered)
            "length": how long the document is in terms of tokens (including those filtered)

        Args:
            docid: The id of the document

        Returns:
            A dictionary with metadata about the document
        """
        return self.document_metadata[doc_id]

    def get_term_metadata(self, term: str) -> dict[str, int]:
        """
        For the given term, returns a dictionary with metadata about that term in the index.
        Metadata should include keys such as the following:
            "term_count": How many times this term appeared in the corpus as a whole
            "doc_frequency": How many documents contain this term

        Args:
            term: The term to be searched for

        Returns:
            A dictionary with metadata about the term in the index
        """
        return {"term_count": self.statistics["vocab"][term], "doc_frequency": len(self.index[term])}

    def get_statistics(self) -> dict[str, int]:
        """
        Returns a dictionary with properties and their values for the index.
        Keys should include at least the following:
            "unique_token_count": how many unique terms are in the index
            "total_token_count": how many total tokens are indexed including filterd tokens),
                i.e., the sum of the lengths of all documents
            "stored_total_token_count": how many total tokens are indexed excluding filterd tokens
            "number_of_documents": the number of documents indexed
            "mean_document_length": the mean number of tokens in a document (including filter tokens)

        Returns:
            A dictionary mapping statistical properties (named as strings) about the index to their values
        """
        self.statistics["unique_token_count"] = len(self.index)
        self.statistics["number_of_documents"] = len(self.document_metadata)
        if self.statistics["number_of_documents"]:
            self.statistics["mean_document_length"] = self.statistics["total_token_count"] / self.statistics["number_of_documents"]
        else:
            self.statistics["mean_document_length"] = 0
        return self.statistics

    def save(self, directory) -> None:
        """
        Saves the state of this index to the provided directory.
        The save state should include the inverted index as well as
        any metadata need to load this index back from disk.

        Args:
            index_directory_name: The name of the directory where the index will be saved
        """
        os.makedirs(directory, exist_ok=True)
        with open(f"{directory}/inverted_index.json", 'w') as file:
            json.dump(self.index, file)

        with open(f"{directory}/inverted_index_metadata.json", 'w') as file:
            json.dump(self.document_metadata, file)

    def load(self, directory) -> None:
        """
        Loads the inverted index and any associated metadata from files located in the directory.
        This method will only be called after save() has been called, so the directory should
        match the filenames used in save(). Note that you call this function on an empty index object.

        Args:
            index_directory_name: The name of the directory that contains the index
        """
        os.makedirs(directory, exist_ok=True)
        with open(f"{directory}/inverted_index.json", 'r') as file:
            self.index = json.load(file)

        with open(f"{directory}/inverted_index_metadata.json", 'r') as file:
            metadata = json.load(file)

        self.document_metadata = {int(k):v for k,v in metadata.items()}


class Indexer:
    '''
    The Indexer class is responsible for creating the index used by the search/ranking algorithm.
    '''

    @staticmethod
    def create_index(index_type: IndexType, dataset_folder: str,
                     document_preprocessor: Tokenizer, stopwords: set[str],
                     minimum_word_frequency: int, text_key="text",
                     max_docs: int = -1, doc_augment_dict: dict[int, list[str]] | None = None) -> InvertedIndex:
        '''
        This function is responsible for going through the documents one by one and inserting them into the index after tokenizing the document

        Args:
            index_type: This parameter tells you which type of index to create, e.g., BasicInvertedIndex
            dataset_path: The file path to your dataset
            document_preprocessor: A class which has a 'tokenize' function which would read each document's text and return a list of valid tokens
            stopwords: The set of stopwords to remove during preprocessing or 'None' if no stopword filtering is to be done
            minimum_word_frequency: An optional configuration which sets the minimum word frequency of a particular token to be indexed
                If the token does not appear in the entire corpus at least for the set frequency, it will not be indexed.
                Setting a value of 0 will completely ignore the parameter.
            text_key: The key in the JSON to use for loading the text
            max_docs: The maximum number of documents to index
                Documents are processed in the order they are seen.
            doc_augment_dict: An optional argument; This is a dict created from the doc2query.csv where the keys are
                the document id and the values are the list of queries for a particular document.

        Returns:
            An inverted index

        '''
         # TODO (HW3): This function now has an optional argument doc_augment_dict; check README.md

        # HINT: Think of what to do when doc_augment_dict exists, how can you deal with the extra information?
        #       How can you use that information with the tokens?
        #       If doc_augment_dict doesn't exist, it's the same as before, tokenizing just the document text

        # TODO: Implement this class properly. This is responsible for going through the documents
        #       one by one and inserting them into the index after tokenizing the document

        # TODO: Figure out what type of InvertedIndex to create.
        #       For HW3, only the BasicInvertedIndex is required to be supported

        # TODO: If minimum word frequencies are specified, process the collection to get the
        #       word frequencies

        # NOTE: Make sure to support both .jsonl.gz and .jsonl as input

        # TODO: Figure out which set of words to not index because they are stopwords or
        #       have too low of a frequency

        # TODO: Read the collection and process/index each document.
        #       Only index the terms that are not stopwords and have high-enough frequency

        if index_type == BasicInvertedIndex or index_type == IndexType.BasicInvertedIndex:
            indexer = BasicInvertedIndex()
        else:
            indexer = PositionalInvertedIndex()


        global_term_freqs = Counter()

        # with open(dataset_path, 'r') as json_file:
        #     dataset = []
        #     for i, line in tqdm(enumerate(json_file), total = max_docs):
        #         if max_docs != -1 and i+1 > max_docs:
        #             break
        #         doc = json.loads(line)
        #         text = doc[text_key]

        #         if doc_augment_dict:
        #             queries = " ".join(doc_augment_dict[doc["docid"]])
        #             text += " " + queries


        #         tokens = document_preprocessor.tokenize(text)

        #         # Remove stopwords and replace them with None to maintain positions
        #         processed_tokens = []
        #         for token in tokens:
        #             if token.lower() in stopwords:
        #                 processed_tokens.append(None)
        #             else:
        #                 processed_tokens.append(token)
        #                 global_term_freqs[token] += 1

        #         dataset.append((doc["docid"], processed_tokens))
        dataset = []
        doc_count = 0

        effective_stopwords = stopwords if stopwords is not None else set()
        filenames = sorted([f for f in os.listdir(dataset_folder) if f.endswith('.json')])

        for filename in filenames:
            file_path = os.path.join(dataset_folder, filename)

            with open(file_path, 'r', encoding='utf-8') as json_file:
                try:
                    content = json.load(json_file)
                    documents = content if isinstance(content, list) else [content]

                    for doc in tqdm(documents, desc=f"Processing {filename}"):
                        if max_docs != -1 and doc_count >= max_docs:
                            break
                        if isinstance(doc, dict):
                            text = doc.get(text_key, '')

                            if doc_augment_dict is not None and "docid" in doc:
                                queries = " ".join(doc_augment_dict.get(doc["docid"], []))
                                text += " " + queries

                            tokens = document_preprocessor.tokenize(text)

                            # Remove stopwords and replace them with None to maintain positions
                            processed_tokens = []
                            for token in tokens:
                                if token.lower() in effective_stopwords:
                                    processed_tokens.append(None)
                                else:
                                    processed_tokens.append(token)
                                    global_term_freqs[token] += 1

                            dataset.append((str(doc["docid"]), processed_tokens))
                            doc_count += 1
                        else:
                            print(f"Skipping invalid JSON object in file {filename}")
                    if max_docs != -1 and doc_count >= max_docs:
                        break  # Break the outer loop as well if max_docs reached
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")

            if max_docs != -1 and doc_count >= max_docs:
                break  # Ensure outer loop exits once max_docs is reached






        # Second pass to filter tokens based on global term frequencies and add documents to the index
        for docid, tokens in tqdm(dataset, total=len(dataset)):
            if minimum_word_frequency > 0:
                filtered_tokens = [token if token is None or global_term_freqs[token] >= minimum_word_frequency else None for token in tokens]
            else:
                filtered_tokens = tokens
            indexer.add_doc(docid, filtered_tokens)

        print(indexer)
        return indexer


'''
The following class is a stub class with none of the essential methods implemented. It is merely here as an example.
'''


class SampleIndex(InvertedIndex):
    '''
    This class does nothing of value
    '''

    def add_doc(self, docid, tokens):
        """Tokenize a document and add term ID """
        for token in tokens:
            if token not in self.index:
                self.index[token] = {docid: 1}
            else:
                self.index[token][docid] = 1

    def save(self):
        print('Index saved!')
