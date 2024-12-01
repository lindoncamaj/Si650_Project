"""
This is the template for implementing the tokenizer for your search engine.
You will be testing some tokenization techniques.
"""
from nltk.tokenize import RegexpTokenizer
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import numpy as np
import spacy
import re
import time

class Tokenizer:
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        A generic class for objects that turn strings into sequences of tokens.
        A tokenizer can support different preprocessing options or use different methods
        for determining word breaks.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
        """
        # TODO: Save arguments that are needed as fields of this class
        self.lowercase = lowercase
        self.multiword_expressions = multiword_expressions

    def postprocess(self, input_tokens: list[str]) -> list[str]:
        """
        Performs any set of optional operations to modify the tokenized list of words such as
        lower-casing and multi-word-expression handling. After that, return the modified list of tokens.

        Args:
            input_tokens: A list of tokens

        Returns:
            A list of tokens processed by lower-casing depending on the given condition
        """
        # TODO: Add support for lower-casing and multi-word expressions
        if self.lowercase:
            input_tokens = [token.lower() for token in input_tokens]

        if self.multiword_expressions:
            # Sort multi-word expressions by length in descending order
            self.multiword_expressions.sort(key=lambda x: len(x.split()), reverse=True)

            # Recombine split tokens with apostrophes if they match any multi-word expression part
            i = 0
            while i < len(input_tokens) - 1:
                combined_token = input_tokens[i] + input_tokens[i + 1]
                if (
                    re.match(r"\w+", input_tokens[i]) 
                    and input_tokens[i + 1] in ["'s", "'re", "'ll", "'ve", "'d", "'m", "n't"]
                    and any(combined_token in expr for expr in self.multiword_expressions)
                ):
                    input_tokens[i] = combined_token
                    del input_tokens[i + 1]
                else:
                    i += 1

            # Process multi-word expressions
            i = 0
            while i < len(input_tokens):
                matched = False
                for expression in self.multiword_expressions:
                    expr_tokens = expression.split()
                    expr_len = len(expr_tokens)

                    # Check if current slice matches the longest expression
                    if input_tokens[i:i + expr_len] == expr_tokens:
                        input_tokens[i:i + expr_len] = [" ".join(expr_tokens)]
                        matched = True
                        break

                if not matched:
                    i += 1

        return input_tokens

    def tokenize(self, text: str) -> list[str]:
        """
        Splits a string into a list of tokens and performs all required postprocessing steps.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # You should implement this in a subclass, not here
        raise NotImplementedError(
            'tokenize() is not implemented in the base class; please use a subclass')


class SplitTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses the split function to tokenize a given string.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        """
        Split a string into a list of tokens using whitespace as a delimiter.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        if len(text) > 0:
            text_list = text.split(" ")

            return super().postprocess(text_list)
        else:
            return []


class RegexTokenizer(Tokenizer):
    def __init__(self, token_regex: str = '\w+', lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Uses NLTK's RegexpTokenizer to tokenize a given string.

        Args:
            token_regex: Use the following default regular expression pattern: '\w+'
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)
        # TODO: Save a new argument that is needed as a field of this class
        # TODO: Initialize the NLTK's RegexpTokenizer
        self.token_regex = RegexpTokenizer(token_regex)

    def tokenize(self, text: str) -> list[str]:
        """
        Uses NLTK's RegexTokenizer and a regular expression pattern to tokenize a string.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        # TODO: Tokenize the given text and perform postprocessing on the list of tokens
        #       using the postprocess function
        text_list = self.token_regex.tokenize(text)

        return super().postprocess(text_list)


class SpaCyTokenizer(Tokenizer):
    def __init__(self, lowercase: bool = True, multiword_expressions: list[str] = None) -> None:
        """
        Use a spaCy tokenizer to convert named entities into single words. 
        Check the spaCy documentation to learn about the feature that supports named entity recognition.

        Args:
            lowercase: Whether to lowercase all the tokens
            multiword_expressions: A list of strings that should be recognized as single tokens
                If set to 'None' no multi-word expression matching is performed.
                No need to perform/implement multi-word expression recognition for HW3; you can ignore this.
        """
        super().__init__(lowercase, multiword_expressions)

    def tokenize(self, text: str) -> list[str]:
        """
        Use a spaCy tokenizer to convert named entities into single words.

        Args:
            text: An input text you want to tokenize

        Returns:
            A list of tokens
        """
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        text_list = [w.text for w in doc]

        return super().postprocess(text_list)


# TODO (HW3): Take in a doc2query model and generate queries from a piece of text
# Note: This is just to check you can use the models;
#       for downstream tasks such as index augmentation with the queries, use doc2query.csv
class Doc2QueryAugmenter:
    """
    This class is responsible for generating queries for a document.
    These queries can augment the document before indexing.

    MUST READ: https://huggingface.co/doc2query/msmarco-t5-base-v1

    OPTIONAL reading
        1. Document Expansion by Query Prediction (Nogueira et al.): https://arxiv.org/pdf/1904.08375.pdf
    """

    def __init__(self, doc2query_model_name: str = 'doc2query/msmarco-t5-base-v1') -> None:
        """
        Creates the T5 model object and the corresponding dense tokenizer.

        Args:
            doc2query_model_name: The name of the T5 model architecture used for generating queries
        """
        self.device = torch.device('cpu')  # Do not change this unless you know what you are doing

        # TODO (HW3): Create the dense tokenizer and query generation model using HuggingFace transformers
        self.tokenizer = T5Tokenizer.from_pretrained(doc2query_model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(doc2query_model_name)

    def get_queries(self, document: str, n_queries: int = 5, prefix_prompt: str = '') -> list[str]:
        """
        Steps
            1. Use the dense tokenizer/encoder to create the dense document vector.
            2. Use the T5 model to generate the dense query vectors (you should have a list of vectors).
            3. Decode the query vector using the tokenizer/decode to get the appropriate queries.
            4. Return the queries.

            Ensure you take care of edge cases.

        OPTIONAL (DO NOT DO THIS before you finish the assignment):
            Neural models are best performing when batched to the GPU.
            Try writing a separate function which can deal with batches of documents.

        Args:
            document: The text from which queries are to be generated
            n_queries: The total number of queries to be generated
            prefix_prompt: An optional parameter that gets added before the text.
                Some models like flan-t5 are not fine-tuned to generate queries.
                So we need to add a prompt to instruct the model to generate queries.
                This string enables us to create a prefixed prompt to generate queries for the models.
                See the PDF for what you need to do for this part.
                Prompt-engineering: https://en.wikipedia.org/wiki/Prompt_engineering

        Returns:
            A list of query strings generated from the text
        """
        # Note: Feel free to change these values to experiment
        if n_queries <= 0 or document == "":
            return []

        document_max_token_length = 400  # as used in OPTIONAL Reading 1
        top_p = 0.85

        # NOTE: See https://huggingface.co/doc2query/msmarco-t5-base-v1 for details
        #ADDED BELOW

        text = prefix_prompt + document
        input_ids = self.tokenizer.encode(text, max_length=document_max_token_length, truncation=True, return_tensors='pt')
        outputs = self.model.generate(
            input_ids=input_ids,
            max_length=64,
            do_sample=True,
            top_p=top_p,
            num_return_sequences=n_queries)


        # TODO (HW3): For the given model, generate a list of queries that might reasonably be issued to search
        #       for that document
        # NOTE: Do not forget edge cases
        #ADDED BELOW
        queries = []
        for i in range(len(outputs)):
            query = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            queries.append(query)


        return queries


class RestaurantTokenizer(Tokenizer):
    def __init__(self, lowercase=True, multiword_expressions=None):
        super().__init__(lowercase, multiword_expressions)
        self.tokenizer = RegexTokenizer(lowercase=self.lowercase, multiword_expressions=self.multiword_expressions)

    def flatten_data(self, data):
        """
        Recursively flattens a dictionary or a list into a single string.
        """
        flattened_str = ""

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (dict, list)):
                    flattened_str += self.flatten_data(value) + " "
                else:
                    flattened_str += f"{value} "
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    flattened_str += self.flatten_data(item) + " "
                else:
                    flattened_str += f"{item} "

        return flattened_str.strip()

    def tokenize_document(self, doc: dict) -> list[str]:
        text = self.flatten_data(doc)
        tokens = self.tokenizer.tokenize(text)
        return tokens




# Don't forget that you can have a main function here to test anything in the file
if __name__ == '__main__':
    pass
