from indexing import Indexer, BasicInvertedIndex
from document_preprocessor import RegexTokenizer

document_preprocessor = RegexTokenizer()
indexer = Indexer()

index = indexer.create_index(BasicInvertedIndex, "data/", document_preprocessor, None, 0, "Title", 20)
index.save("main_index")