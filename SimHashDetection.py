from HSmodule import *
from source.FaissSearch import FaissSearch
from source.Preprocessor import Shingling, TextEmbedder

class SimHashDetection:
    def __init__(self):
        self.preprocessor = TextEmbedder()

        self.hasher = SimHash()
        self.outputDim = 128

        self.searcher = FaissSearch()
        self.searcher.threshold = 0.2
        self.searcher.setDisFunc("cosine")
        self.searcher.dim = self.outputDim

    def detect(self, ListOfText : list ):
        ListOfVecRecord = self.preprocessor(ListOfText)  

        # cấu hình hasher cho phù hợp với kích thước vector đầu vào
        sizeOfVector = len(ListOfVecRecord[0].vec)
        self.hasher.setInOutput( sizeOfVector, self.outputDim)

        listVecHashed = self.hasher.hash(ListOfVecRecord)

        clusters = self.searcher.classify(listVecHashed)

        # clustering đang là dạng list of list of VecRecord
        
        return clusters
