from typing import List


class CountVectorizer:
    """
    Converts text to vector
    """

    def __init__(self):
        self._vocabulary = []

    @staticmethod
    def create_vocabulary(corp: List[str]) -> List[str]:
        words = []
        for text in corp:
            for word in text.lower().split(' '):
                if word not in words:
                    words.append(word)
        return words

    def fit_transform(self, corp: List[str]) -> List[int]:
        self._vocabulary = self.create_vocabulary(corp)
        words_matrix = []
        for text in corp:
            word_nums = dict.fromkeys(self._vocabulary, 0)
            for word in text.lower().split(' '):
                word_nums[word] += 1
            words_matrix.append(list(word_nums.values()))
        return words_matrix

    def get_feature_names(self):
        return self._vocabulary

