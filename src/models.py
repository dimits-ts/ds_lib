from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


def _record_label(label_dict: dict, word: str, label: str) -> None:
    """
   Helper function to record the label count for a given word.
   :param label_dict: Dictionary to store label counts for each word.
   :param word: The word for which the label count is recorded.
   :param label: The label associated with the word.
   """
    if word in label_dict and label in label_dict[word]:
        label_dict[word][label] += 1
    else:
        label_dict[word] = {label: 1}


class BaselineLabelClassifier(ClassifierMixin, BaseEstimator):
    """
    A sklearn majority classifier which classifies a word with the majority label associated with it.
    Unknown words are assigned the globally most likely label.
    """
    def __init__(self):
        self.word_pos_dict = {}
        self.most_popular_pos_tag = None
        self.is_fitted_ = False

    def fit(self, x: list[str], y: list[str]):
        """
        Fit the BaselineLabelClassifier on the training data.
        :param x: List of words.
        :param y: List of corresponding labels.
        :return: The fitted classifier object.
        """
        all_labels_dict = {}
        inter_dict = {}

        for word, label in zip(x, y):
            _record_label(self.word_pos_dict, word, label)

            # find global maximum label
            if label in all_labels_dict:
                all_labels_dict[label] += 1
            else:
                all_labels_dict[label] = 1

        self.most_popular_pos_tag = max(all_labels_dict, key=all_labels_dict.get)

        for word in inter_dict.keys():
            self.word_pos_dict = max(word, key=word.get)

        self.is_fitted_ = True
        return self

    def predict(self, words: list[str]):
        """
        Predict the labels for a list of words using the fitted model.
        :param words: List of words to predict labels for.
        :return: List of predicted labels.
        """
        check_is_fitted(self)
        response = []

        for word in words:
            if word in self.word_pos_dict:
                response.append(next(iter(self.word_pos_dict[word])))
            else:
                response.append(self.most_popular_pos_tag)
        return response
