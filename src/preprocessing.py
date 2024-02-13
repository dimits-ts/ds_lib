import itertools
from typing import Callable

import numpy as np
import pandas as pd
from more_itertools import windowed
from sklearn.preprocessing import LabelBinarizer
from tqdm.auto import tqdm


def preprocess(df: pd.DataFrame, window_size: int, max_window_count: int, embed_func: Callable[[str], np.ndarray],
               binarizer: LabelBinarizer, pad_token: str, train_binarizer: bool = False) \
        -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess the data for training a neural network by generating context-aware window embeddings
    and corresponding binary labels for POS tags.

    :param df: The DataFrame containing the data with columns 'words', 'pos', and 'sent_id'.
    :param window_size: Size of the context window.
    :param max_window_count: Maximum number of windows to consider.
    :param embed_func: A function that generates the embedding for a single word.
    :param binarizer: The LabelBinarizer used to transform POS tags into binary labels.
    :param train_binarizer: Whether to fit the binarizer on the target data.
    :param pad_token: The string with which the padding tokens will be denoted.
    :return: Tuple containing the generated embeddings (X) and binary labels (y).
    """

    print("Calculating windows...")
    windows = windowed_sentence(df, window_size, pad_token)

    print("Calculating targets...")
    targets = windowed_tags(df, window_size, pad_token)

    print("Computing window embeddings...")
    window_lim = min(max_window_count, len(windows))

    embeddings = compute_embeddings(embed_func, windows[:window_lim])
    embeddings = np.array(embeddings)

    if train_binarizer:
        target_vec = binarizer.fit_transform(targets[:window_lim])
    else:
        target_vec = binarizer.transform(targets[:window_lim])

    return embeddings, target_vec


def windowed_sentence(df, window_size: int, pad_token: str) -> list[tuple[str, ...]]:
    """
    Generate windowed sentences from a DataFrame.
    :param df: The DataFrame containing the data with columns 'words', 'pos', and 'sent_id'.
    :param window_size: Size of the context window.
    :param pad_token: The padding token used in window creation.
    :return: List of windowed sentences.
    """
    windows = []

    for sentence_id in tqdm(set(df.sent_id)):
        words_df = df[df.sent_id == sentence_id]
        sentence = list(words_df.words)
        padding = [pad_token] * (window_size // 2)

        for window in windowed(itertools.chain(padding, sentence, padding), window_size, fillvalue=pad_token):
            windows.append(window)

    return windows


def windowed_tags(df, window_size: int, pad_token: str) -> list[str]:
    """
    Generate windowed POS tags from a DataFrame.
    :param df: The DataFrame containing the data with columns 'words', 'pos', and 'sent_id'.
    :param window_size: Size of the context window.
    :param pad_token: The padding token used in window creation.
    :return: List of windowed POS tags.
    """
    targets = []

    for sentence_id in tqdm(set(df.sent_id)):
        words_df = df[df.sent_id == sentence_id]
        tags = list(words_df.pos)
        padding = [pad_token] * (window_size // 2)

        for window in windowed(itertools.chain(padding, tags, padding), window_size, fillvalue=pad_token):
            targets.append(window[window_size // 2])

    return targets


def compute_embeddings(embed_func: Callable[[str], np.ndarray], windows: list[tuple[str, ...]]) -> list[np.ndarray]:
    """
    Compute context-aware window embeddings for a list of windowed strings.
    :param embed_func: A function that generates the embedding for a single word.
    :param windows: a list of windows, each being a list of strings

    :return: A list of window embeddings.
    :rtype: list[str]
    """
    embeddings = []

    for window in tqdm(windows):
        word_embeddings = []

        for word in window:
            word_embeddings.append(embed_func(word))

        embeddings.append(np.hstack(word_embeddings))

    return embeddings
