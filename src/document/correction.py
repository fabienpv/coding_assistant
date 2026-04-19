import re
import glob
import pandas as pd
import Levenshtein
import json
import time
from typing import Union

from src.paths import SPELLING



def load_df_spellcheck():
    """Loads the dataframe used to make the spelling correction.
    Creates a columns counting the length of the entries.
    Sorts by length (from shorter to longer) and by number of use
    (from highest to lowest), then return df."""
    df = pd.read_csv(str(SPELLING) + "/linux_dict_cleaned.csv")
    df["length"] = df["word"].str.len()
    df = df.sort_values(by=["length", "count"], ascending=[True, False]).reset_index(drop=True)
    return df


df_spellcheck = load_df_spellcheck()

MAX_LEN_SPELLCHECK = df_spellcheck["word"].str.len().max()
SET_SPELLCHECK_WORDS = set(df_spellcheck["word"])


def redundant_characters_md_format(text: str) -> str:
    """Removes redundant ~ and * found in md formatting."""
    _text_ = text
    pat1 = r"(?:(?<=~{2})|(?<=~{2}.))\*+"
    pat2 = r"\*{2}(?=.?~{2})"
    pattern = rf"({pat1}|{pat2})"
    _text_ = re.sub(pattern, "", _text_)
    _text_ = re.sub(r"~", "", _text_)
    _text_ = re.sub(r"\*\*", "", _text_)
    _text_ = re.sub(r"-{4,}", "---", _text_)
    return _text_


def remove_extra_space(text: str) -> str:
    """Remove extra spaces in a string.
    
        :param str text: The input string.
        :return: The string with extra spaces removed.
        :rtype: str"""
    text = re.sub(r"(?<=\d)\s+,\s+(?=\d)", ",", text)
    text = re.sub(r"(?<=\d)\s+\.\s+(?=\d)", ",", text)
    text = re.sub(r" +", " ", text)
    return text


def remove_redundant_linebreak(text: str) -> str:
    """Remove redundant linebreaks from a string.
    
    :param str text: The input string.
    :return: The string with redundant linebreaks removed.
    :rtype: str"""
    _text_ = re.sub(r"\n{3,}", "\n\n", text)
    return _text_


def remove_ocr_repetition(text: str, threshold: int = 100) -> str:
    """Correction for bugs when a OCR model, especially VLM,
    repeats the same tokens dozens of times.
    The text as input should be the text for a single page."""
    lines = text.split("\n\n")
    if len(lines) > threshold:  # suspicious above this threshold
        series_lines = pd.Series(lines).astype(str)
        vc = series_lines.value_counts()
        reps = list((vc[vc > 5]).index)
        corrected_lines = [l_ for l_ in lines if l_ not in reps]
        text = "\n\n".join(corrected_lines)
    return text


class Abbreviations:
    def __init__(self):
        """Initializes the class.
        
            Loads abbreviations, initializes edit timer to None, and preview
            dictionary to None."""
        self.__abbreviations = Abbreviations.load()

    @staticmethod
    def load():
        """Loads the dataframe used to make the spelling correction.
        
            Creates columns counting the length of the entries.
            Sorts by length (from shorter to longer) and by number of use
            (from highest to lowest), then return df."""
        with open(f"{SPELLING}/abbreviations.json", "r") as f_:
            abbreviations = json.load(f_)
        return abbreviations

    def explain(self, text: str) -> str:
        """Explains abbreviations in the input text.
        
            :param str text: The text to explain.
            :return: The text with abbreviations explained.
            :rtype: str"""
        abbreviations = self.__abbreviations
        for k, v in abbreviations.items():
            text = re.sub(rf"{v['pattern']}(?!.{'{0,3}'}{k})", f"{v['description']} ({k})", text, flags=re.DOTALL)
            text = re.sub(rf"(?:{v['description']}\s\(|(?<=^)|(?<=\s)|(?<=\W)){k}(?:\))?(?=\W|$|\s)", f"{v['description']} ({k})", text, flags=re.DOTALL)
        return text
    
    def get_list(self) -> list[str]:
        """Return a list of the abbreviations.
        
        :return: A list of strings representing the abbreviations.
        :rtype: list[str]"""
        return list(self.__abbreviations.keys())
    
    def show(self) -> str:
        """Display the abbreviations and their descriptions.
        
        :return: A string representation of the abbreviations."""
        text = []
        for k, v in self.__abbreviations.items():
            line = f"{k}: {v['description']}"
            text.append(line)
        return "\n".join(text)


abbreviations = None

def get_abbreviations():
    """Return the global `Abbreviations` instance.
    
        If the instance doesn't exist, it is created first."""
    global abbreviations
    if abbreviations is None:
        abbreviations = Abbreviations()
    return abbreviations


def words_correction(text: str, use_hamming: bool = False) -> str:
    """The correction relies on the hamming distance,
    correcting words only with the best found match in the
    dictionary that has the same length than the initial word.
    
    :param text: the text that should be corrected
    :param use_hamming: boolean. If true, uses the hamming distance.
        Hamming only corrects with a word of the same length, thus not
        accounting for deletion or insertion. If False, uses 'levenshtein',
        that can makes correction with up to 1 deletion or insertion, with
        a stronger penalty for insertion and deletion as compared to
        substitution. Default to Levenshtein.
    :return: the corrected text
    :rtype: str"""
    # split text into words and create a pandas series with the words to apply the corrections
    text_words = pd.Series(re.split(r"((?<=\b)|(?<=\n)[a-z]+(?:\b))", text))
    # exclude words with capital letter (they could be acronyms or proper nouns that are not in the spelling dictionnary)
    # exclude short words (the risk to correct them wrong is too high)
    # exclude long words (could be a code or technical words, that should not or cannot be corrected)
    # exclude words found in the dictionary (they do not need to be corrected)
    mask = (text_words.str.contains(r"\d|[A-Z]{1}|\W", regex=True)) | (text_words.str.len() < 5) | (text_words.str.len() > MAX_LEN_SPELLCHECK) | text_words.isin(SET_SPELLCHECK_WORDS)
    dict_correct = {} # dictionnary of words to correct. key: wrong spelling found in text, value: correction
    for word in text_words.loc[~mask].unique():
        lower_word = word.lower()
        if use_hamming: 
            candidates = df_spellcheck.query(f"length == {len(lower_word)}")["word"]
        else: 
            candidates = df_spellcheck.query(f"length in [{len(lower_word)-1}, {len(lower_word)}, {len(lower_word)+1}]")["word"]
        if len(candidates) > 0:
            if use_hamming:
                distance = candidates.apply(lambda x: Levenshtein.hamming(lower_word, x))
            else:
                distance = candidates.apply(lambda x: Levenshtein.distance(lower_word, x, weights=(2,2,1), score_cutoff=5))
            idx_min = int(distance.idxmin())
            if distance[idx_min] == 1:
                dict_correct[word] = candidates.loc[idx_min]
    # apply the corrections
    mask = text_words.isin(dict_correct.keys())
    text_words.loc[mask] = text_words.loc[mask].apply(lambda x: dict_correct[x])
    # transform back into a text and return result
    return "".join(text_words)


def text_correction_light_pipeline(text: str):
    """Apply a light text correction pipeline.
    
      This pipeline removes redundant characters, extra spaces, redundant
      linebreaks, and explains abbreviations.
    
      :param str text: The input text.
      :return: The corrected text.
      :rtype: str"""
    _text_ = text
    abbreviations = get_abbreviations()
    _text_ = redundant_characters_md_format(_text_)
    _text_ = remove_extra_space(_text_)
    _text_ = remove_redundant_linebreak(_text_)
    _text_ = abbreviations.explain(_text_)
    return _text_
