from collections import Counter
import string
import re
import unicodedata
from underthesea import word_tokenize
from nltk.translate.meteor_score import meteor_score

def normalize_answer(s):
    """
    Normalizes a Vietnamese answer by:
    - Lowercasing the text.
    - Removing punctuation.
    - Removing extra whitespace.
    - Removing diacritics (accents).
    """
    def lower(text):
        return text.lower()

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_diacritics(text):
        """
        Removes diacritics (accents) from Vietnamese text.
        """
        normalized = unicodedata.normalize('NFKD', text)
        return re.sub(r'[\u0300-\u036f]', '', normalized)
    
    return white_space_fix(remove_diacritics(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    """Computes the exact match score."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    """Computes the F1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def rouge_1_score(prediction, ground_truth):
    """Computes the ROUGE-1 score."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    n = 1
    
    if len(prediction_tokens) < n or len(ground_truth_tokens) < n:
        return 0.0
        
    prediction_ngrams = [tuple(prediction_tokens[i:i+n]) for i in range(len(prediction_tokens) - n + 1)]
    ground_truth_ngrams = [tuple(ground_truth_tokens[i:i+n]) for i in range(len(ground_truth_tokens) - n + 1)]
    
    common_ngrams = Counter(prediction_ngrams) & Counter(ground_truth_ngrams)
    num_same = sum(common_ngrams.values())
    
    if num_same == 0:
      return 0.0
    
    precision = 1.0 * num_same / len(prediction_ngrams)
    recall = 1.0 * num_same / len(ground_truth_ngrams)
    
    if precision + recall == 0:
        return 0.0
    
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def meteor_score_metric(references, hypothesis):
    """
    Calculates the METEOR score between a reference (or list of references) and a hypothesis.

    Args:
        references: The correct answer(s) (string, list of strings, or list of lists of strings).
        hypothesis: The model's prediction (string or list of strings).

    Returns:
        The METEOR score (float between 0 and 1).
    """

    if isinstance(hypothesis, str):
        hypothesis = [hypothesis]

    if isinstance(references, str):
        references = [references]

    total_meteor_score = 0
    for ref, hypo in zip(references, hypothesis):
        tokenized_ref = word_tokenize(ref) if isinstance(ref, str) else ref
        tokenized_hypo = word_tokenize(hypo) if isinstance(hypo, str) else hypo
        total_meteor_score += meteor_score([tokenized_ref], tokenized_hypo)


    return total_meteor_score / len(hypothesis)



def compute_metrics(predictions, ground_truths):
    """
    Computes all evaluation metrics.

    Args:
        predictions (list): List of predicted answers.
        ground_truths (list): List of ground truth answers.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    em_scores = [exact_match_score(p, g) for p, g in zip(predictions, ground_truths)]
    f1_scores = [f1_score(p, g) for p, g in zip(predictions, ground_truths)]
    rouge1_scores = [rouge_1_score(p, g) for p, g in zip(predictions, ground_truths)]
    meteor_scores = [meteor_score_metric([normalize_answer(g)], normalize_answer(p)) for p, g in zip(predictions, ground_truths)] # Call my_meteor_score

    return {
        "em": sum(em_scores) / len(em_scores) if em_scores else 0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
        "rouge1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0,
        "meteor": sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    }
