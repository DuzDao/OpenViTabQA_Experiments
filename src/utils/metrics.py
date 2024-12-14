import nltk
from nltk.translate.meteor_score import meteor_score
from collections import Counter
import string
import re

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


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


def meteor_score_metric(prediction, ground_truth):
    """Computes the METEOR score."""
    try:
      return meteor_score([normalize_answer(ground_truth)], normalize_answer(prediction))
    except:
      return 0.0


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
    meteor_scores = [meteor_score_metric(p, g) for p, g in zip(predictions, ground_truths)]

    return {
        "em": sum(em_scores) / len(em_scores) if em_scores else 0,
        "f1": sum(f1_scores) / len(f1_scores) if f1_scores else 0,
        "rouge1": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0,
        "meteor": sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0
    }

