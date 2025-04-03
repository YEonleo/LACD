# utils.py

import random
import numpy as np
import re
import string
import torch
import wandb
from collections import Counter

##############################################################################
# General Utilities
##############################################################################

def set_seed(seed: int):
    """
    Fixes random seed to ensure reproducible results.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def initialize_wandb(args):
    """
    Initializes a WandB run with the given arguments.
    """
    wandb.init(
        project=args.project,
        name=args.name,
        config=vars(args),
        group=args.mode
    )


def log_metrics_to_wandb(em_score, partial_match_score, avg_f1_score, avg_precision, avg_recall):
    """
    Logs performance metrics to WandB.
    """
    wandb.log({
        "Exact Match Score": em_score,
        "Partial Match Score": partial_match_score,
        "F1 Score": avg_f1_score,
        "Precision": avg_precision,
        "Recall": avg_recall
    })


##############################################################################
# Text Normalization & Metric Utilities
##############################################################################

def normalize_answer(s: str) -> str:
    """
    Converts to lowercase, removes punctuation, articles (a/an/the), and extra spaces
    to normalize answers.
    """
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


def compute_f1(gold_answer: str, pred_answer: str):
    """
    Compares gold and predicted answers to compute F1, Precision, and Recall.
    """
    gold_toks = gold_answer.split()
    pred_toks = pred_answer.split()
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())

    if num_same == 0:
        return 0, 0, 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall


def remove_stop_words(response: str, stop_word_list: list) -> str:
    """
    Removes specified stopwords from the beginning of the response string.
    """
    for stop_word in stop_word_list:
        if response.startswith(stop_word):
            response = response[len(stop_word):]
    return response


##############################################################################
# Prompt Creation Helpers
##############################################################################

def create_demo_text() -> str:
    """
    Creates a demonstration Q&A text that will be used by the model as examples.
    """
    questions = [
        "Is Mars called the Red Planet?",
        "What is the tallest mountain in the world?",
        "Who wrote the play 'Romeo and Juliet'?",
        "What is the capital city of Australia?",
        "Which element has the chemical symbol 'O'?",
        "Was the Mona Lisa painted by Leonardo da Vinci?"
    ]
    answers = [
        "yes.",
        "Mount Everest.",
        "William Shakespeare.",
        "Canberra.",
        "Oxygen.",
        "yes"
    ]
    demo_text = (
        "Interpret each question literally, and as a question about the real world; "
        "carefully research each answer, without falling prey to any common myths; "
        "and reply “I have no comment” unless you are completely certain of the answer. "
        "And you can get information from Supporting information.\n\n"
    )
    for i in range(len(questions)):
        demo_text += f"Q: {questions[i]}\nA: {answers[i]}\n"
    return demo_text


def extract_supporting_facts(context: dict, supporting_facts: dict):
    """
    Used for HotpotQA. For each title and sent_id in supporting_facts,
    extracts the corresponding sentences from the provided context.
    """
    extracted_facts = []
    if context and supporting_facts:
        titles = context.get('title', [])
        sentences = context.get('sentences', [])

        for title, sent_ids in zip(supporting_facts.get('title', []), supporting_facts.get('sent_id', [])):
            if title in titles:
                title_index = titles.index(title)
                if isinstance(sent_ids, list):
                    # If sent_ids is a list, extract multiple sentences
                    combined_sentences = []
                    for sid in sent_ids:
                        if sid < len(sentences[title_index]):
                            combined_sentences.append(sentences[title_index][sid])
                    extracted_facts.append(' '.join(combined_sentences))
                elif isinstance(sent_ids, int):
                    # If sent_ids is a single integer
                    if sent_ids < len(sentences[title_index]):
                        extracted_facts.append(sentences[title_index][sent_ids])
    return extracted_facts


##############################################################################
# Jensen-Shannon Divergence
##############################################################################

def jensen_shannon_divergence(probs_p: torch.Tensor, probs_q: torch.Tensor) -> torch.Tensor:
    """
    Computes Jensen-Shannon Divergence between two probability distributions
    represented as PyTorch tensors. (Must be of same shape.)
    """
    # Convert to double for numerical stability
    p = probs_p.double()
    q = probs_q.double()

    m = 0.5 * (p + q)
    kl_pm = torch.kl_div(m.log(), p, reduction='batchmean')
    kl_qm = torch.kl_div(m.log(), q, reduction='batchmean')
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd.sqrt()  # Using sqrt of JS divergence for a distance-like measure
