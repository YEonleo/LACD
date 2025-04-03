# data_utils.py

from datasets import load_dataset
from utils import create_demo_text, extract_supporting_facts

def load_data_and_create_prompts(args):
    """
    Loads the specified dataset and creates prompts (with and without context)
    along with the answers.
    Returns: (prompts_with_context, prompts_without_context, answers)
    """
    if args.dataset == 'hotpot_qa':
        dataset = load_dataset("hotpot_qa", "distractor")
        validation_dataset = dataset['validation']
        prompts_with_context, answers = create_prompts_from_hotpot(validation_dataset, include_context=True)
        prompts_without_context, _ = create_prompts_from_hotpot(validation_dataset, include_context=False)

    elif args.dataset == 'squad':
        dataset = load_dataset("rajpurkar/squad", "plain_text")
        validation_dataset = dataset['validation']
        prompts_with_context, answers = create_prompts_from_squad(validation_dataset, include_context=True)
        prompts_without_context, _ = create_prompts_from_squad(validation_dataset, include_context=False)

    else:
        raise ValueError("Invalid dataset name. Choose from 'hotpot_qa' or 'squad'.")

    return prompts_with_context, prompts_without_context, answers


def create_prompts_from_hotpot(dataset, include_context: bool):
    """
    Creates prompts and answers for HotpotQA.
    If include_context=True, attaches supporting facts to the prompts.
    """
    prompts = []
    answers = []

    for item in dataset:
        question = item['question']
        answer = item['answer']
        context = item.get('context', {})
        supporting_facts = item.get('supporting_facts', {})

        # Build prompt
        instruction = create_demo_text()
        if include_context:
            facts = extract_supporting_facts(context, supporting_facts)
            if facts:
                facts_str = ' '.join(facts)
                prompt = f"{instruction}Supporting information: {facts_str}\n\nQ: {question}\nA: "
            else:
                prompt = f"{instruction}\n\nQ: {question}\nA: "
        else:
            prompt = f"{instruction}\n\nQ: {question}\nA: "

        prompts.append(prompt)
        answers.append(answer)

    return prompts, answers


def create_prompts_from_squad(dataset, include_context: bool):
    """
    Creates prompts and answers for SQuAD.
    If include_context=True, attaches the paragraph context to the prompts.
    """
    prompts = []
    answers = []

    for item in dataset:
        question = item['question']
        # 'answers' is a list in SQuAD; take the first one
        answer = item['answers']['text'][0]
        context = item['context']

        instruction = create_demo_text()
        if include_context:
            prompt = f"{instruction}Supporting information: {context}\n\nQ: {question}\nA: "
        else:
            prompt = f"{instruction}\n\nQ: {question}\nA: "

        prompts.append(prompt)
        answers.append(answer)

    return prompts, answers
