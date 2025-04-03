# main.py

import argparse
import wandb

from utils import (
    set_seed,
    initialize_wandb,
    log_metrics_to_wandb
)
from model import Base_Model
from data_utils import load_data_and_create_prompts
from evaluation import evaluate_model, save_incorrect_details


def main(args):
    # 1. Set random seed
    set_seed(42)
    print(f"Mode: {args.mode}")

    # 2. Initialize model
    llm = Base_Model(
        model_name=args.model_name,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory
    )

    # 3. Define stop words
    stop_word_list = ["Q:", "Supporting information:", "\n", "\n\n##"]
    llm.set_stop_words(stop_word_list)

    # 4. Load dataset & create prompts
    prompts_with_context, prompts_without_context, answers = load_data_and_create_prompts(args)

    # 5. Evaluate the model
    try:
        em_score, partial_match, avg_f1, avg_precision, avg_recall, incorrect_details = evaluate_model(
            llm,
            prompts_with_context,
            prompts_without_context,
            answers,
            stop_word_list,
            args
        )

        # 6. Save incorrect predictions
        save_incorrect_details(incorrect_details, args.name)

        # 7. Initialize WandB
        initialize_wandb(args)

        # 8. Log metrics to WandB
        log_metrics_to_wandb(em_score, partial_match, avg_f1, avg_precision, avg_recall)

        # 9. Finish WandB run
        wandb.finish()

    except Exception as e:
        print(f"An error occurred: {e}")
        exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # ------------------------ Model-related arguments
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b",
                        help="Name of the model to use.")
    parser.add_argument("--num-gpus", type=str, default="1",
                        help="Number of GPUs to use.")
    parser.add_argument("--max_gpu_memory", type=int, default=24,
                        help="Maximum GPU memory to allocate.")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda",
                        help="Device to run the model on.")

    # ------------------------ Data-related arguments
    parser.add_argument("--dataset", type=str, choices=['hotpot_qa', 'squad'],
                        required=True, help="Dataset to use for training and evaluation.")

    # ------------------------ Decoding/Inference-related arguments
    parser.add_argument("--mode", type=str, default='final_layer_context',
                        help="Generation mode (e.g., CAD, DOLA, final_layer_context, etc.).")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Alpha value for logits adjustment.")
    parser.add_argument("--layer_alpha", type=float, default=0.1,
                        help="Layer alpha for additional adjustments.")
    parser.add_argument("--start_layer", type=int, default=17,
                        help="Starting layer for comparisons.")

    # Example of a subset_layers argument if needed
    # parser.add_argument("--subset-layers", type=int, nargs='+', default=None,
    #                     help="Optional list of layers for contrast_layer_context_nocontext_jsd_subset")

    # ------------------------ Logging-related arguments
    parser.add_argument("--name", type=str, default='default',
                        help="Name of the experiment.")
    parser.add_argument("--project", type=str, default='Decoding_Exp_val2',
                        help="WandB project name.")
    parser.add_argument("--group", type=str, help="Group name for the experiment")

    args = parser.parse_args()
    main(args)
