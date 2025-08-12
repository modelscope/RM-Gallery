import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import numpy as np
from datasets import concatenate_datasets
from fastchat.conversation import get_conv_template
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from rewardbench import load_eval_dataset_multi, process_single_model, save_to_hub
from rewardbench.generative_v2 import (
    ANTHROPIC_MODEL_LIST,
    API_MODEL_LIST,
    GEMINI_MODEL_LIST,
    OPENAI_MODEL_LIST,
    format_judge_answers,
    get_single_rating,
    process_judgement,
    run_judge_four,
    run_judge_ratings_multi,
)

# get token from HF_TOKEN env variable, but if it doesn't exist pass none
HF_TOKEN = os.getenv("HF_TOKEN", None)
# this is necessary to automatically log in when running this script in docker/batch beaker jobs
if HF_TOKEN is not None:
    from huggingface_hub._login import _login

    _login(token=HF_TOKEN, add_to_git_credential=False)


def get_args():
    """
    Parse arguments strings model and chat_template
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        nargs="+",  # allow list of models (ensemble)
        required=True,
        help="name of OpenAI model to use (TODO add more providers/models)",
    )
    parser.add_argument("--dataset", type=str, default="allenai/reward-bench-2", help="path to huggingface dataset")
    parser.add_argument("--chat_template", type=str, default=None, help="fastchat chat template (optional)")
    parser.add_argument(
        "--trust_remote_code", action="store_true", default=False, help="directly load model instead of pipeline"
    )
    parser.add_argument(
        "--score_w_ratings", action="store_true", default=False, help="score with ratings instead of pairwise ranking"
    )
    parser.add_argument("--num_gpus", type=int, default=1, help="number of gpus to use, for multi-node vllm")
    parser.add_argument("--vllm_gpu_util", type=float, default=0.9, help="gpu utilization for vllm")
    # parser.add_argument("--vllm_max_seq_length", type=int, default=None, help="max sequence length for vllm")
    parser.add_argument("--do_not_save", action="store_true", help="do not save results to hub (for debugging)")
    parser.add_argument(
        "--pref_sets", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--debug", action="store_true", help="run on common preference sets instead of our custom eval set"
    )
    parser.add_argument(
        "--num_threads", type=int, default=10, help="number of threads to use for parallel processing of examples"
    )
    parser.add_argument(
        "--disable_beaker_save", action="store_true", help="disable saving the main results in a file for AI2 Beaker"
    )
    parser.add_argument(
        "--force_local", action="store_true", default=False, help="force local run, even if model is on Together API"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    ###############
    # Setup logging
    ###############
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = logging.INFO
    logger.setLevel(log_level)

    logger.info(f"Running reward model on {args.model} with chat template {args.chat_template}")

    model_type = "Generative RM"

    # if model is list, make type + PoLL and check multiple is odd
    if isinstance(args.model, list) and len(args.model) == 1:
        args.model = args.model[0]
    elif isinstance(args.model, list):
        model_type += " PoLL"
        # assert that is odd and > 1
        assert len(args.model) % 2 == 1

    # define variable if is API or local
    if args.force_local:
        is_api_models = False
    else:
        is_api_models = isinstance(args.model, list) or args.model in API_MODEL_LIST

    # if model isn't API, load via vllm
    if not is_api_models:
        # if multi gpu, set multiproc method to spawn
        if args.num_gpus > 1:
            # Set the environment variable
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # load model
        model = LLM(
            args.model,
            trust_remote_code=args.trust_remote_code,
            tensor_parallel_size=args.num_gpus,
            gpu_memory_utilization=args.vllm_gpu_util,
            # max_seq_length=args.vllm_max_seq_length,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if "Llama-3" in args.model or "llama3-8b" in args.model and "3.1" not in args.model:
            stop_token_ids = [128009]
        else:
            stop_token_ids = None

        sampling_params = SamplingParams(
            n=1,
            temperature=0,
            top_p=1,
            max_tokens=2048,
            stop_token_ids=stop_token_ids,
        )

    # handle off-case models
    # use different prompt for prometheus/gemini models
    if "prometheus" in args.model:
        model_modifier = "prometheus"
    elif "Con-J" in args.model:
        model_modifier = "Con-J"
    elif "OffsetBias" in args.model:
        model_modifier = "offsetbias"
    elif "Atla" in args.model:
        logger.info("Using ATLA model")
        model_modifier = "Atla"
    elif "gemini" in args.model:
        model_modifier = "gemini"
    elif "RISE-Judge" in args.model:
        model_modifier = "RISE-Judge"
    else:
        model_modifier = None

    ############################
    # Load dataset
    ############################
    logger.info("*** Load dataset ***")
    # to handle the Ties subset, we keep the "subset" and "num_correct" columns for RB2.
    dataset = load_eval_dataset_multi(
        core_set=not args.pref_sets,
        dataset=args.dataset,
        conv=get_conv_template("raw"),  # not used in this script (handled later)
        custom_dialogue_formatting=True,  # handle formatting later
        tokenizer=None,
        logger=logger,
        keep_columns=["texts_chosen", "texts_rejected", "id", "subset", "num_correct"],
        max_turns=4,
    )

    # copy id for saving, then remove
    # save ties_ids separately because they are needed for processing ties results
    ties_ids = dataset.filter(lambda example: example["subset"] == "Ties")["id"]

    # separate dataset into dataset for non-ties and ties_dataset for ties based on "subset" == "Ties"
    ties_dataset = dataset.filter(lambda example: example["subset"] == "Ties")
    dataset = dataset.filter(lambda example: example["subset"] != "Ties")
    nonties_ids = dataset["id"]
    dataset = dataset.remove_columns("id")

    # debug: use only 20 examples, 10 from dataset and 10 from ties_dataset
    if args.debug:
        dataset = dataset.select(range(10))
        ties_dataset = ties_dataset.select(range(10))
        ties_ids = ties_ids[:10]  # add ties ids to ties_ids
        nonties_ids = nonties_ids[:10]  # add ties ids to ids

    # output_path = f"final_results_{args.model}.jsonl"
    # if os.path.exists(output_path):
    #     os.remove(output_path)
    # logger.info(f"**Outputting scores to {output_path}**")

    if is_api_models:
        ############################
        # Run inference via API
        ############################
        def update_progress_bar(done, total):
            # Simple text-based progress bar
            progress = int(50 * done / total)  # Calculate progress (50 chars width)
            sys.stdout.write("\r[{}{}] {}/{}".format("#" * progress, "." * (50 - progress), done, total))
            sys.stdout.flush()

        def get_judgement(batch, is_ties=False, debug=args.debug):
            mult_turn = True if len(batch["texts_chosen"][0]) > 2 else False
            prompt = batch["texts_chosen"][0][0]["content"]

            # ties dataset must be scored with absolute ratings
            if not args.score_w_ratings and not is_ties:
                # only look at 4 options in direct judgment case
                answer_a = batch["texts_chosen"][0]
                answer_b = batch["texts_rejected"][0]
                answer_c = batch["texts_rejected"][1]
                answer_d = batch["texts_rejected"][2]

                shuffle_option = np.random.randint(0, 4)

                if shuffle_option == 0:
                    # Original order
                    winner_text = "A"
                    loser_texts = ["B", "C", "D"]  # or any other
                elif shuffle_option == 1:
                    # swap A and B
                    answer_a, answer_b = answer_b, answer_a
                    winner_text = "B"
                    loser_texts = ["A", "C", "D"]
                elif shuffle_option == 2:
                    # swap A and C
                    answer_a, answer_c = answer_c, answer_a
                    winner_text = "C"
                    loser_texts = ["A", "B", "D"]
                elif shuffle_option == 3:
                    # swap A and D
                    answer_a, answer_d = answer_d, answer_a
                    winner_text = "D"
                    loser_texts = ["A", "B", "C"]

                if len(batch["texts_chosen"][0]) <= 4:  # set up only for 1 or 2 turns
                    winner, request, judgement = run_judge_four(
                        prompt,
                        answer_a,
                        answer_b,
                        answer_c,
                        answer_d,
                        args.model,
                        multi_turn=mult_turn,
                        model_modifier=model_modifier,
                    )
                    if debug:
                        print(f"Prompt: {request}")
                        print(f"Judgement: {judgement}")

                    # handle voting
                    if isinstance(winner, list):
                        # print votes if debug
                        if debug:
                            print(winner)
                        winner = max(set(winner), key=winner.count)

                    if winner == winner_text:
                        return 1
                    elif winner in loser_texts:
                        return 0
                    else:  # if "error"
                        return 0.25  # effectively a tie
                else:
                    print("Error: more than 4 turns")
                    return 0.25

            # scoring with ratings
            else:
                # no shuffling needed for absolute rating
                batch["texts_chosen"].extend(batch["texts_rejected"])
                answers = batch["texts_chosen"]
                winners, requests, judgements = run_judge_ratings_multi(
                    prompt, answers, args.model, multi_turn=mult_turn, model_modifier=model_modifier, is_ties=is_ties
                )

                if debug:
                    print(f"Prompt: {requests}")
                    print(f"Judgement: {judgements}")
                    if winners != "error":
                        print(f"Score: {(0 in winners)/len(winners)}")

                # for ties subset, return the set of scores for aggregate results to be computed later
                if is_ties:
                    return judgements["ratings"]

                # for non ties data, return
                if winners == "error":
                    # effectively a tie
                    return 0.25

                # handle ties, first response (index 0) is the correct one
                # 1 if the first response is the winner, 0.5 if joint (2-way) winner, 0.33 if 3-way, etc.
                return (0 in winners) / len(winners)

        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            # Map 'my_function' across the vector, executing in parallel using threads

            # First run on non-Ties subsets
            logger.info("*** Run inference on non-ties subsets ***")
            # Progress bar version
            results = [None] * len(dataset)  # Preallocate results list
            done_tasks = 0  # Counter for completed tasks
            judge_fn = partial(get_judgement, is_ties=False)
            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                # Submit all tasks and hold their futures in a list
                future_to_index = {executor.submit(judge_fn, x): i for i, x in enumerate(dataset)}

                # As tasks complete, update progress and store results in the original order
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    results[index] = future.result()
                    done_tasks += 1
                    update_progress_bar(done_tasks, len(dataset))

            # Run on Ties subset
            logger.info("*** Run inference on Ties subset ***")
            results_ties = [None] * len(ties_dataset)  # Preallocate results list
            done_tasks = 0  # Counter for completed tasks
            judge_fn_ties = partial(get_judgement, is_ties=True)
            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                # Submit all tasks and hold their futures in a list
                future_to_index = {executor.submit(judge_fn_ties, x): i for i, x in enumerate(ties_dataset)}

                # As tasks complete, update progress and store results in the original order
                for future in as_completed(future_to_index):
                    index = future_to_index[future]
                    print(future.result())
                    results_ties[index] = future.result()
                    done_tasks += 1
                    update_progress_bar(done_tasks, len(ties_dataset))

            # Print newline after progress bar
            print()
    else:
        ############################
        # Run model weights with vllm
        ############################

        # Prepare vllm_model dict for ratings functions
        # At the top of the VLLM section:
        if args.chat_template is not None:
            chat_template = get_conv_template(args.chat_template)
        else:
            chat_template = None

        vllm_model_dict = {
            "model": model,
            "tokenizer": tokenizer,
            "sampling_params": sampling_params,
            "chat_template": chat_template,  # Add this
        }

        def format_judgements(batch, optional_chat_template=None):
            # TODO expand this to include fastchat chat templates if needed
            mult_turn = True if len(batch["texts_chosen"]) > 2 else False
            prompt = batch["texts_chosen"][0][0]["content"]
            answer_a = batch["texts_chosen"][0]
            answer_b = batch["texts_rejected"][0]
            answer_c = batch["texts_rejected"][1]
            answer_d = batch["texts_rejected"][2]

            shuffle_option = np.random.randint(0, 4)

            # shuffle correct answer into random position, option 0 is original order
            if shuffle_option == 1:
                # swap A and B
                answer_a, answer_b = answer_b, answer_a
            elif shuffle_option == 2:
                # swap A and C
                answer_a, answer_c = answer_c, answer_a
            elif shuffle_option == 3:
                # swap A and D
                answer_a, answer_d = answer_d, answer_a

            system_prompt, user_prompt = format_judge_answers(
                prompt, answer_a, answer_b, answer_c, answer_d, multi_turn=mult_turn, model_modifier=model_modifier
            )

            if optional_chat_template is not None:
                optional_chat_template.set_system_message(system_prompt)
                optional_chat_template.messages = []
                optional_chat_template.append_message(optional_chat_template.roles[0], user_prompt)
                optional_chat_template.append_message(optional_chat_template.roles[1], None)
                prompt = optional_chat_template.get_prompt()
            else:
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": user_prompt},
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                # chat template already include special tokens
                # when vllm runs model.generate on prompts, the tokenizer is applied to the prompts
                # defaulting to add_special_tokens=True - this will end up duplicating the special tokens
                # so we need to tokenize without adding special tokens
                tokenized_prompt = tokenizer(prompt, add_special_tokens=False, return_length=True)
                prompt_ids = tokenized_prompt["input_ids"]
            batch["text"] = prompt
            batch["shuffle_position"] = shuffle_option
            batch["prompt_ids"] = prompt_ids
            return batch

        def format_ratings(batch, is_ties=False):
            """Format batch for ratings-based evaluation"""
            mult_turn = True if len(batch["texts_chosen"][0]) > 2 else False
            prompt = batch["texts_chosen"][0][0]["content"]  # Get the user question

            # Combine chosen and rejected answers
            # texts_chosen is [[messages]], texts_rejected is [messages, messages, messages]
            all_answers = batch["texts_chosen"] + batch["texts_rejected"]  # Remove the extra [0] indexing

            # Format each answer for rating
            formatted_answers = []
            for answer in all_answers:
                answer_text = answer[1]["content"]  # Get the assistant's response
                formatted_answers.append(answer_text)

            batch["prompt"] = prompt
            batch["answers"] = formatted_answers
            batch["mult_turn"] = mult_turn
            return batch

        def get_vllm_judgement(batch, is_ties=False):
            """Get judgement using VLLM model"""
            if not args.score_w_ratings and not is_ties:
                # Direct 4-way comparison (existing logic)
                mult_turn = batch["mult_turn"] if "mult_turn" in batch else (len(batch["texts_chosen"]) > 2)
                prompt = batch["texts_chosen"][0][0]["content"]

                # Get 4 answers and shuffle
                answer_a = batch["texts_chosen"]
                answer_b = batch["texts_rejected"][0]
                answer_c = batch["texts_rejected"][1]
                answer_d = batch["texts_rejected"][2]

                shuffle_option = np.random.randint(0, 4)
                if shuffle_option == 0:
                    winner_text = "A"
                    loser_texts = ["B", "C", "D"]
                elif shuffle_option == 1:
                    answer_a, answer_b = answer_b, answer_a
                    winner_text = "B"
                    loser_texts = ["A", "C", "D"]
                elif shuffle_option == 2:
                    answer_a, answer_c = answer_c, answer_a
                    winner_text = "C"
                    loser_texts = ["A", "B", "D"]
                elif shuffle_option == 3:
                    answer_a, answer_d = answer_d, answer_a
                    winner_text = "D"
                    loser_texts = ["A", "B", "C"]

                # Format prompt for 4-way comparison
                system_prompt, user_prompt = format_judge_answers(
                    prompt, answer_a, answer_b, answer_c, answer_d, multi_turn=mult_turn, model_modifier=model_modifier
                )

                # Generate with VLLM
                messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                outputs = model.generate([formatted_prompt], sampling_params=sampling_params)
                judgement = outputs[0].outputs[0].text.strip()

                winner = process_judgement(judgement, model_modifier)

                if winner == winner_text:
                    return 1
                elif winner in loser_texts:
                    return 0
                else:
                    return 0.25

            else:
                # Ratings-based evaluation
                prompt = batch["prompt"] if "prompt" in batch else batch["texts_chosen"][0][0]["content"]
                mult_turn = batch["mult_turn"] if "mult_turn" in batch else (len(batch["texts_chosen"]) > 2)

                # Prepare all answers
                if "answers" in batch:
                    all_answers_text = batch["answers"]
                else:
                    all_answers = [batch["texts_chosen"]] + batch["texts_rejected"]
                    all_answers_text = [ans[1]["content"] for ans in all_answers]

                # Get ratings for each answer
                ratings = []
                for answer_text in all_answers_text:
                    rating, _ = get_single_rating(
                        question_text=prompt,
                        answer_text=answer_text,
                        model=args.model,
                        model_modifier=model_modifier,
                        is_ties=is_ties,
                        vllm_model=vllm_model_dict,
                    )
                    ratings.append(rating)

                if is_ties:
                    return ratings

                # Find winners (non-ties case)
                valid_ratings = [r for r in ratings if r != -1]
                if not valid_ratings:
                    return 0.25

                max_rating = max(valid_ratings)
                winners = [i for i, r in enumerate(ratings) if r == max_rating]

                if args.debug:
                    logger.info(f"Raw judgment: {valid_ratings}")

                # Return score based on whether first answer (chosen) is among winners
                return (0 in winners) / len(winners)

        # Choose processing method based on scoring approach
        if args.score_w_ratings:
            # Process non-ties dataset with ratings
            logger.info("*** Run inference on non-ties subsets with ratings ***")
            dataset_formatted = dataset.map(format_ratings, fn_kwargs={"is_ties": False})
            results = []
            for i, batch in enumerate(dataset_formatted):
                if args.debug and i % 10 == 0:
                    print(f"Processing non-ties {i}/{len(dataset_formatted)}")
                result = get_vllm_judgement(batch, is_ties=False)
                results.append(result)

        else:
            # Process non-ties dataset with 4-way comparison
            logger.info("*** Run inference on non-ties subsets with 4-way comparison ***")
            if args.chat_template is not None:
                chat_template = get_conv_template(args.chat_template)
            else:
                chat_template = None
            dataset_prompts = dataset.map(format_judgements, fn_kwargs={"optional_chat_template": chat_template})

            # Generate judgements
            prompts = dataset_prompts["text"]
            prompt_ids = dataset_prompts["prompt_ids"]
            shuffle_position = dataset_prompts["shuffle_position"]

            logger.info("*** Run inference ***")
            if model_modifier == "Atla":
                logger.info("Using Atla model for inference")
                outputs = model.generate(prompt_token_ids=prompt_ids, sampling_params=sampling_params)
            else:
                outputs = model.generate(prompts, sampling_params=sampling_params)
            logger.info("*** Inference done ***")

            answers = [o.outputs[0].text for o in outputs]
            winners = [process_judgement(a, model_modifier) for a in answers]
            if args.debug:
                logger.info(f"Winners: {winners}")

            def process_shuffled(win, shuffle_position):
                options = ["A", "B", "C", "D"]
                winner_text = options.pop(shuffle_position)
                loser_texts = options

                if win == winner_text:
                    return 1
                elif win in loser_texts:
                    return 0
                else:  # if "error"
                    return 0.25  # effectively a tie

            results = [process_shuffled(w, s) for w, s in zip(winners, shuffle_position)]

        # Process ties dataset with ratings (mandatory)
        logger.info("*** Run inference on Ties subset with ratings ***")
        ties_dataset_formatted = ties_dataset.map(format_ratings, fn_kwargs={"is_ties": True})
        results_ties = []
        for i, batch in enumerate(ties_dataset_formatted):
            if args.debug and i % 10 == 0:
                print(f"Processing ties {i}/{len(ties_dataset_formatted)}")
            result = get_vllm_judgement(batch, is_ties=True)
            results_ties.append(result)

    ############################
    # Print & process results
    ############################
    # add column for results for easy printing
    out_dataset = dataset.add_column("results", results)
    out_dataset = out_dataset.add_column("id", nonties_ids)

    # process results for ties, then merge datasets
    out_dataset_ties = ties_dataset.add_column("scores", results_ties)
    out_dataset_ties, ties_score = process_single_model(out_dataset_ties)

    out_dataset = concatenate_datasets([out_dataset, out_dataset_ties], axis=0)

    # model name concat if list
    if isinstance(args.model, list):
        model_name = "_".join(args.model)
        model_name = "PoLL/" + model_name
    else:
        model_name = args.model
    # if model in openai or Anthropic list, append org to model name
    if args.model in OPENAI_MODEL_LIST:
        model_name = "openai/" + model_name
    elif args.model in ANTHROPIC_MODEL_LIST:
        model_name = "anthropic/" + model_name
    elif args.model in GEMINI_MODEL_LIST:
        model_name = "google/" + model_name

    # get core dataset
    results_grouped = {}
    results_grouped["model"] = model_name
    results_grouped["model_type"] = model_type
    results_grouped["chat_template"] = args.chat_template

    # print per subset and log into results_grouped file
    present_subsets = np.unique(out_dataset["subset"])
    logger.info(f"Present subsets: {present_subsets}")
    for subset in present_subsets:
        if subset.lower() == "ties":
            print(f"{subset}: Ties score: {ties_score}")
            results_grouped[subset] = ties_score
        else:
            subset_dataset = out_dataset.filter(lambda example: example["subset"] == subset)
            num_correct = sum(subset_dataset["results"])
            num_total = len(subset_dataset["results"])
            print(f"{subset}: {num_correct}/{num_total} ({num_correct/num_total})")
            results_grouped[subset] = num_correct / num_total

    ############################
    # Upload results to hub
    #############################
    sub_path = "eval-set/"
    results_url = save_to_hub(
        results_grouped,
        model_name,
        sub_path,
        args.debug,
        local_only=args.do_not_save,
        save_metrics_for_beaker=not args.disable_beaker_save,
        best_of_n=True,
    )
    if not args.do_not_save:
        logger.info(f"Uploaded reward model results to {results_url}")

    logger.info("Not uploading chosen-rejected text with scores due to model compatibility")

    ############################
    # Save per-prompt results to hub
    ############################
    # create new json with scores and upload
    scores_dict = out_dataset.to_dict()
    scores_dict["model"] = model_name
    scores_dict["model_type"] = model_type

    sub_path_scores = "eval-set-scores/"
    scores_url = save_to_hub(
        scores_dict, model_name, sub_path_scores, args.debug, local_only=args.do_not_save, best_of_n=True
    )

    logger.info(f"Uploading chosen-rejected text with scores to {scores_url}")


if __name__ == "__main__":
    main()