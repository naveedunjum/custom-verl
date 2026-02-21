import re
from typing import Optional, Tuple
import sacrebleu

bleu = sacrebleu.BLEU()


def compute_bleu(lg_pair: str, ref: str, pred: str) -> float:
    pred = pred if isinstance(pred, str) else ""
    tgt_lang = lg_pair.split("-")[1]
    if tgt_lang == "zh":
        tokenize = "zh"
    elif tgt_lang == "ja":
        tokenize = "ja-mecab"
    else:
        tokenize = "13a"

    refs = [[ref]]
    preds = [pred]
    bleu_result_str = str(bleu.corpus_bleu(preds, refs, tokenize=tokenize))
    match = re.search(r"BLEU = (\d+\.\d+)", bleu_result_str)
    bleu_score = float(match.group(1)) if match else 0.0

    print(f"[BLEU Score] {bleu_score}")
    return bleu_score


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        processed_str = solution_str.split(
            "<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    answer_pattern = r"<translate>(.*?)</translate>"
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if not matches:
        print("[Error] No valid <translate> tag found")
        return None, processed_str

    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def validate_response_structure(processed_str: str) -> bool:
    print("\n[Structure Validation]")
    validation_passed = True
    tags = {
        "think_start": "<think>",
        "think_end": "</think>",
        "answer_start": "<translate>",
        "answer_end": "</translate>",
    }
    positions = {}
    for tag_name, tag_str in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        print(f"  {tag_str}: count={count}, position={pos}")
        if count != 1:
            print(f"  [Error] {tag_str} appears {count} times (expected 1)")
            validation_passed = False

    if validation_passed:
            if (
                positions["think_start"] > positions["think_end"]
                or positions["think_end"] > positions["answer_start"]
                or positions["answer_start"] > positions["answer_end"]
            ):
                print(
                    "  [Error] Incorrect tag order: Expected <think>...</think><translate>...</translate>")
                validation_passed = False
            else:
                print("  Tag sequence validation passed")

    return validation_passed


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """
    verl-compatible reward function signature.
    Extra parameters are passed via extra_info:
      - lg_pair        (str)   : language pair, e.g. "en-de"  [required]
      - reward_type    (str)   : "continuous" or "discrete"   [default: "continuous"]
      - bleu_threshold (float) : threshold for discrete mode  [default: 25.0]
      - scale_factor   (float) : divisor for continuous mode  [default: 100.0]
      - check_think    (bool)  : validate <think> tags        [default: True]
      - format_reward  (float) : reward/penalty for format    [default: 1.0]
    """
    try:
        extra_info = extra_info or {}
        lg_pair = extra_info.get("lg_pair", "en-de")
        reward_type = extra_info.get("reward_type", "continuous")
        bleu_threshold = float(extra_info.get("bleu_threshold", 25.0))
        scale_factor = float(extra_info.get("scale_factor", 100.0))
        check_think = extra_info.get("check_think", True)
        format_reward = float(extra_info.get("format_reward", 1.0))

        print("\n" + "=" * 80)
        print(f" Processing Training Sample — data_source: {data_source} ".center(80, "="))

        answer_text, processed_str = extract_solution(solution_str)

        if check_think:
            format_correct = validate_response_structure(processed_str)
        else:
            format_correct = answer_text is not None

        format_score = format_reward if format_correct else -abs(format_reward)
        print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
        print(f"  Format score: {format_score}")

        # Answer reward
        answer_score = 0.0
        if format_correct and answer_text:
            bleu_score = compute_bleu(lg_pair, ground_truth, answer_text)

            if reward_type == "discrete":
                answer_score = 2.0 if bleu_score > bleu_threshold else -1.5
            elif reward_type == "continuous":
                answer_score = float(bleu_score) / scale_factor
            else:
                print(f"[Warning] Unknown reward_type '{reward_type}', defaulting to continuous.")
                answer_score = float(bleu_score) / scale_factor

            print(f"\n[Content Validation]")
            print(f"  Reference:  {ground_truth}")
            print(f"  Hypothesis: {answer_text}")
            print(f"  BLEU Score: {bleu_score}")
        else:
            answer_score = -2.0
            print("\n[Content Validation] Skipped — format errors or missing <translate> block")

        total_score = format_score + answer_score

        print("\n" + "-" * 80)
        print(" Reward Score ".center(80, "-"))
        print(f"  Format : {format_score}")
        print(f"  Answer : {answer_score}")
        print(f"  Total  : {total_score}")
        print("=" * 80 + "\n")

        return total_score

    except Exception as e:
        print(f"[Error] Reward computation failed: {e}")
        return 0.0


def compute_score_val_bleu(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """
    Validation-only reward function — returns raw BLEU score (0–100).
    Same verl-compatible signature as compute_score.
    """
    try:
        extra_info = extra_info or {}
        lg_pair = extra_info.get("lg_pair", "en-de")

        print("\n" + "=" * 80)
        print(f" Processing Validation Sample — data_source: {data_source} ".center(80, "="))
        print(f"\n[Prompt + Response]\n{solution_str}")

        answer_text, processed_str = extract_solution(solution_str)

        if answer_text:
            bleu_score = compute_bleu(lg_pair, ground_truth, answer_text)
            print(f"  Reference:  {ground_truth}")
            print(f"  Hypothesis: {answer_text}")
        else:
            # Fallback: score the raw assistant output
            bleu_score = compute_bleu(lg_pair, ground_truth, processed_str)
            print(f"  Reference:  {ground_truth}")
            print(f"  Hypothesis (raw, no tags found): {processed_str}")

        print("\n" + "-" * 80)
        print(f"  BLEU Score: {bleu_score}")
        print("=" * 80 + "\n")

        return bleu_score

    except Exception as e:
        print(f"[Error] Validation BLEU computation failed: {e}")
        return 0.0
