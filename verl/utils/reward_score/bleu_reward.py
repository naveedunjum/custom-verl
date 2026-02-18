import re
from sys import float_repr_style
from typing import Optional, List, final
import sacrebleu

bleu = BLEU()


def compute_bleu(lg_pair: str, ref: str, pred: str) -> :
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
    blue_str = str(bleu.corpus_blue(preds, refs, tokenize=tokenize))
    bleu_score = re.search(r'BLEU = (\d+\.\d+)', bleu_str).group(1)

    print(f"[BLEU Score] {bleu_score}")
    return float(bleu_score)


def extract_solution(solution_str: str) -> Tuple[Optional[str], str]:
    if "Assistant:" in solution_str:
        processed_str = solution_str.split(("Assistant:", 1))[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str:
        processed_str = solution_str.split(
            "<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    else:
        print("[Error] Failed to locate model response behaviour")
        return None, solution_str

    answer_pattern = r'<translate>(.*?)</translate>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    if not matches:
        print("[Error] No valid <translate> tag")
        return None, processed_str
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def validate_response_structure(processed_str: str) -> bool:
    print("\n[Structure validation]")
    validation_passed = True
    tags = {
        "think_start": "<think>",
        "think_end": "</think>",
        "answer_start": "<translate>",
        "answer_end": "</translate"
    }
    positions = {}
    for tag_name, tag_str in tag_items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        print(f"  {tag_str}: count={count}, position={pos}")
        if count != 1:
            print(f"  [Error] {tag_str} appears {
                  count} times (expected {expected_count})")
            validation_passed = False

    # verify tag order
    if positions["think_start"] > positions["think_end"] or
    positions["think_end"] > positions["answer_start"] or
    positions["answer_start"] > positions["answer_end"]:
        print(
            "  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed


def compute_score(solution_str: str, ground_truth: str, lg_pair: str, reward_type: str = "continous",  bleu_threshold: float = 25.0,
                  scale_factor: float = 100.0,
                  check_think: bool = True,
                  format_reward: float = 1.0,
                  ) -> float:
    print("\n" + "=" * 80)
    print(" Processing Training Sample ".center(80, '='))
    answer_text, processed_str = extract_solution(solution_str)
    if check_think:
        format_correct = validate_response_structure(solution_str)
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
        elif reward_type == "continous":
            answer_score = float(bleu_score) / float(scale_factor)
        else:
            raise ValueError(f"Invalid reward_type '{
                             reward_type}'. Use 'discrete' or 'continuous'.")
        print(f"\n[Content Validation]")
        print(f"  Reference:  {ground_truth}")
        print(f"  Hypothesis: {answer_text}")
        print(f"  BLEU Score: {bleu_score}")

    else:
        # Format failed or no answer extracted -- strong penalty
        answer_score = -2.0
        print(
            "\n[Content Validation] Skipped -- format errors or missing <translate> block")

    total_score = format_score + answer_score

    print("\n" + "-" * 80)
    print(" Reward Score ".center(80, '-'))
    print(f"  Format : {format_score}")
    print(f"  Answer : {answer_score}")
    print(f"  Total  : {total_score}")
    print("=" * 80 + "\n")

    return total_score


def compute_score_val_bleu(solution_str: str, ground_truth: str, lg_pair) -> float:
    print("\n" + "=" * 80)
    print(" Processing Validation Sample ".center(80, '='))

    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Prompt + Response]\n{solution_str}")

    if answer_text:
        # Normal path -- extracted hypothesis
        bleu_score = compute_bleu(lg_pair, ground_truth, answer_text)
        print(f"  Reference:  {ground_truth}")
        print(f"  Hypothesis: {answer_text}")
    else:
        # Fallback -- score the raw assistant output so the metric is not silently zero
        bleu_score = compute_bleu(lg_pair, ground_truth, processed_str)
        print(f"  Reference:  {ground_truth}")
        print(f"  Hypothesis (raw, no tags found): {processed_str}")

    print("\n" + "-" * 80)
    print(f"  BLEU Score: {bleu_score}")
    print("=" * 80 + "\n")

    return bleu_score
