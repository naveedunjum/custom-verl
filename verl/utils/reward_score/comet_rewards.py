import re
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# 1.  Lazy model singletons
# ---------------------------------------------------------------------------

_comet_model = None      # reference-based
_cometkiwi_model = None  # reference-free


def _get_comet_model(model_path: str = "Unbabel/wmt22-comet-da"):
    global _comet_model
    if _comet_model is None:
        import torch
        from comet import download_model, load_from_checkpoint
        _comet_model = load_from_checkpoint(download_model(model_path))
        _comet_model = _comet_model.to(torch.cuda.current_device())
    return _comet_model


def _get_cometkiwi_model(model_path: str = "Unbabel/wmt23-cometkiwi-da-xl"):
    global _cometkiwi_model
    if _cometkiwi_model is None:
        import torch
        from comet import download_model, load_from_checkpoint
        _cometkiwi_model = load_from_checkpoint(download_model(model_path))
        _cometkiwi_model = _cometkiwi_model.to(torch.cuda.current_device())
    return _cometkiwi_model


# ---------------------------------------------------------------------------
# 2.  Shared helpers (identical logic to bleu_reward.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3.  Core COMET scoring helpers
# ---------------------------------------------------------------------------

def _score_comet(
    source: str,
    hypothesis: str,
    reference: str,
    model_path: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
) -> float:
    model = _get_comet_model(model_path)
    data = [{"src": source, "mt": hypothesis, "ref": reference}]
    output = model.predict(data, batch_size=batch_size, gpus=0)
    return float(output.scores[0])


def _score_cometkiwi(
    source: str,
    hypothesis: str,
    model_path: str = "Unbabel/wmt23-cometkiwi-da-xl",
    batch_size: int = 8,
) -> float:
    model = _get_cometkiwi_model(model_path)
    data = [{"src": source, "mt": hypothesis}]
    output = model.predict(data, batch_size=batch_size, gpus=0)
    return float(output.scores[0])


# ---------------------------------------------------------------------------
# 4.  Public reward entry points  (called from __init__.py dispatcher)
# ---------------------------------------------------------------------------

def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    try:
        extra_info = extra_info or {}
        src_text = extra_info.get("src_text", "")
        model_path = extra_info.get("comet_model_path", "Unbabel/wmt22-comet-da")
        reward_type = extra_info.get("reward_type", "continuous")
        comet_threshold = float(extra_info.get("comet_threshold", 0.75))
        format_reward = float(extra_info.get("format_reward", 1.0))
        check_think = extra_info.get("check_think", True)

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

        answer_score = 0.0
        if format_correct and answer_text:
            if not src_text:
                print("[Warning] src_text missing from extra_info — cannot compute COMET. Returning format score only.")
                return float(format_score)

            comet_score = _score_comet(
                source=src_text,
                hypothesis=answer_text,
                reference=ground_truth,
                model_path=model_path,
            )

            if reward_type == "discrete":
                answer_score = 2.0 if comet_score > comet_threshold else -1.5
            elif reward_type == "continuous":
                answer_score = float(comet_score)
            else:
                print(f"[Warning] Unknown reward_type '{reward_type}', defaulting to continuous.")
                answer_score = float(comet_score)

            print(f"\n[Content Validation]")
            print(f"  Source:      {src_text}")
            print(f"  Reference:   {ground_truth}")
            print(f"  Hypothesis:  {answer_text}")
            print(f"  COMET Score: {comet_score}")
        else:
            answer_score = -2.0
            print("\n[Content Validation] Skipped — format errors or missing <translate> block")

        total_score = format_score + answer_score

        print("\n" + "-" * 80)
        print(" Reward Score ".center(80, "-"))
        print(f"  Ground Truth: {ground_truth}")
        print(f"  Answer text:  {answer_text}")
        print(f"  Format : {format_score}")
        print(f"  Answer : {answer_score}")
        print(f"  Total  : {total_score}")
        print("=" * 80 + "\n")

        return total_score

    except Exception as e:
        print(f"[Error] COMET reward computation failed: {e}")
        return 0.0


def compute_score_kiwi(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    try:
        extra_info = extra_info or {}
        src_text = extra_info.get("src_text", "")
        model_path = extra_info.get("cometkiwi_model_path", "Unbabel/wmt23-cometkiwi-da-xl")
        reward_type = extra_info.get("reward_type", "continuous")
        kiwi_threshold = float(extra_info.get("kiwi_threshold", 0.75))
        format_reward = float(extra_info.get("format_reward", 1.0))
        check_think = extra_info.get("check_think", True)

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

        answer_score = 0.0
        if format_correct and answer_text:
            if not src_text:
                print("[Warning] src_text missing from extra_info — cannot compute COMETKiwi. Returning format score only.")
                return float(format_score)

            kiwi_score = _score_cometkiwi(
                source=src_text,
                hypothesis=answer_text,
                model_path=model_path,
            )

            if reward_type == "discrete":
                answer_score = 2.0 if kiwi_score > kiwi_threshold else -1.5
            elif reward_type == "continuous":
                answer_score = float(kiwi_score)
            else:
                print(f"[Warning] Unknown reward_type '{reward_type}', defaulting to continuous.")
                answer_score = float(kiwi_score)

            print(f"\n[Content Validation]")
            print(f"  Source:          {src_text}")
            print(f"  Hypothesis:      {answer_text}")
            print(f"  COMETKiwi Score: {kiwi_score}")
        else:
            answer_score = -2.0
            print("\n[Content Validation] Skipped — format errors or missing <translate> block")

        total_score = format_score + answer_score

        print("\n" + "-" * 80)
        print(" Reward Score ".center(80, "-"))
        print(f"  Answer text:  {answer_text}")
        print(f"  Format : {format_score}")
        print(f"  Answer : {answer_score}")
        print(f"  Total  : {total_score}")
        print("=" * 80 + "\n")

        return total_score

    except Exception as e:
        print(f"[Error] COMETKiwi reward computation failed: {e}")
        return 0.0
