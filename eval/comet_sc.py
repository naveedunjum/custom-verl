import sys
from comet import load_from_checkpoint

# model_path = "/hnvme/workspace/slcl100h-vllm/custom-verl/comet_models/wmt22-cometkiwi-da/checkpoints/model.ckpt" 
# src_file = "/hnvme/workspace/slcl100h-vllm/custom-verl/compare_results_models/hnvme/workspace/slcl100h-vllm/custom-verl/trained_models/qwen_urdu/en-ur/texts/all_source.txt"
# hyp_file = "/hnvme/workspace/slcl100h-vllm/custom-verl/compare_results_models/hnvme/workspace/slcl100h-vllm/custom-verl/trained_models/qwen_urdu/en-ur/texts/translations.txt"
# ref_file = "/hnvme/workspace/slcl100h-vllm/custom-verl/compare_results_models/hnvme/workspace/slcl100h-vllm/custom-verl/trained_models/qwen_urdu/en-ur/texts/all_target.txt"
model_path = sys.argv[1]
src_file = sys.argv[2]
hyp_file = sys.argv[3]
ref_file = sys.argv[4] if len(sys.argv) > 4 else None

src = open(src_file).read().splitlines()
hyp = open(hyp_file).read().splitlines()
print(f"Loaded {len(src)} source sentences and {len(hyp)} hypotheses.")
model = load_from_checkpoint(model_path)
print(f"Loaded COMET model from {model_path}")

if ref_file:
    ref = open(ref_file).read().splitlines()
    data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(src, hyp, ref)]
else:
    data = [{"src": s, "mt": h} for s, h in zip(src, hyp)]

output = model.predict(data, batch_size=64, gpus=1)
for score in output.scores:
    print(score)
print(f"Score: {output.system_score}")