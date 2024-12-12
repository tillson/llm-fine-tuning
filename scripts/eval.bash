CUDA_VISIBLE_DEVICES="1" ACCELERATE_LOG_LEVEL=info
TRAINING_CONFIGS=(
  "training_configs/mistral-7b-base-simpo.yaml"
  "training_configs/mistral-7b-base-simpo-length.yaml"
  "training_configs/mistral-7b-base-simpo-complexity.yaml"
  "training_configs/mistral-7b-base-dpo.yaml"
  "training_configs/mistral-7b-base-dpo-length.yaml"
  "training_configs/mistral-7b-base-dpo-complexity.yaml"
)

for config in "${TRAINING_CONFIGS[@]}"; do
  for checkpoint in $(seq 0.1 0.1 1.0); do
    CUDA_VISIBLE_DEVICES="1" ACCELERATE_LOG_LEVEL=info accelerate launch --main_process_port 25566 \
      --config_file accelerate_configs/deepspeed_zero3_multi.yaml \
      scripts/eval_simpo.py "$config" --checkpoint="$checkpoint"
    #   break
  done
#   break
done
