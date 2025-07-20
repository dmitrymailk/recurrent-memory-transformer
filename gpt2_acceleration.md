## gpt2

#### baseline
```bash
#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetune_babilong_baseline.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=decoder
MEMORY_CELL=modeling_rmt.language_modeling:MemoryCell
RECURRENT_WRAPPER=modeling_rmt.language_modeling:RecurrentWrapper
BACKBONE_CLS=transformers:AutoModelForCausalLM
NOISE_DATASET=pg19
METRIC=exact_match

MODEL_NAME=gpt2 # backbone model
# MODEL_NAME=unsloth/Llama-3.2-1B-Instruct # backbone model
# MODEL_NAME=unsloth/Llama-3.2-1B # backbone model

ITERS=5000
TBS=64
# TBS=2
NP=1

train_folder=/code/rmt_workdir
model_base_folder="$train_folder"/babilong
dataset_folder="$model_base_folder"/data/tasks_1-20_v1-2/en-10k
mkdir -p $train_folder
mkdir -p $model_base_folder
mkdir -p $dataset_folder

export HF_HUB_ENABLE_HF_TRANSFER=1

for TASK_DATASET in qa1_single-supporting-fact; do

  for LR in 1e-05; do

    for SEGMENT_SIZE in 512; do # size of one segment in tokens
      # MAX_N_SEGMENTSS=(0 1 2 4 6 8 16 32)
      MAX_N_SEGMENTSS=(0 1 2 4)
      # BSS=(32 32 16 16 8 8 4 2)
      BSS=(4 4 4 4 4 4 4 2)
      # BSS=(1 1 1 1 1 1 1 1)

      for ((j = 2; j < ${#MAX_N_SEGMENTSS[@]}; j++)); do
        MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}
        BS=${BSS[j]}

        j1=$((j - 1))
        SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j1]}

        j2=$((j - 2))
        SRC_SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j2]}

        for MEMORY_SIZE in 16; do

          SAMPLE_SIZE=$((MAX_N_SEGMENTS * SEGMENT_SIZE)) # length of task sample in tokens

          GRAD_ACC_STEPS=$(($TBS / ($BS * $NP)))

          SCHEDULER=linear

          for N in 6; do

            K2=-1 # BPTT unroll length

            NP=$NP
            # ACCEL_CONFIG=/home/jovyan/rmt/babilong/accel_configs/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
            ACCEL_CONFIG=/code/accel_configs/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
            cd accel_configs/
            python create_config.py \
              --bf16 \
              --train_batch_size $TBS --train_micro_batch_size_per_gpu $BS --gradient_accumulation_steps $GRAD_ACC_STEPS --np $NP --gradient_clipping 1.0
            cd ..
            export WANDB_RUN_NAME="$TASK_DATASET MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS"
            echo RUNNING: TASK_DATASET $WANDB_RUN_NAME
            echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME LR $LR N $N
            echo gradient accumulation steps $GRAD_ACC_STEPS

            # accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29007 run_finetuning_babilong_rmt.py \
            #         --task_dataset $TASK_DATASET \
            #         --noise_dataset $NOISE_DATASET \
            #         --babi_path /home/jovyan/rmt/babilong/data/tasks_1-20_v1-2/en-10k \
            #         --model_path /home/jovyan/rmt/runs/babilong/${TASK_DATASET}/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}_from_cpt_${SRC_N_SEGMENTS}-${MAX_N_SEGMENTS}/run_$N \
            #         --model_cpt /home/jovyan/rmt/runs/babilong/${TASK_DATASET}/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_${SRC_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}_from_cpt_${SRC_SRC_N_SEGMENTS}-${SRC_N_SEGMENTS}/run_$N/model_best \
            #         --from_pretrained $MODEL_NAME \
            #         --model_type $MODEL_TYPE \
            #         --memory_cell_cls $MEMORY_CELL \
            #         --recurrent_wrapper_cls $RECURRENT_WRAPPER \
            #         --model_cls $BACKBONE_CLS \
            #         --segment_size $SEGMENT_SIZE \
            #         --sample_size $SAMPLE_SIZE \
            #         --num_mem_tokens $MEMORY_SIZE \
            #         --max_n_segments $MAX_N_SEGMENTS\
            #         --vary_n_segments \
            #         --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
            #         --num_training_steps $((ITERS*2)) \
            #         --iters $ITERS \
            #         --use_generate_on_valid \
            #         --save_best \
            #         --k2 $K2 \
            #         --optimizer AdamW  --weight_decay 0.01 \
            #         --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
            #         --data_n_workers 2 \
            #         --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/20)) \
            #         --optimize_metric $METRIC --optimize_mode max \
            #         --show_valid_examples 5 \
            #         --early_stopping_patience 15 \
            #         --seed $(($N+42)) \
            #         --clip_grad_norm 1.0

            #       --model_cpt $model_base_folder/${TASK_DATASET}/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_${SRC_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}_from_cpt_${SRC_SRC_N_SEGMENTS}-${SRC_N_SEGMENTS}/run_$N/model_best \
            # ==============================================================================
            # === DEBUG: PRINTING ALL PARAMETERS ===========================================
            # ==============================================================================
            echo "--- DEBUG: Printing all resolved arguments before execution ---"

            echo "accelerate launch"
            echo "  --config_file: $ACCEL_CONFIG"
            echo "  --main_process_port: 29007"
            echo "  run_finetuning_babilong_rmt.py"
            echo "    --task_dataset: $TASK_DATASET"
            echo "    --noise_dataset: $NOISE_DATASET"
            echo "    --babi_path: $dataset_folder"
            echo "    --model_path: $MODEL_PATH_CONSTRUCTED"
            echo "    --from_pretrained: $MODEL_NAME"
            echo "    --model_type: $MODEL_TYPE"
            echo "    --memory_cell_cls: $MEMORY_CELL"
            echo "    --recurrent_wrapper_cls: $RECURRENT_WRAPPER"
            echo "    --model_cls: $BACKBONE_CLS"
            echo "    --segment_size: $SEGMENT_SIZE"
            echo "    --sample_size: $SAMPLE_SIZE"
            echo "    --num_mem_tokens: $MEMORY_SIZE"
            echo "    --max_n_segments: $MAX_N_SEGMENTS"
            echo "    --vary_n_segments: [flag set]"
            echo "    --batch_size: $BS"
            echo "    --gradient_accumulation_steps: $GRAD_ACCUM_STEPS (calculated from TBS=$TBS, BS=$BS, NP=$NP)"
            echo "    --num_training_steps: $TOTAL_TRAINING_STEPS (calculated from ITERS=$ITERS)"
            echo "    --iters: $ITERS"
            echo "    --save_best: [flag set]"
            echo "    --k2: $K2"
            echo "    --optimizer: AdamW"
            echo "    --weight_decay: 0.01"
            echo "    --lr: $LR"
            echo "    --lr_scheduler: $SCHEDULER"
            echo "    --num_warmup_steps: $NUM_WARMUP_STEPS (calculated from ITERS=$ITERS)"
            echo "    --data_n_workers: 0"
            echo "    --log_interval: $LOG_INTERVAL (calculated from ITERS=$ITERS)"
            echo "    --valid_interval: $VALID_INTERVAL (calculated from ITERS=$ITERS)"
            echo "    --optimize_metric: $METRIC"
            echo "    --optimize_mode: max"
            echo "    --show_valid_examples: 5"
            echo "    --early_stopping_patience: 15"
            echo "    --seed: $SEED (calculated from N=$N)"
            echo "    --clip_grad_norm: 1.0"
            # --- DEBUG: Printing all resolved arguments before execution ---
            # accelerate launch
            #   --config_file: /code/accel_configs/accelerate/deepspeed_bf16_tbs2bs1g2c1.0np1.yaml
            #   --main_process_port: 29007
            #   run_finetuning_babilong_rmt.py
            #     --task_dataset: qa1_single-supporting-fact
            #     --noise_dataset: pg19
            #     --babi_path: /code/rmt_workdir/babilong/data/tasks_1-20_v1-2/en-10k
            #     --model_path:
            #     --from_pretrained: unsloth/Llama-3.2-1B
            #     --model_type: decoder
            #     --memory_cell_cls: modeling_rmt.language_modeling:MemoryCell
            #     --recurrent_wrapper_cls: modeling_rmt.language_modeling:RecurrentWrapper
            #     --model_cls: transformers:AutoModelForCausalLM
            #     --segment_size: 512
            #     --sample_size: 1024
            #     --num_mem_tokens: 16
            #     --max_n_segments: 2
            #     --vary_n_segments: [flag set]
            #     --batch_size: 1
            #     --gradient_accumulation_steps:  (calculated from TBS=2, BS=1, NP=1)
            #     --num_training_steps:  (calculated from ITERS=5000)
            #     --iters: 5000
            #     --save_best: [flag set]
            #     --k2: -1
            #     --optimizer: AdamW
            #     --weight_decay: 0.01
            #     --lr: 1e-05
            #     --lr_scheduler: linear
            #     --num_warmup_steps:  (calculated from ITERS=5000)
            #     --data_n_workers: 0
            #     --log_interval:  (calculated from ITERS=5000)
            #     --valid_interval:  (calculated from ITERS=5000)
            #     --optimize_metric: exact_match
            #     --optimize_mode: max
            #     --show_valid_examples: 5
            #     --early_stopping_patience: 15
            #     --seed:  (calculated from N=6)
            #     --clip_grad_norm: 1.0

            accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29007 run_finetuning_babilong_rmt.py \
              --task_dataset $TASK_DATASET \
              --noise_dataset $NOISE_DATASET \
              --babi_path $dataset_folder \
              --model_path $model_base_folder/${TASK_DATASET}/$MODEL_NAME/${SCHEDULER}_adamw_wd1e-03_${MAX_N_SEGMENTS}x${SEGMENT_SIZE}_mem${MEMORY_SIZE}_bs${TBS}_bptt-${K2}_from_cpt_${SRC_N_SEGMENTS}-${MAX_N_SEGMENTS}/run_$N \
              --from_pretrained $MODEL_NAME \
              --model_type $MODEL_TYPE \
              --memory_cell_cls $MEMORY_CELL \
              --recurrent_wrapper_cls $RECURRENT_WRAPPER \
              --model_cls $BACKBONE_CLS \
              --segment_size $SEGMENT_SIZE \
              --sample_size $SAMPLE_SIZE \
              --num_mem_tokens $MEMORY_SIZE \
              --max_n_segments $MAX_N_SEGMENTS --vary_n_segments \
              --batch_size $BS --gradient_accumulation_steps $(($TBS / ($BS * $NP))) \
              --num_training_steps $((ITERS * 2)) \
              --iters $ITERS \
              --save_best \
              --k2 $K2 \
              --optimizer AdamW --weight_decay 0.01 \
              --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS / 10)) \
              --data_n_workers 0 \
              --log_interval $(($ITERS / 100)) --valid_interval $(($ITERS / 20)) \
              --optimize_metric $METRIC --optimize_mode max \
              --show_valid_examples 5 \
              --early_stopping_patience 15 \
              --seed $(($N + 42)) \
              --clip_grad_norm 1.0
            # --use_generate_on_valid \

          done
        done
      done
    done
  done
done
echo "done"
```

- memory optimizations
- default: 8858MB (opt_1)
- убрал вычисление логитов для токенов, которые мы и так выкидываем для бабилонга: 1.14it/s, 7622MB  (opt_2)
- добавил torch.compile, cut-cross-entropy, отменил вычисление логитов для ненужных токенов 1.15it/sб 6602MB (opt_3)
- добавил torch.compile, float8, cut-cross-entropy, отменил вычисление логитов для ненужных токенов 1.15it/sб 6570MB (opt_4)
- добавил torch.compile opt_5


##### A100 (llama3.2_1b)
- opt_1 - 21/5000 [02:50<6:40:31,  0.21it/s], 33.61GB
- opt_5 -  | 42/5000 [04:32<6:40:05,  0.21it/s] 33.61GB
- opt_3 -  50/5000 [04:09<5:31:40,  0.25it/s] 21.24 - убрал deepspeed, вернулся к обычному accelerate