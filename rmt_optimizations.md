### opt_1 (gpt2, default)
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

MODEL_NAME=gpt2  # backbone model

ITERS=5000
TBS=64

train_folder=/code/rmt_workdir
model_base_folder="$train_folder"/babilong
dataset_folder="$model_base_folder"/data/tasks_1-20_v1-2/en-10k
noise_folder="$train_folder"/pg19

mkdir -p $train_folder
mkdir -p $model_base_folder
mkdir -p $dataset_folder
mkdir -p $noise_folder

for TASK_DATASET in qa1_single-supporting-fact;
do
  
  for LR in 1e-05;
  do
    
    for SEGMENT_SIZE in 512;
    do # size of one segment in tokens
      
      MAX_N_SEGMENTSS=(0 1 2 4 6 8 16 32)
      BSS=(32 32 16 16 8 8 4 2)
      
      for (( j=2; j<${#MAX_N_SEGMENTSS[@]}; j++ ));
      do
        MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}
        BS=${BSS[j]}
        
        j1=$((j-1))
        SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j1]}
        
        j2=$((j-2))
        SRC_SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j2]}
        
        
        for MEMORY_SIZE in 16;
        do
          
          SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens
          
          GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
          
          SCHEDULER=linear
          
          for N in 6;
          do
            
            K2=-1   # BPTT unroll length
            
            NP=$NP
            #     ACCEL_CONFIG=/home/jovyan/rmt/babilong/accel_configs/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
            ACCEL_CONFIG=/code/accel_configs/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
            cd accel_configs/
            python create_config.py \
            --bf16 \
            --train_batch_size $TBS\
            --train_micro_batch_size_per_gpu $BS\
            --gradient_accumulation_steps $GRAD_ACC_STEPS\
            --np $NP\
            --gradient_clipping 1.0
            cd ..
            
            echo RUNNING: TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
            echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME  LR $LR N $N
            echo gradient accumulation steps $GRAD_ACC_STEPS
            
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
            --max_n_segments $MAX_N_SEGMENTS\
            --vary_n_segments \
            --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
            --num_training_steps $((ITERS*2)) \
            --iters $ITERS \
            --save_best \
            --k2 $K2 \
            --optimizer AdamW  --weight_decay 0.01 \
            --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
            --data_n_workers 2 \
            --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/20)) \
            --optimize_metric $METRIC --optimize_mode max \
            --show_valid_examples 5 \
            --early_stopping_patience 15 \
            --seed $(($N+42)) \
            --clip_grad_norm 1.0 \
            --opt_level "opt_1_MEM_SIZE_$MEMORY_SIZE SEG_SIZE_$SEGMENT_SIZE MAX_N_SEG_$MAX_N_SEGMENTS"
            # --use_generate_on_valid \
            
          done
        done
      done
    done
  done
done

echo "done"
```

- 14:51<1:04:22,  1.06it/s,
- 23.34GB
Проблемы с изначальным кодом что результаты отличаются от запусков и сидов.

Также если запускать оригинальный код, на середине обучения лоссы и метрики accuracy сильно разнятся. 
Например 112к шагов
- 1) 0.91
- 2) 0.82
- 3) 0.51

В оригинальном скрипте если изменить параметры размера батча
- с TBS=64 до TBS=8 и сам batch_size изменить с 16 до 4. Скрипт выполняется 13 минут, модель также обучается до 95%

### opt_2 (gpt2, default, set seed)
- поставил везде сиды и написал 
```python
torch.manual_seed(args.seed)
random.seed(args.seed)
np.random.seed(args.seed)
```
- в семплер тоже передаю seed, но графики от запусков отличаются.
- решил отказаться. критерием того что ничего не сломалось является достижение 95% и более на valid


### opt_3 (gpt2, flash attn, bfloat16)
- добавил flash attention и загрузку весов сразу в bfloat16 
- [06:13<1:05:18,  1.17it/s
- 21.55GB

### opt_4 (gpt2, flash attn, bfloat16, no_deepspeed)
- OOM
================================
================================
================================
================================
================================
================================
Я короч еще подробнее разобрал логику обучения rmt babilong. получается что в коде 
1) объявляется глобальный размер батча. допустим 64. 
2) потом решается на основе локального батча сколько мы хотим делать gradient accumulation https://github.com/booydar/recurrent-memory-transformer/blob/babilong-release/scripts_exp/babilong/finetune_babilong_qa1_rmt_vary_n_seg_iter_tasks_curriculum.sh#L91

3) затем мы берет датасет, в котором 9999 примеров, засовываем в даталоадер, который берет в качестве батча глобальный https://github.com/booydar/recurrent-memory-transformer/blob/babilong-release/run_finetuning_babilong_rmt.py#L256

4) далее в трейне логика такая, мы семплируем из даталоадера до тех пор пока не совершим заявленные 5000*2 шагов https://github.com/booydar/recurrent-memory-transformer/blob/babilong-release/lm_experiments_tools/trainer.py#L390

5) на этапе train step совершается разбиение глобального батча, на батчи размером допустим 16, чтобы они уже непосредственно влезли на устройство.

Только из этого получается что количество увиденных токенов контролируется не эпохами и общем размером датасета, а размером батча. Что в целом сложно переносить между устройствами и мы становимся зависимыми от gradient accumulation чтобы сохранить сравнимые запуски. Плюс аккумуляция даже медленее иногда, чем просто сделать шаг меньшего размера.


Всвязи с этим переписал код. Теперь акумуляция 1, количество обучения измеряется в эпохах.

### opt_1 (gpt2, default)
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

MODEL_NAME=gpt2  # backbone model

ITERS=5000
# TBS=64
TBS=19

train_folder=/code/rmt_workdir
model_base_folder="$train_folder"/babilong
dataset_folder="$model_base_folder"/data/tasks_1-20_v1-2/en-10k
noise_folder="$train_folder"/pg19

mkdir -p $train_folder
mkdir -p $model_base_folder
mkdir -p $dataset_folder
mkdir -p $noise_folder

opt_level="opt_1"

for TASK_DATASET in qa1_single-supporting-fact;
do
  
  for LR in 1e-05;
  do
    
    for SEGMENT_SIZE in 512;
    do # size of one segment in tokens
      
      MAX_N_SEGMENTSS=(0 1 2 4 6 8 16 32)
      BSS=(32 32 $TBS 16 8 8 4 2)
      
      for (( j=2; j<${#MAX_N_SEGMENTSS[@]}; j++ ));
      do
        MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}
        BS=${BSS[j]}
        
        j1=$((j-1))
        SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j1]}
        
        j2=$((j-2))
        SRC_SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j2]}
        
        
        for MEMORY_SIZE in 16;
        do
          
          SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens
          
          # GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
          GRAD_ACC_STEPS=1
          
          SCHEDULER=linear
          
          for N in 6;
          do
            
            K2=-1   # BPTT unroll length
            
            NP=$NP
            #     ACCEL_CONFIG=/home/jovyan/rmt/babilong/accel_configs/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
            ACCEL_CONFIG=/code/accel_configs/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
            cd accel_configs/
            python create_config.py \
            --bf16 \
            --train_batch_size $TBS\
            --train_micro_batch_size_per_gpu $BS\
            --gradient_accumulation_steps $GRAD_ACC_STEPS\
            --np $NP\
            --gradient_clipping 1.0
            cd ..
            
            echo RUNNING: TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
            echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME  LR $LR N $N
            echo gradient accumulation steps $GRAD_ACC_STEPS
            
            # accelerate launch run_finetuning_babilong_rmt.py \
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
            --max_n_segments $MAX_N_SEGMENTS\
            --vary_n_segments \
            --batch_size $BS --gradient_accumulation_steps 1 \
            --num_training_steps $((ITERS*2)) \
            --iters $ITERS \
            --save_best \
            --k2 $K2 \
            --optimizer AdamW  --weight_decay 0.01 \
            --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
            --data_n_workers 2 \
            --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/20)) \
            --optimize_metric $METRIC --optimize_mode max \
            --show_valid_examples 5 \
            --early_stopping_patience 15 \
            --seed $(($N+42)) \
            --clip_grad_norm 1.0 \
            --opt_name "$opt_level $MODEL_NAME MEM_SIZE_$MEMORY_SIZE SEG_SIZE_$SEGMENT_SIZE MAX_N_SEG_$MAX_N_SEGMENTS" \
            --opt_level $opt_level \
            --max_epochs 25
            # --use_generate_on_valid \
            # --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
            
          done
        done
      done
    done
  done
done

echo "done"
```
- [01:50<1:04:40,  3.31it/s
- 23.26GB

==========================
==========================
==========================
==========================
==========================
==========================
# LLAMA3.2
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

# MODEL_NAME=gpt2  # backbone model
MODEL_NAME=unsloth/Llama-3.2-1B  # backbone model

ITERS=5000
# TBS=64
TBS=4

train_folder=/code/rmt_workdir
model_base_folder="$train_folder"/babilong
dataset_folder="$model_base_folder"/data/tasks_1-20_v1-2/en-10k
noise_folder="$train_folder"/pg19

mkdir -p $train_folder
mkdir -p $model_base_folder
mkdir -p $dataset_folder
mkdir -p $noise_folder

opt_level="opt_1"

for TASK_DATASET in qa1_single-supporting-fact;
do
  
  for LR in 1e-05;
  do
    
    for SEGMENT_SIZE in 512;
    do # size of one segment in tokens
      
      MAX_N_SEGMENTSS=(0 1 2 4 6 8 16 32)
      BSS=(32 32 $TBS 16 8 8 4 2)
      
      for (( j=2; j<${#MAX_N_SEGMENTSS[@]}; j++ ));
      do
        MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]}
        BS=${BSS[j]}
        
        j1=$((j-1))
        SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j1]}
        
        j2=$((j-2))
        SRC_SRC_N_SEGMENTS=${MAX_N_SEGMENTSS[j2]}
        
        
        for MEMORY_SIZE in 16;
        do
          
          SAMPLE_SIZE=$((MAX_N_SEGMENTS*SEGMENT_SIZE)) # length of task sample in tokens
          
          # GRAD_ACC_STEPS=$(($TBS/($BS*$NP)))
          GRAD_ACC_STEPS=1
          
          SCHEDULER=linear
          
          for N in 6;
          do
            
            K2=-1   # BPTT unroll length
            
            NP=$NP
            #     ACCEL_CONFIG=/home/jovyan/rmt/babilong/accel_configs/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
            ACCEL_CONFIG=/code/accel_configs/accelerate/deepspeed_bf16_tbs${TBS}bs${BS}g${GRAD_ACC_STEPS}c1.0np${NP}.yaml
            cd accel_configs/
            python create_config.py \
            --bf16 \
            --train_batch_size $TBS\
            --train_micro_batch_size_per_gpu $BS\
            --gradient_accumulation_steps $GRAD_ACC_STEPS\
            --np $NP\
            --gradient_clipping 1.0
            cd ..
            
            echo RUNNING: TASK_DATASET $TASK_DATASET MEMORY_SIZE $MEMORY_SIZE SEGMENT_SIZE $SEGMENT_SIZE MAX_N_SEGMENTS $MAX_N_SEGMENTS
            echo SAMPLE_SIZE $SAMPLE_SIZE MODEL_NAME $MODEL_NAME  LR $LR N $N
            echo gradient accumulation steps $GRAD_ACC_STEPS
            
            # accelerate launch --config_file $ACCEL_CONFIG --main_process_port 29007 run_finetuning_babilong_rmt.py \
            accelerate launch run_finetuning_babilong_rmt.py \
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
            --max_n_segments $MAX_N_SEGMENTS\
            --vary_n_segments \
            --batch_size $BS --gradient_accumulation_steps 1 \
            --num_training_steps $((ITERS*2)) \
            --iters $ITERS \
            --save_best \
            --k2 $K2 \
            --optimizer AdamW  --weight_decay 0.01 \
            --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
            --data_n_workers 2 \
            --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/20)) \
            --optimize_metric $METRIC --optimize_mode max \
            --show_valid_examples 5 \
            --early_stopping_patience 15 \
            --seed $(($N+42)) \
            --clip_grad_norm 1.0 \
            --opt_name "$opt_level $MODEL_NAME MEM_SIZE_$MEMORY_SIZE SEG_SIZE_$SEGMENT_SIZE MAX_N_SEG_$MAX_N_SEGMENTS" \
            --opt_level $opt_level \
            --max_epochs 3
            # --use_generate_on_valid \
            # --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
            
          done
        done
      done
    done
  done
done

echo "done"

```
#### opt_1(llama3.2, bfloat16, flash_attn_2)
- из-за того что токенизаторы работают по разному в gpt-2 и llama. llama добавляет дополнительные токены, когда мы токенизируем слова. получалось что специальный токен генерации был вместо [11332], был [128000, 11332]. То есть добавлялся лишний BOS. Из-за этого функции для подсчета метрик ошибались. Тоже самое происходило и в классе NoiseInjectionDataset, в 3 части фактов, вопросов и ответов добавлялся BOS что сильно смущало модель и она показывала плохие метрики. После того как я пофиксил это метрики выросли до обычных 0.98 на трейне, 0.92 на valid.
- от deepspeed я вообще отказался, он почему-то приводит к OOM.

- [07:20<42:10,  2.64it/s, batch=4, mem=22.65 ~ 49:30

#### opt_2(torch-compile)
- [03:41<41:45,  2.83it/s, batch=4, mem=20.44 ~ 45:26
- [03:35<39:48,  2.37it/s, batch=5, mem=22.82 ~ 43:23

#### opt_3(torch-compile+float8)
- [03:19<39:48,  3.00it/s, batch=4, mem=20.77 ~ 43:07

#### opt_4(torch-compile+max-autotune+float8)
- compilation error

#### opt_5(cut_cross_entropy)
Пришлось переписать форварды для LlamaForCausalLM, MemoryCell, RecurrentWrapper.
Данный метод не возвращает логитов при обучении, потому что выполняет вычисления лосса inplace. Это влечет за собой что мы не можем считать на трейне никакие метрики кроме loss. Потому что если их считать смысл в лоссе теряется. Но так как на valid нужны метрики, приходится постоянно свитчится между оригиными функциями и модифицированными.

На данном этапе лосс считается так. Мы не вычисляем логиты по всем сегментам, сохраняем только hidden_states. Только когда прошли все сегменты, по маске выбираем только те hidden_states, которые необходимы для подсчета ошибок, лишние в логиты не преобразуем ну и наконец передаем отобраные hidden_states в cut_cross_entropy.

Думаю если отбор по маске проводить сразу и не собирать лишние hidden_states можно еще больше памяти сохранить

- [06:12<33:46,  3.31it/s, batch=4, mem=19.35 ~ 39:58

#### opt_6(cut_cross_entropy+torch-compile+float8)

- [04:25<29:54,  3.85it/s, batch=4, mem=19.1 ~ 34:19, в 3.85/2.64=1.458 быстрее оригинала
- [03:14<25:15,  3.07it/s, batch=6, mem=22.8
- [10:17<17:02,  2.54it/s, batch=8, mem=23.41 ~ 27:19 в (49*60+30)/(27*60+19)=1.812
- OOM после 07:14, [06:38<19:18,  2.30it/s, batch=9, mem=23.31 ~ 25:56, (49*60+30)/(25*60+56)=1.9087

#### opt_7(cut_cross_entropy+torch-compile+float8+bnb.optim.Adam8bit)
- [07:41<17:04,  2.76it/s,, batch=8, mem=22.59 ~ 24:45, (49*60+30)/(24*60+45)=2.0
- OOM после 06:42, batch=9, mem=
- OOM после 04:57 [04:23<19:12,  2.23it/s, batch=10, mem=22.93
