#!/bin/bash

set +x

OUTPUT_DIR="klue_output"
DATA_DIR="data/klue_benchmark"  # default submodule for data from https://github.com/KLUE-benchmark/KLUE
VERSION="v1.1"

# YNAT
task="ynat"
for model_name in "klue/bert-base" "klue/roberta-small" "klue/roberta-base"; do
    python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION} --model_name_or_path ${model_name} --learning_rate 5e-5 --train_batch_size 32 --warmup_ratio 0.1 --max_seq_length 128 --patience 100000 --metric_key macro_f1 --gpus 0 --num_workers 4
done

python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION} --model_name_or_path klue/roberta-large --learning_rate 5e-5 --train_batch_size 32 --warmup_ratio 0.2 --max_seq_length 128 --patience 100000 --metric_key macro_f1 --gpus 0 --num_workers 4


# KLUE-STS
task="klue-sts"
for model_name in "klue/roberta-small" "klue/roberta-base" "klue/bert-base"; do
    python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION} --model_name_or_path ${model_name} --learning_rate 5e-5 --num_train_epochs 4 --train_batch_size 32 --warmup_ratio 0.1 --max_grad_norm 1.0 --weight_decay 0 --max_seq_length 128 --metric_key pearsonr --gpus 0 --num_workers 4
done

python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION} --model_name_or_path klue/roberta-large --learning_rate 2e-5 --num_train_epochs 4 --train_batch_size 32 --warmup_ratio 0.1 --max_grad_norm 1.0 --weight_decay 0 --max_seq_length 128 --metric_key pearsonr --gpus 0 --num_workers 4


# KLUE-NLI
task="klue-nli"
for model_name in "klue/roberta-small" "klue/roberta-base" "klue/bert-base"; do
    python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION} --model_name_or_path ${model_name} --learning_rate 5e-5 --num_train_epochs 5 --train_batch_size 32 --max_grad_norm 1.0 --warmup_ratio 0.1 --weight_decay 0 --max_seq_length 128 --metric_key accuracy --gpus 0 --num_workers 4
done

python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path klue/roberta-large --learning_rate 5e-5 --num_train_epochs 5 --train_batch_size 32 --max_grad_norm 1.0 --warmup_ratio 0.2 --weight_decay 0 --max_seq_length 128 --metric_key accuracy --gpus 0 --num_workers 4


# KLUE-NER
task="klue-ner"
for model_name in "klue/roberta-small" "klue/roberta-base" "klue/bert-base"; do
    python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path ${model_name} --num_train_epochs 3 --max_seq_length 510 --metric_key character_macro_f1 --gpus 0 --num_workers 4
done

python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path klue/roberta-large --learning_rate 1e-5 --num_train_epochs 3 --train_batch_size 8 --max_seq_length 510 --metric_key character_macro_f1 --gpus 0 --num_workers 4


# KLUE-RE
task="klue-re"
for model_name in "klue/roberta-small" "klue/roberta-base" "klue/bert-base"; do
    python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path ${model_name} --learning_rate 5e-5 --num_train_epochs 3 --train_batch_size 32 --warmup_ratio 0.1 --patience 10000 --max_seq_length 256 --metric_key micro_f1 --gpus 0 --num_workers 4
done

python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path klue/roberta-large --learning_rate 2e-5 --num_train_epochs 10 --train_batch_size 16 --warmup_ratio 0.2 --patience 10000 --max_seq_length 256 --metric_key micro_f1 --gpus 0 1 --num_workers 8


# KLUE-DP
task="klue-dp"
for model_name in "klue/roberta-small" "klue/roberta-base" "klue/bert-base"; do
    python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path ${model_name} --learning_rate 5e-5 --num_train_epochs 10 --warmup_ratio 0.1 --train_batch_size 32 --patience 10000 --max_seq_length 256 --metric_key las_macro_f1 --gpus 0 --num_workers 4
done

python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path klue/roberta-large --learning_rate 5e-5 --num_train_epochs 15 --warmup_ratio 0.2 --train_batch_size 32 --patience 10000 --max_seq_length 256 --metric_key uas_macro_f1 --gpus 0 --num_workers 4


# KLUE-MRC
task="klue-mrc"
for model_name in "klue/roberta-small" "klue/roberta-base" "klue/roberta-large"; do
    python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path ${model_name} --learning_rate 3e-5 --num_train_epochs 5 --train_batch_size 16 --patience 100000 --max_seq_length 510 --metric_key rouge_w --gpus 0 --num_workers 4
done

python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path klue/bert-base --learning_rate 3e-5 --num_train_epochs 5 --train_batch_size 16 --patience 100000 --max_seq_length 512 --metric_key rouge_w --gpus 0 --num_workers 4


# WoS
task="wos"
for model_name in "klue/roberta-small" "klue/roberta-base" "klue/bert-base"; do
    python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path ${model_name} --learning_rate 3e-5 --num_train_epochs 5 --train_batch_size 16 --eval_batch_size 16 --max_seq_length 510 --gradient_accumulation_steps 1 --warmup_ratio 0.1 --weight_decay 0.01 --max_grad_norm 1.0 --patience 100000 --parallel_decoding --metric_key slot_micro_f1 --truncate --gpus 0 --num_workers 4
done

python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path klue/roberta-large --learning_rate 2e-5 --num_train_epochs 5 --train_batch_size 8 --eval_batch_size 8 --max_seq_length 510 --gradient_accumulation_steps 2 --warmup_ratio 0.1 --weight_decay 0.01 --max_grad_norm 1.0 --patience 100000 --parallel_decoding --metric_key slot_micro_f1 --truncate --gpus 0 --num_workers 4

# for multi gpu
#for model_name in "klue/roberta-small" "klue/roberta-base" "klue/bert-base"; do
#    python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path ${model_name} --learning_rate 3e-5 --num_train_epochs 10 --train_batch_size 32 --eval_batch_size 32 --max_seq_length 510 --gradient_accumulation_steps 1 --warmup_ratio 0.2 --weight_decay 0.01 --max_grad_norm 1.0 --patience 100000 --metric_key slot_micro_f1 --gpus 0 1 --num_workers 8
#done

#python run_klue.py train --task ${task} --output_dir ${OUTPUT_DIR} --data_dir ${DATA_DIR}/${task}-${VERSION}  --model_name_or_path klue/roberta-large --learning_rate 3e-5 --num_train_epochs 10 --train_batch_size 16 --eval_batch_size 16 --max_seq_length 510 --gradient_accumulation_steps 2 --warmup_ratio 0.2 --weight_decay 0.01 --max_grad_norm 1.0 --patience 100000 --metric_key slot_micro_f1 --gpus 0 1 --num_workers 8

