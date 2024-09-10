#!/bin/bash
set -e

python main.py --gpus 4 --seed 0 --output_dir results/bert_tagger \
                train data/conll03/train.jsonl data/conll03/dev.jsonl data/mpqa/test.jsonl \
                --mini_batch_size 32 --accumulation_steps 1 --evaluate_batch_size 64 \
                --lr 2e-5 --no_pret_lr 1e-3 --warmup_steps 500 --clip_grad_norm "200.0" \
                --epochs 200 --patience 9 --metric +F1 \
                bert_lstm_crf_tagger --plm_dir /data/hfmodel/bert-large-uncased --lstm_layers 2
