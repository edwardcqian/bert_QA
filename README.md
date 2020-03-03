# BERT QA inferencing Module 
A simple SQUAD inferencing model using pretrained bert from `transformers` library

## Training
```
python run_squad.py \
  --model_type bert \
  --model_name_or_path weights \
  --do_eval \
  --do_eval \
  --do_lower_case \
  --train_file train-v2.0.json \
  --predict_file dev-v2.0.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir /tmp/debug_squad/
```

## Inferencing
```python
from model_inference import ModelInference

mi = ModelInference('weights/')

mi.add_target_text('some paragraph of interest')

mi.evaluate('some question')
```