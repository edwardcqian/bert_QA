# BERT QA inferencing Module 
A simple SQUAD inferencing model using pretrained bert from `transformers` library

## Training
```
python3 run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --do_eval \
  --do_lower_case \
  --train_file data/train-v2.0.json \
  --predict_file data/dev-v2.0.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir weights/
```

## Inferencing
```python
from model_inference import ModelInference

mi = ModelInference('weights/')

mi.add_target_text((
  'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) '
  'were the people who in the 10th and 11th centuries gave their name to '
  'Normandy, a region in France. They were descended from Norse '
  '(\"Norman\" comes from \"Norseman\") raiders and pirates from Denmark, '
  'Iceland and Norway who, under their leader Rollo, '
  'agreed to swear fealty to King Charles III of West Francia. '
  'Through generations of assimilation and mixing with the native '
  'Frankish and Roman-Gaulish populations, their descendants would gradually '
  'merge with the Carolingian-based cultures of West Francia. '
  'The distinct cultural and ethnic identity of the Normans emerged initially '
  'in the first half of the 10th century, '
  'and it continued to evolve over the succeeding centuries.'
))

mi.evaluate('Where is Normandy')
# france

mi.evaluate('What are Normans called in latin?')
# normanni

mi.evaluate('When was normandy named?')
# in the 10th and 11th centuries

mi.evaluate('What kind of songs did the Normans sing?')
# No valid answers found

mi.evaluate('What is you quest?')
# No valid answers found
```