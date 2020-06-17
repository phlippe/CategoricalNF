# Language modeling

This experiment folder summarizes the experiments on language modeling.

## Training

To train a Categorical Normalizing Flow on Penn-Treebank, use the following command:
```
python train.py --dataset penntreebank \
                --max_iterations 100000 \
                --max_seq_len 288 \
                --batch_size 128 \
                --encoding_dim 3 \
                --coupling_hidden_layers 1 \
                --coupling_num_mixtures 51 \
                --coupling_dropout 0.3 \
                --coupling_input_dropout 0.1 \
                --optimizer 4 \
                --learning_rate 7.5e-4 \
                --cluster
```
For training the LSTM baseline instead, add the argument `--use_rnn` to the command above.

For training the models on text8, use the following command:
```
python train.py --dataset text8 \
                --max_iterations 100000 \
                --max_seq_len 256 \
                --batch_size 128 \
                --encoding_dim 3 \
                --coupling_hidden_layers 2 \
                --coupling_num_mixtures 27 \
                --coupling_dropout 0.0 \
                --coupling_input_dropout 0.05 \
                --optimizer 4 \
                --learning_rate 7.5e-4 \
                --cluster
```

For training the models on Wikitext, use the following command:
```
python train.py --dataset wikitext \
                --max_iterations 100000 \
                --max_seq_len 256 \
                --batch_size 128 \
                --encoding_dim 10 \
                --coupling_hidden_layers 2 \
                --coupling_num_mixtures 64 \
                --coupling_dropout 0.0 \
                --coupling_input_dropout 0.05 \
                --optimizer 4 \
                --learning_rate 7.5e-4 \
                --cluster
```


## Evaluation and Pretrained models

We plan to release pretrained models for language modeling along with evaluation scripts soon as well.

## Results

| Model | Penn-Treebank | Text8 | Wikitext103 |
|---|---|---|---|
| LSTM baseline | 1.28bpd | 1.44bpd | 4.81bpd |
| Categorical Normalizing Flow | 1.27bpd | 1.45bpd | 5.43bpd |
