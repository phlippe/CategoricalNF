# Set modeling

This experiment folder summarizes the experiments on set shuffling and set summation. The definition of the datasets can be found at [datasets/set_shuffling.py](datasets/set_shuffling.py) and [datasets/set_summation.py](datasets/set_summation.py).

## Training

To train a Categorical Normalizing Flow with mixture model encoding on set shuffling, use the following command:
```
python train.py --dataset shuffling \
                --max_iterations 100000 \
                --batch_size 1024 \
                --encoding_dim 4 \
                --optimizer 4 \
                --learning_rate 7.5e-4 \
                --cluster
```
For training on set summation, replace `--dataset shuffling` with `--dataset summation`. The `--cluster` argument is set to reduce the printed output to stdout for longer experiments.

To train the Categorical Normalizing Flow with a linear flow encoding, use the argument `--encoding_num_flows` to specify the number of coupling layers in the linear flows. A value of 0 specifies a mixture model. The value used in the paper for linear flows was 4. For using variational encoding, add the keyword `--encoding_variational`.

To train a Normalizing Flow with Variational Dequantization on set modeling, use the following command:
```
python train.py --dataset shuffling \
                --max_iterations 100000 \
                --batch_size 1024 \
                --encoding_dequantization \
                --encoding_dim 1 \
                --encoding_num_flows 4 \
                --optimizer 4 \
                --learning_rate 7.5e-4 \
                --cluster
```

To train a Discrete Normalizing Flow, run the following command: 
```
python train.py --dataset shuffling \
                --max_iterations 100000 \
                --batch_size 1024 \
                --use_discrete \
                --optimizer 4 \
                --learning_rate 1e-4 \
                --cluster
```

## Evaluation

All models can be evaluated with the same command:
```
python eval.py --checkpoint_path path_to_folder
```
where `path_to_folder` should be replaced with the path to the actual folder with the checkpoints. The evaluation script applies the saved model to the test set and saves the results in the file `eval_metrics.json`.

## Pretrained models

Pretrained models for set modeling can be found [here](https://drive.google.com/drive/folders/14Ff1hzxbucgfrQ2haqW-Au7icvxwUJla?usp=sharing). After downloading the model folders, place them inside the `checkpoints` folder (for instance, `checkpoints/summation_CNF_mixture_model`. You can evaluate the pretrained models by running the evaluation script:

```
python eval.py --checkpoint_path checkpoints/summation_CNF_mixture_model/
```

## Results

### Set shuffling dataset

| Model | Bits per variable |
|---|---|
| Discrete NF | 3.87bpd |
| Variational Dequantization | 3.01bpd |
| CNF + Mixture model ([pretrained](https://drive.google.com/drive/folders/1qm14bWvvLtusQEDPZZKc4j3rW0a3y58X?usp=sharing)) | 2.78bpd |
| CNF + Linear flows | 2.78bpd |
| CNF + Variational encoding | 2.79bpd |
| Optimum | 2.77bpd |

### Set summation dataset

| Model | Bits per variable |
|---|---|
| Discrete NF | 2.51bpd |
| Variational Dequantization | 2.29bpd |
| CNF + Mixture model ([pretrained](https://drive.google.com/drive/folders/1CVjQqO6YElnO9LdXh-k7Q-HaJgZEyHPA?usp=sharing)) | 2.24bpd |
| CNF + Linear flows | 2.25bpd |
| CNF + Variational encoding | 2.25bpd |
| Optimum | 2.24bpd |
