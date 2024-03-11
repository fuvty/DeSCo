# DeSCo: Towards Generalizable and Scalable Deep Subgraph Counting

This repository is the official implementation of the [paper](https://arxiv.org/abs/2308.08198): *DeSCo: Towards Generalizable and Scalable Deep Subgraph Counting.* Please consider staring us if you find it interesting.

The paper is accepted by WSDM'24. You can view our [project page](https://fuvty.notion.site/Paper-accepted-by-WSDM-24-385bb3245c12484495f6b448f61304a3?pvs=4).

![DeSCo workflow](github_resource/workflow.png?raw=true "DeSCo workflow")

## Code Structure

`main.py` is the implementation of DeSCo.

`subgraph_counting` contains all the modules needed by python scripts.

>`baseline.py` is the implementation of two neural baselines (DIAMNet and LRP) that is compared with DeSCo in the paper.
`ablation_gnns.py` is used for the ablation study of the expressive power of SHMP. It implements other expressive GNNs.
`ablation_wo_canonical.py` is used for the ablation study of canonical partition. It implements DeSCo's neighborhood counting stage without canonical partition.

## Requirements

Python >= 3.9

To install requirements:

```setup
pip install -r requirements.txt
```

## Pre-trained Models

The neighborhood counting and gossip propagation model in our paper is trained on our synthetic dataset. Users can download our pre-trained model from [here](https://drive.google.com/drive/folders/1JsOepzJxUBLRsFM2O_-Zzd3APJPNrn-m?usp=drive_link)

## Evaluation

To evaluate the trained models on real-world datasets, please run the following command:

```eval
python main.py --test_dataset COX2 --neigh_checkpoint ckpt/{checkpoint_path}/neigh/{model_name}.ckpt --gossip_checkpoint ckpt/{checkpoint_path}/gossip/{model_name}.ckpt --test_gossip
```

The above command gives an example of evaluating the trained models on COX2. The path of checkpoints should be replaced by the real path of your trained model checkpoints.


The code comes with analysis methods in `subgraph_counting/workload.py`, which outputs the inference count of the model. Users should be able to get any desired metrics with these count easily.

## Train from Scratch

Alternatively, if you wish to train your own model instead of using our pre-trained version, here are the instructions you may need.

### Dataset

To benefit future research, we release the large synthetic dataset with subgraph count ground-truth that we used in our pre-trained model. Users can download the dataset zip file from [here](https://drive.google.com/drive/folders/1JsOepzJxUBLRsFM2O_-Zzd3APJPNrn-m?usp=drive_link) and move the unziped folder under ```DeSCo/data/``` to train from scratch.

### Code and configurations

If you desire to train with the official configuration of DeSCo, simply run this command:

```train
python main.py --train_dataset Syn_1827 --valid_dataset Syn_1827 --test_dataset MUTAG --train_neigh --train_gossip --test_gossip
```

To train the model(s) in the paper with other configurations, please specifies the parameters in the command.

The bool parameters `train_neigh`, `train_gossip`, and `test_gossip`, determine whether to train and to test the neighborhood counting and gossip propagation model.


> Please refer to the Appendix for the detailed training parameters.
<!-- The official configuration file of DeSCo will also be released shortly. -->

<!--

## Results

Our model achieves the following performance. The numbers are normalized MSEs of twenty-nine standard queries.

| Normed MSE |         | MUTAG   |         |         | COX2    |         |         | ENZYMES |         |         | IMDB-BINARY |         |          | MSRC-21  |          |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|-----------------|---------|----------|----------|----------|
| Query size | 3       | 4       | 5       | 3       | 4       | 5       | 3       | 4       | 5       | 3       | 4               | 5       | 3        | 4        | 5        |
| MOTIVO     | 2.9E-01 | 6.7E-01 | 1.2E+00 | 1.6E-01 | 3.4E-01 | 5.9E-01 | 1.6E-01 | 1.9E-01 | 3.0E-01 | 2.7E-02 | **3.9E-02**         | **5.0E-02** | 4.8E-02  | 7.2E-02  | 9.5E-02  |
| LRP        | 1.5E-01 | 2.7E-01 | 3.5E-01 | 1.4E-01 | 2.9E-02 | 1.1E-01 | 8.5E-01 | 5.4E-01 | 6.2E-01 | inf     | inf             | inf     | 2.4E+00  | 1.4E+00  | 1.1E+00  |
| DIAMNet    | 4.1E-01 | 5.6E-01 | 4.7E-01 | 1.1E+00 | 7.8E-01 | 7.2E-01 | 1.4E+00 | 1.1E+00 | 1.0E+00 | 1.1E+00 | 1.0E+00         | 1.0E+00 | 2.7E+00  | 1.6E+00  | 1.3E+00  |
| DMPNN      | 6.1E+02 | 6.6E+02 | 3.0E+02 | 2.6E+03 | 2.4E+03 | 3.0E+03 | 2.9E+03 | 1.4E+03 | 1.2E+03 | 2.1E+04 | 1.3E+02         | 1.4E+02 | 1.1E+04  | 1.3E+03  | 4.1E+02  |
| DeSCo      | **2.3E-03** | **8.4E-04** | **6.5E-03** | **6.9E-04** | **5.3E-04** | **5.4E-03** | **5.3E-03** | **5.7E-02** | **5.3E-02** | **8.7E-03** | 2.1E-01         | 4.5E-01 | **2.6E-03** | **3.9E-03** | **8.5E-02** |

-->

## Citation

If you find our work useful, please consider citing:

```bibtex
@inproceedings{fu2024desco,
  title={Desco: Towards generalizable and scalable deep subgraph counting},
  author={Fu, Tianyu and Wei, Chiyue and Wang, Yu and Ying, Rex},
  booktitle={Proceedings of the 17th ACM International Conference on Web Search and Data Mining},
  pages={218--227},
  year={2024}
}
```

## Contributing

Welcome to use the code or contribute to the project!
