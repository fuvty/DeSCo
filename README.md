# DeSCo: Towards Scalable Deep Subgraph Counting

This repository is the official implementation of DeSCo: Towards Scalable Deep Subgraph Counting.  

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

## Training

If you desire to train with the official configuration of DeSCo, straightly run this command:

```train
python main.py
```

To train the model(s) in the paper with other configurations, please first change the parameters in the last part of `subgraph_counting/config.py` as shown below, then run the command above.

```
    opt_parser.set_defaults(
        train_dataset="Syn_1827",
        valid_dataset="Syn_1827",
        test_dataset="MUTAG",
        gpu=0,
        num_cpu=8,
        output_dir=None,
        neigh_checkpoint=None,
        gossip_checkpoint=None,
        train_neigh=True,
        train_gossip=True,
        test_gossip=True,
    )
```

Set the bool parameters train_neigh, train_gossip, and test_gossip, to determine whether to train and to test the neighborhood counting and gossip propagation model.


> Please refer to the Appendix for the detailed training parameters.
<!-- The official configuration file of DeSCo will also be released shortly. -->

## Evaluation

To evaluate the trained models on real-world datasets, please set the parameters in the last part of `subgraph_counting/config.py` as shown above and then run the command:

```eval
python main.py
```

Here is an example for setting the parameters. You should change the following parameters in the last part of `subgraph_counting/config.py` and remain other parameters unchanged. The path of checkpoints should be replaced by the real path of your trained model checkpoints.

```
test_dataset="COX2"
neigh_checkpoint="ckpt/{checkpoint_path}/neigh/{model_name}.ckpt"
gossip_checkpoint="ckpt/{checkpoint_path}/gossip/{model_name}.ckpt"
train_neigh=False
train_gossip=False,
```


The code comes with analysis methods in `subgraph_counting/workload.py`, which outputs the inference count of the model. Users should be able to get any desired metrics with these count easily.

## Pre-trained Models

Given the anonymous request during the revision period, we cannot provide the link to download the pre-trained model for now. It'll be available with the official release of our paper.
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



<!-- ### MUTAG

 Query-Size   | 3       | 4       | 5  
------------|:-------:|:-------:|:-------:
 DeSCo | 7.3E-05 | 5.2E-04 | 1.1E-02

### COX2

 Query-Size   | 3       | 4       | 5  
------------|:-------:|:-------:|:-------:
 DeSCo | 2.3E-05 | 9.5E-05 | 7.2E-03

### ENZYMES

 Query-Size   | 3       | 4       | 5  
------------|:-------:|:-------:|:-------:
 DeSCo | 1.1E-03 | 2.0E-03 | 1.0E-02 -->

## Contributing

Welcome to use the code or contribute to the project upon the official release.
