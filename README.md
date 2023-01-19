# DeSCo: Towards Scalable Deep Subgraph Counting

This repository is the official implementation of DeSCo: Towards Scalable Deep Subgraph Counting.  

![DeSCo workflow](github_resource/workflow.png?raw=true "DeSCo workflow")

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, please first set the parameters in `subgraph_counting/config.py`, then run this command:

```train
python main.py
```

> Please refer to the Appendix for the detailed training parameters. The official configuration file of DeSCo will also be released shortly.

## Evaluation

The code comes with analysis methods in `subgraph_counting/workload.py`, which outputs the inference count of the model. Users should be able to get any desired metrics with these count easily.

```eval
python main.py
```

> The dataset used for evaluation is configured in `subgraph_counting/config.py`

## Pre-trained Models

Given the anonymous request during the revision period, we cannot provide the link to download the pre-trained model for now. It'll be available with the official release of our paper.
## Results

Our model achieves the following performance. The numbers are normalized MSEs of twenty-nine standard queries.

### MUTAG

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
 DeSCo | 1.1E-03 | 2.0E-03 | 1.0E-02

## Contributing

Welcome to use the code or contribute to the project upon the official release.
