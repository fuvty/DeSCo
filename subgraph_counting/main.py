import os
import sys

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import argparse
from ctypes.wintypes import INT
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pytorch_lightning as pl
import torch
import torch_geometric as pyg
import torch_geometric.transforms as T
from common import utils
from playground.lib.Anchor import SymmetricFactor
from subgraph_matching.config import parse_encoder
from torch_geometric.loader import DataLoader
from tqdm import tqdm
from tqdm.contrib import tzip

from subgraph_counting.config import parse_gossip
from subgraph_counting.data import (OTFSynCanonicalDataSource, count_canonical,
                                    count_graphlet, get_neigh_canonical,
                                    get_neigh_hetero, load_data,
                                    sample_graphlet, sample_neigh_canonical)
from subgraph_counting.lightning_model import (GossipCountingModel,
                                               NeighborhoodCountingModel)
from subgraph_counting.workload import NeighborhoodDataModule, Workload

if __name__ == "__main__":
    # define dataset
    train_dataset_name = 'ENZYMES'
    test_dataset_name = 'COX2'

    # define workload
    # TODO: add valid set mask support
    train_dataset = load_data(train_dataset_name)
    test_dataset = load_data(test_dataset_name)
    
    train_workload = Workload(train_dataset, 'data/'+train_dataset_name, hetero_graph=True)
    test_workload = Workload(test_dataset, 'data/'+test_dataset_name, hetero_graph=True)

    # add this line when you need to compute ground truth
    query_ids = [6,7]
    train_workload.compute_groundtruth(query_ids= query_ids)
    test_workload.compute_groundtruth(query_ids= query_ids)

    # generate pipeline dataset
    train_workload.generate_pipeline_datasets(depth_neigh=4)
    test_workload.generate_pipeline_datasets(depth_neigh=4)
    
    # begin canonical count
    neighborhood_train_dataloader = DataLoader(train_workload.neighborhood_dataset, batch_size=64, shuffle=False, num_workers=4)
    neighborhood_test_dataloader = DataLoader(test_workload.neighborhood_dataset, batch_size=64, shuffle=False, num_workers=4)
    # neighborhoood_dataloader = NeighborhoodDataModule(train_dataset= train_workload.neighborhood_dataset, test_dataset= test_workload.neighborhood_dataset, val_dataset= test_workload.neighborhood_dataset, batch_size=64, shuffle=False)
    # lightningDataLoader = pyg.data.LightningDataset(train_dataset= train_workload.neighborhood_dataset, val_dataset= test_workload.neighborhood_dataset,test_dataset= test_workload.neighborhood_dataset, batch_size=64)

    # canonical count training
    # define neighborhood counting model
    parser = argparse.ArgumentParser(description='Order embedding arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    model_args = parser.parse_args()
    model_args.use_hetero = True

    neighborhood_model = NeighborhoodCountingModel(input_dim=1, hidden_dim=64, args=model_args)
    neighborhood_model.to_hetero(tconv_target= False, tconv_query= False)

    neighborhood_model.set_queries(query_ids)

    neighborhood_trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices=[4])
    # neighborhood_trainer.fit(neighborhood_model, datamodule=neighborhoood_dataloader)
    neighborhood_trainer.fit(model=neighborhood_model, train_dataloaders=neighborhood_train_dataloader, val_dataloaders=neighborhood_test_dataloader)

    neighborhood_trainer.test(neighborhood_model, dataloaders=neighborhood_test_dataloader) 

    # canonical count inference
    # checkpoint = 'some/path/to/checkpoint'
    # model = NeighborhoodCountingModel.load_from_checkpoint(checkpoint)
    neighborhood_count_train = torch.cat([neighborhood_model.graph_to_count(g) for g in neighborhood_train_dataloader], dim=0) # size = (#neighborhood, #queries)
    neighborhood_count_test = torch.cat([neighborhood_model.graph_to_count(g) for g in neighborhood_test_dataloader], dim=0)

    # apply neighborhood count output to gossip dataset
    train_workload.apply_neighborhood_count(neighborhood_count_train)
    test_workload.apply_neighborhood_count(neighborhood_count_test)

    # define gossip counting model
    parser = argparse.ArgumentParser(description='Order embedding arguments')
    utils.parse_optimizer(parser)
    parse_gossip(parser)
    args_gossip = parser.parse_args() 

    input_dim = 1
    args_gossip.use_hetero = False
    gossip_model = GossipCountingModel(input_dim, 64, args_gossip, emb_channels= 64)

    gossip_model.set_query_emb(neighborhood_model.get_query_emb())

    # gossip training
    gossip_train_dataloader = DataLoader(train_workload.gossip_dataset)
    gossip_test_dataloader = DataLoader(test_workload.gossip_dataset)

    gossip_trainer = pl.Trainer(max_epochs=100, accelerator="gpu", devices=[4])
    gossip_trainer.fit(model=gossip_model, train_dataloaders=gossip_train_dataloader, val_dataloaders=gossip_test_dataloader)

    # gossip inference
    gossip_trainer.test(gossip_model, dataloaders=gossip_test_dataloader)
    