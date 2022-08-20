import os
import sys

from subgraph_counting.transforms import ToTconvHetero, ZeroNodeFeat

parentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(parentdir)

import argparse
import datetime
from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch_geometric as pyg
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader

from subgraph_counting.config import (parse_gossip, parse_neighborhood,
                                      parse_optimizer)
from subgraph_counting.data import gen_query_ids, load_data
from subgraph_counting.lightning_model import (GossipCountingModel,
                                               NeighborhoodCountingModel)
from subgraph_counting.workload import Workload


def main(args_neighborhood, args_gossip, args_opt, train_neighborhood: bool = True, train_gossip: bool = True, neighborhood_checkpoint = None, gossip_checkpoint = None):
    '''
    train the model and test accorrding to the config
    '''

    # define queries
    query_ids = gen_query_ids(query_size= [3,4,5])
    print('use queries with atlas ids:', query_ids)

    # define pre-transform
    zero_node_feat_transform = ZeroNodeFeat() if args_neighborhood.zero_node_feat else None

    # neighborhood transformation
    neighborhood_transform = ToTconvHetero() if args_neighborhood.use_tconv else None
    assert args_neighborhood.use_hetero if args_neighborhood.use_tconv else True

    # define training workload
    if train_neighborhood or train_gossip:
        train_dataset_name = args_opt.train_dataset
        train_dataset = load_data(train_dataset_name, transform=zero_node_feat_transform) # TODO: add valid set mask support
        train_workload = Workload(train_dataset, 'data/'+train_dataset_name, hetero_graph=True)
        if train_workload.exist_groundtruth(query_ids=query_ids):
            train_workload.canonical_count_truth = train_workload.load_groundtruth(query_ids=query_ids)
        else:
            train_workload.canonical_count_truth = train_workload.compute_groundtruth(query_ids= query_ids, save_to_file= True)
        train_workload.generate_pipeline_datasets(depth_neigh=args_neighborhood.depth, neighborhood_transform=neighborhood_transform) # generate pipeline dataset, including neighborhood dataset and gossip dataset

    # define testing workload
    test_dataset_name = args_opt.test_dataset
    test_dataset = load_data(test_dataset_name, transform=zero_node_feat_transform)
    test_workload = Workload(test_dataset, 'data/'+test_dataset_name, hetero_graph=True)
    if test_workload.exist_groundtruth(query_ids=query_ids): 
        test_workload.canonical_count_truth = test_workload.load_groundtruth(query_ids=query_ids)
    else:
        test_workload.canonical_count_truth = test_workload.compute_groundtruth(query_ids= query_ids, save_to_file= True) # compute ground truth if not any
    test_workload.generate_pipeline_datasets(depth_neigh=args_neighborhood.depth, neighborhood_transform=neighborhood_transform) # generate pipeline dataset, including neighborhood dataset and gossip dataset


    ########### begin neighborhood counting ###########
    # define neighborhood counting dataset
    if train_neighborhood or train_gossip:
        neighborhood_train_dataloader = DataLoader(train_workload.neighborhood_dataset, batch_size=args_opt.neighborhood_batch_size, shuffle=False, num_workers=4)
    neighborhood_test_dataloader = DataLoader(test_workload.neighborhood_dataset, batch_size=args_opt.neighborhood_batch_size, shuffle=False, num_workers=4)

    # define neighborhood counting model
    neighborhood_trainer = pl.Trainer(max_epochs=args_neighborhood.num_epoch, accelerator="gpu", devices=[args_opt.gpu], default_root_dir=args_neighborhood.model_path)

    if train_neighborhood:
        neighborhood_model = NeighborhoodCountingModel(input_dim=1, hidden_dim=args_neighborhood.hidden_dim, args=args_neighborhood)
        neighborhood_model.to_hetero(tconv_target= args_neighborhood.use_tconv, tconv_query= args_neighborhood.use_tconv)
    else:
        assert neighborhood_checkpoint is not None
        neighborhood_model = NeighborhoodCountingModel.load_from_checkpoint(neighborhood_checkpoint) # to hetero is automatically done upon loading 
    neighborhood_model.set_queries(query_ids, transform=neighborhood_transform)

    # train neighborhood model
    if train_neighborhood:
        neighborhood_trainer.fit(model=neighborhood_model, train_dataloaders=neighborhood_train_dataloader, val_dataloaders=neighborhood_test_dataloader)

    # test neighborhood counting model
    neighborhood_trainer.test(neighborhood_model, dataloaders=neighborhood_test_dataloader) 
    

    ########### begin gossip counting ###########
    # apply neighborhood count output to gossip dataset
    if train_gossip:
        neighborhood_count_train = torch.cat([neighborhood_model.graph_to_count(g) for g in neighborhood_train_dataloader], dim=0) # size = (#neighborhood, #queries)
        train_workload.apply_neighborhood_count(neighborhood_count_train)
    neighborhood_count_test = torch.cat([neighborhood_model.graph_to_count(g) for g in neighborhood_test_dataloader], dim=0)
    test_workload.apply_neighborhood_count(neighborhood_count_test)

    # define gossip counting dataset
    gossip_test_dataloader = DataLoader(test_workload.gossip_dataset)

    # define gossip counting model
    input_dim = 1
    args_gossip.use_hetero = False
    if train_gossip:
        gossip_model = GossipCountingModel(input_dim, args_gossip.hidden_dim, args_gossip, emb_channels= args_neighborhood.hidden_dim)
    else:
        assert gossip_checkpoint is not None
        gossip_model = GossipCountingModel.load_from_checkpoint(gossip_checkpoint)
    gossip_model.set_query_emb(neighborhood_model.get_query_emb())

    gossip_trainer = pl.Trainer(max_epochs=args_gossip.num_epoch, accelerator="gpu", devices=[args_opt.gpu], default_root_dir=args_gossip.model_path)

    # train gossip model
    if train_gossip:
        gossip_train_dataloader = DataLoader(train_workload.gossip_dataset)
        gossip_trainer.fit(model=gossip_model, train_dataloaders=gossip_train_dataloader, val_dataloaders=gossip_test_dataloader)

    gossip_trainer.test(gossip_model, dataloaders=gossip_test_dataloader)
    
    ########### output graphlet results ###########
    time = datetime.datetime.now().strftime('%Y%m%d_%H:%M:%S')

    neighborhood_count_test = torch.cat([neighborhood_model.graph_to_count(g) for g in neighborhood_test_dataloader], dim=0)
    graphlet_neighborhood_count_test = test_workload.neighborhood_dataset.aggregate_neighborhood_count(neighborhood_count_test) # user can get the graphlet count of each graph in this way
    pd.DataFrame(torch.round(graphlet_neighborhood_count_test).detach().cpu().numpy()).to_csv(os.path.join('results/raw_results', 'neighborhood_{}_{}_{}.csv'.format(args_neighborhood.conv_type, args_opt.test_dataset, time))) # save the inferenced results to csv file

    gossip_count_test = torch.cat([gossip_model.graph_to_count(g) for g in gossip_test_dataloader], dim=0)
    graphlet_gossip_count_test = test_workload.gossip_dataset.aggregate_neighborhood_count(gossip_count_test) # user can get the graphlet count of each graph in this way
    pd.DataFrame(torch.round(graphlet_gossip_count_test).detach().cpu().numpy()).to_csv(os.path.join('results/raw_results', 'gossip_{}_{}_{}.csv'.format(args_gossip.conv_type, args_opt.test_dataset, time))) # save the inferenced results to csv file

    print('done')


if __name__ == "__main__":
    # load parameters
    parser = argparse.ArgumentParser(description='Neighborhood embedding arguments')
    parse_optimizer(parser)
    args_opt = parser.parse_args()

    # define neighborhood counting model
    parse_neighborhood(parser)
    args_neighborhood = parser.parse_args()

    # define gossip counting model
    parser = argparse.ArgumentParser(description='Neighborhood embedding arguments') # TODO: cannot use parse_gossip in command line, conflict with args_neighborhood
    parse_gossip(parser)
    args_gossip = parser.parse_args()   

    # debug; TODO: the following restrictions are added because of the limited implemented senarios
    assert args_neighborhood.use_hetero == True

    neighborhood_checkpoint = 'ckpt/neighborhood/lightning_logs/version_0/checkpoints/epoch=299-step=655800.ckpt'
    gossip_checkpoint = 'test/gossip/lightning_logs/version_4/checkpoints/epoch=0-step=600.ckpt'

    main(args_neighborhood, args_gossip, args_opt, train_neighborhood= False, train_gossip= False, neighborhood_checkpoint= neighborhood_checkpoint, gossip_checkpoint= gossip_checkpoint) 
