import argparse

from torch.utils.data import dataset
from subgraph_counting import utils

def parse_count(parser, arg_str=None):
    cnt_parser = parser.add_argument_group()
    #utils.parse_optimizer(parser)

    cnt_parser.add_argument('--count_type', type=str,
                        help='model used to count the number of query')
    cnt_parser.add_argument('--embs_path', type=str,
                        help='file path of the embs')
    cnt_parser.add_argument('--batch_size', type=int,
                        help='batch size of the training epoch')
    cnt_parser.add_argument('--num_cpu', type=int,
                        help='number of cpu used for ground truth')
    cnt_parser.add_argument('--model_path', type=str,
                        help='path to save the model')
    '''
    cnt_parser.add_argument('--eval_interval', type=int,
                        help='how often to eval & save model during training')
    '''
    cnt_parser.add_argument('--val_size', type=int,
                        help='validation set size')
    cnt_parser.add_argument('--dataset', type=str,
                        help='name of the dataset used for training')
    cnt_parser.add_argument('--gpu', type=str,
                        help='input cuda:x to select which GPU to run on')
    cnt_parser.add_argument('--use_log', type=str,
                        help='use log2(count+1) as the ground truth of number of pattern')
    cnt_parser.add_argument('--use_norm', type=str,
                        help='use (log2(count+1)-mean)/std as the ground truth of number of pattern')
    # cnt_parser.add_argument('--use_hetero', type=str,
    #                     help='view graph as heterogeneous graph')
    cnt_parser.add_argument('--objective', type=str,
                        help='which obejective is the model going to learn, choosing from canonical/graphlet')
    cnt_parser.add_argument('--relabel_mode',
                        help='which mode to use to relabel node index, if do not wish to relabel, set to None')

    cnt_parser.set_defaults(
        # experiment
        count_type= 'motif',
        objective= 'canonical',
        embs_path= '',
        num_cpu= 1,
        # n_neighborhoods =64*900, # more query (6\7\8)
        # n_neighborhoods = 64*100,
        n_neighborhoods= 64*100,
        val_size=64*100,
        model_path = 'ckpt/tmp',
        use_log = True,
        use_norm = False,
        # use_hetero = True,
        # training
        batch_size= 64,
        weight_decay= 0.0,
        lr= 1e-3,
        num_epoch = 100,
        dataset= "syn",
        gpu="cuda",
        # relabel_mode="decreasing_degree"
        relabel_mode= None
    )

    #return cnt_parser.parse_args(arg_str)

def parse_gossip(parser, arg_str=None):
    gos_parser = parser.add_argument_group()
    #utils.parse_optimizer(parser)

    gos_parser.add_argument('--conv_type', type=str,
                        help='type of convolution')
    gos_parser.add_argument('--method_type', type=str,
                        help='type of embedding')
    gos_parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    gos_parser.add_argument('--n_layers', type=int,
                        help='Number of graph conv layers')
    gos_parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    # enc_parser.add_argument('--skip', type=str,
    #                     help='"all" or "last"')
    gos_parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    gos_parser.add_argument('--n_batches', type=int,
                        help='Number of training minibatches')
    gos_parser.add_argument('--margin', type=float,
                        help='margin for loss')
    gos_parser.add_argument('--dataset', type=str,
                        help='Dataset')
    gos_parser.add_argument('--test_set', type=str,
                        help='test set filename')
    gos_parser.add_argument('--eval_interval', type=int,
                        help='how often to eval during training')
    gos_parser.add_argument('--val_size', type=int,
                        help='validation set size')
    gos_parser.add_argument('--model_path', type=str,
                        help='path to save/load model')
    gos_parser.add_argument('--opt_scheduler', type=str,
                        help='scheduler name')
    gos_parser.add_argument('--node_anchored', action="store_true",
                        help='whether to use node anchoring in training')
    gos_parser.add_argument('--test', action="store_true")
    gos_parser.add_argument('--n_workers', type=int)
    gos_parser.add_argument('--tag', type=str,
        help='tag to identify the run')

    gos_parser.add_argument('--use_centrality', type=bool)

    gos_parser.set_defaults(
        conv_type='GOSSIP',
        n_layers=2,
        hidden_dim=64,
        dropout=0.0,
        n_workers=4,
        model_path="ckpt/general/gossip/pfconv_345_syn_qs_L2_ON_sage_main_model_epo150.pt",
        # model_path="ckpt/general/gossip/tmp.pt",
        lr= 1e-3,
        num_epoch = 50,
        weight_decay= 0.0,
        # n_neighborhoods= 64*100,
        n_neighborhoods= 64*100,
        use_log = True
    )

def parse_encoder(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    #utils.parse_optimizer(parser)

    enc_parser.add_argument('--conv_type', type=str,
                        help='type of convolution')
    enc_parser.add_argument('--method_type', type=str,
                        help='type of embedding')
    enc_parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    enc_parser.add_argument('--n_layers', type=int,
                        help='Number of graph conv layers')
    enc_parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    # enc_parser.add_argument('--skip', type=str,
    #                     help='"all" or "last"')
    enc_parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    enc_parser.add_argument('--n_batches', type=int,
                        help='Number of training minibatches')
    enc_parser.add_argument('--margin', type=float,
                        help='margin for loss')
    enc_parser.add_argument('--dataset', type=str,
                        help='Dataset')
    enc_parser.add_argument('--test_set', type=str,
                        help='test set filename')
    enc_parser.add_argument('--eval_interval', type=int,
                        help='how often to eval during training')
    enc_parser.add_argument('--val_size', type=int,
                        help='validation set size')
    enc_parser.add_argument('--model_path', type=str,
                        help='path to save/load model')
    enc_parser.add_argument('--opt_scheduler', type=str,
                        help='scheduler name')
    enc_parser.add_argument('--node_anchored', action="store_true",
                        help='whether to use node anchoring in training')
    enc_parser.add_argument('--test', action="store_true")
    enc_parser.add_argument('--n_workers', type=int)
    enc_parser.add_argument('--tag', type=str,
        help='tag to identify the run')

    enc_parser.add_argument('--use_centrality', type=bool)

    enc_parser.set_defaults(conv_type='SAGE',
                        method_type='order',
                        dataset='syn',
                        n_layers=8,
                        batch_size=64,
                        hidden_dim=64,
                        dropout=0.0,
                        n_batches=1000000,
                        opt='adam',   # opt_enc_parser
                        opt_scheduler='none',
                        opt_restart=100,
                        weight_decay=0.0,
                        lr=1e-4,
                        margin=0.1,
                        test_set='',
                        eval_interval=1000,
                        n_workers=4,
                        model_path="ckpt/degree_model.pt",
                        tag='',
                        val_size=4096,
                        node_anchored=True,
                        # skip="learnable",
                        use_centrality=False
                        )

    #return enc_parser.parse_args(arg_str)

