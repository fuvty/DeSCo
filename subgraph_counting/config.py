import argparse

from torch.utils.data import dataset
from subgraph_counting import utils

def parse_neighborhood(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    #utils.parse_optimizer(parser)

    enc_parser.add_argument('--conv_type', type=str,
                        help='type of convolution')
    enc_parser.add_argument('--batch_size', type=int,
                        help='Training batch size')
    enc_parser.add_argument('--n_layers', type=int,
                        help='Number of graph conv layers')
    enc_parser.add_argument('--hidden_dim', type=int,
                        help='Training hidden size')
    enc_parser.add_argument('--dropout', type=float,
                        help='Dropout rate')
    enc_parser.add_argument('--n_batches', type=int,
                        help='Number of training minibatches')
    enc_parser.add_argument('--model_path', type=str,
                        help='path to save/load model')
    enc_parser.add_argument('--opt_scheduler', type=str,
                        help='scheduler name')
    enc_parser.add_argument('--node_anchored', action="store_true",
                        help='whether to use node anchoring in training')
    enc_parser.add_argument('--num_epoch', type=int,
                        help='number of epochs')
    enc_parser.add_argument('--depth', type=int,
                        help='depth of the neighborhood')
    
    enc_parser.set_defaults(
        conv_type='SAGE',
        n_layers=8,
        hidden_dim=64,
        use_hetero=True,
        depth=4,
        
        opt='adam',   # opt_enc_parser
        opt_scheduler='none',
        opt_restart=100,
        weight_decay=0.0,
        lr=1e-4,
        num_epoch = 300,
        
        n_workers=4,
        model_path="ckpt/neighborhood",
        dropout=0.0
    )


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
        model_path="ckpt/gossip",
        # model_path="ckpt/general/gossip/tmp.pt",
        lr= 1e-3,
        num_epoch = 50,
        weight_decay= 0.0,
        # n_neighborhoods= 64*100,
        n_neighborhoods= 64*100,
        use_log = True
    )

def parse_optimizer(parser):
    opt_parser = parser.add_argument_group()

    opt_parser.add_argument('--train_dataset', type=str,
                        help='name of the training dataset')
    opt_parser.add_argument('--test_dataset', type=str,
                        help='name of the test dataset')
    opt_parser.add_argument('--gpu', type=int,
                        help='the id of gpu to use')


#     opt_parser.add_argument('--opt', dest='opt', type=str,
#             help='Type of optimizer')
#     opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
#             help='Type of optimizer scheduler. By default none')
#     opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
#             help='Number of epochs before restart (by default set to 0 which means no restart)')
#     opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
#             help='Number of epochs before decay')
#     opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
#             help='Learning rate decay ratio')
#     opt_parser.add_argument('--lr', dest='lr', type=float,
#             help='Learning rate.')
#     opt_parser.add_argument('--clip', dest='clip', type=float,
#             help='Gradient clipping.')
#     opt_parser.add_argument('--weight_decay', type=float,
#             help='Optimizer weight decay.')

    opt_parser.set_defaults(
        train_dataset='syn_6400',
        test_dataset='ENZYMES',
        gpu=0
    )