import argparse

from torch.utils.data import dataset
from subgraph_counting import utils


def parse_neighborhood(parser, arg_str=None) -> list[argparse._StoreAction]:
    enc_parser = parser.add_argument_group("neighborhood counting model arguments")

    # general model settings
    enc_parser.add_argument("--neigh_conv_type", type=str, help="type of convolution")
    enc_parser.add_argument(
        "--neigh_layer_num", type=int, help="Number of graph conv layers"
    )
    enc_parser.add_argument("--neigh_input_dim", type=int, help="Training input size")
    enc_parser.add_argument("--neigh_hidden_dim", type=int, help="Training hidden size")
    enc_parser.add_argument("--neigh_dropout", type=float, help="Dropout rate")
    enc_parser.add_argument(
        "--neigh_model_path", type=str, help="path to save/load model"
    )
    enc_parser.add_argument(
        "--neigh_epoch_num", type=int, help="number of training epochs"
    )
    enc_parser.add_argument(
        "--neigh_batch_size", type=int, help="number of training epochs"
    )

    # DeSCo sepecific settings
    enc_parser.add_argument(
        "--depth", type=int, help="depth of the canonical neighborhood"
    )
    enc_parser.add_argument(
        "--use_hetero", action="store_true", help="whether to use heterogeneous GNNs"
    )
    enc_parser.add_argument(
        "-t",
        "--use_tconv",
        action="store_true",
        help="whether to use triangle convolution (a case of SHMP)",
    )
    enc_parser.add_argument(
        "-z",
        "--zero_node_feat",
        action="store_true",
        help="whether to zero out existing node features",
    )
    enc_parser.add_argument(
        "-f",
        "--use_node_feature",
        action="store_true",
        help="whether to use node features",
    )

    # optimizer settings
    enc_parser.add_argument("--neigh_weight_decay", type=float, help="weight decay")
    enc_parser.add_argument("--neigh_lr", type=float, help="learning rate")
    enc_parser.add_argument(
        "--neigh_tune_lr", action="store_true", help="auto tune learning rate"
    )
    enc_parser.add_argument(
        "--neigh_tune_bs", action="store_true", help="auto tune batch size"
    )

    enc_parser.set_defaults(
        neigh_conv_type="SAGE",
        neigh_layer_num=8,
        neigh_input_dim=1,
        neigh_hidden_dim=64,
        neigh_dropout=0.0,
        neigh_model_path="ckpt/kdd23/neighborhood",
        neigh_epoch_num=300,
        neigh_batch_size=512,
        depth=4,
        use_hetero=True,
        use_tconv=True,
        zero_node_feat=True,
        use_node_feature=False,
        neigh_weight_decay=0.0,
        neigh_lr=1e-4,
    )

    # TODO: add the following arguments
    # opt_enc_parser
    # opt="adam",
    # opt_scheduler="none",
    # opt_restart=100,

    # return the keys of the parser
    return enc_parser._group_actions


def parse_gossip(parser, arg_str=None) -> list[argparse._StoreAction]:
    gos_parser = parser.add_argument_group("gossip counting model arguments")

    # general model settings
    gos_parser.add_argument("--gossip_conv_type", type=str, help="type of convolution")
    gos_parser.add_argument(
        "--gossip_layer_num", type=int, help="Number of graph conv layers"
    )
    gos_parser.add_argument(
        "--gossip_hidden_dim", type=int, help="Training hidden size"
    )
    gos_parser.add_argument("--gossip_dropout", type=float, help="Dropout rate")
    gos_parser.add_argument(
        "--gossip_model_path", type=str, help="path to save/load model"
    )
    gos_parser.add_argument(
        "--gossip_epoch_num", type=int, help="number of training epochs"
    )
    gos_parser.add_argument(
        "--gossip_batch_size", type=int, help="number of training epochs"
    )

    # optimizer settings
    gos_parser.add_argument(
        "--gossip_lr",
        type=float,
        help="learning rate, if None, use hyperparameter search",
    )
    gos_parser.add_argument("--weight_decay", type=float, help="weight decay")
    gos_parser.add_argument(
        "--gossip_tune_lr", action="store_true", help="auto tune learning rate"
    )
    gos_parser.add_argument(
        "--gossip_tune_bs", action="store_true", help="auto tune batch size"
    )

    gos_parser.set_defaults(
        gossip_conv_type="GOSSIP",
        gossip_layer_num=2,
        gossip_hidden_dim=64,
        gossip_dropout=0.01,
        gossip_model_path="ckpt/kdd23/gossip",
        gossip_epoch_num=30,
        gossip_batch_size=256,
        gossip_lr=1e-3,
        weight_decay=0.0,
    )

    # return the namespace of the parser
    return gos_parser._group_actions


def parse_optimizer(parser) -> list[argparse._StoreAction]:
    opt_parser = parser.add_argument_group("optimizer arguments")

    opt_parser.add_argument(
        "--train_dataset", type=str, help="name of the training dataset"
    )
    opt_parser.add_argument("--test_dataset", type=str, help="name of the test dataset")
    opt_parser.add_argument(
        "--gpu", nargs="+", type=int, help="the id of gpus to use, support multi-gpu"
    )

    opt_parser.add_argument("--num_cpu", type=int, help="number of cpu to use")
    opt_parser.add_argument("--output_dir", type=str, help="path to save raw output")
    opt_parser.add_argument(
        "--neigh_checkpoint", type=str, help="path to load neighborhood counting model"
    )
    opt_parser.add_argument(
        "--gossip_checkpoint", type=str, help="path to load gossip counting model"
    )
    opt_parser.add_argument(
        "--train_neigh",
        action="store_true",
        help="whether to train neighborhood counting model",
    )
    opt_parser.add_argument(
        "--train_gossip",
        action="store_true",
        help="whether to train gossip counting model",
    )
    opt_parser.add_argument(
        "--test_gossip",
        action="store_true",
        help="whether to test gossip counting model",
    )

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
        train_dataset="syn_128",
        test_dataset="ENZYMES",
        gpu=0,
        num_cpu=8,
        output_dir=None,
        neigh_checkpoint=None,
        gossip_checkpoint=None,
        train_neigh=False,
        train_gossip=False,
        test_gossip=True,
    )

    # return the keys of the parser
    return opt_parser._group_actions
