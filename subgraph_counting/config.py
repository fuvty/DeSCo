import argparse

from torch.utils.data import dataset
from subgraph_counting import utils


def parse_encoder(parser, arg_str=None):
    enc_parser = parser.add_argument_group()
    # utils.parse_optimizer(parser)

    enc_parser.add_argument("--conv_type", type=str, help="type of convolution")
    enc_parser.add_argument("--method_type", type=str, help="type of embedding")
    enc_parser.add_argument("--batch_size", type=int, help="Training batch size")
    enc_parser.add_argument("--n_layers", type=int, help="Number of graph conv layers")
    enc_parser.add_argument("--hidden_dim", type=int, help="Training hidden size")
    # enc_parser.add_argument('--skip', type=str,
    #                     help='"all" or "last"')
    enc_parser.add_argument("--dropout", type=float, help="Dropout rate")
    enc_parser.add_argument(
        "--n_batches", type=int, help="Number of training minibatches"
    )
    enc_parser.add_argument("--margin", type=float, help="margin for loss")
    enc_parser.add_argument("--dataset", type=str, help="Dataset")
    enc_parser.add_argument("--test_set", type=str, help="test set filename")
    enc_parser.add_argument(
        "--eval_interval", type=int, help="how often to eval during training"
    )
    enc_parser.add_argument("--val_size", type=int, help="validation set size")
    enc_parser.add_argument("--model_path", type=str, help="path to save/load model")
    enc_parser.add_argument("--opt_scheduler", type=str, help="scheduler name")
    enc_parser.add_argument(
        "--node_anchored",
        action="store_true",
        help="whether to use node anchoring in training",
    )
    enc_parser.add_argument("--test", action="store_true")
    enc_parser.add_argument("--n_workers", type=int)
    enc_parser.add_argument("--tag", type=str, help="tag to identify the run")

    enc_parser.add_argument("--use_centrality", type=bool)

    enc_parser.set_defaults(
        conv_type="SAGE",
        method_type="order",
        dataset="ENZYMES",
        n_layers=8,
        batch_size=64,
        hidden_dim=64,
        dropout=0.0,
        n_batches=1000000,
        opt="adam",  # opt_enc_parser
        opt_scheduler="none",
        opt_restart=100,
        weight_decay=0.0,
        lr=1e-3,
        margin=0.1,
        test_set="",
        eval_interval=1000,
        n_workers=4,
        model_path="ckpt/degree_model.pt",
        tag="",
        val_size=4096,
        node_anchored=True,
        # skip="learnable",
        use_centrality=False,
    )


def parse_count(parser, arg_str=None):
    cnt_parser = parser.add_argument_group()
    # utils.parse_optimizer(parser)

    cnt_parser.add_argument(
        "--count_type", type=str, help="model used to count the number of query"
    )
    cnt_parser.add_argument("--embs_path", type=str, help="file path of the embs")
    cnt_parser.add_argument(
        "--batch_size", type=int, help="batch size of the training epoch"
    )
    cnt_parser.add_argument(
        "--num_cpu", type=int, help="number of cpu used for ground truth"
    )
    cnt_parser.add_argument("--model_path", type=str, help="path to save the model")
    """
    cnt_parser.add_argument('--eval_interval', type=int,
                        help='how often to eval & save model during training')
    """
    cnt_parser.add_argument("--val_size", type=int, help="validation set size")
    cnt_parser.add_argument(
        "--dataset", type=str, help="name of the dataset used for training"
    )
    cnt_parser.add_argument(
        "--gpu", type=str, help="input cuda:x to select which GPU to run on"
    )
    cnt_parser.add_argument(
        "--use_log",
        type=str,
        help="use log2(count+1) as the ground truth of number of pattern",
    )
    cnt_parser.add_argument(
        "--use_norm",
        type=str,
        help="use (log2(count+1)-mean)/std as the ground truth of number of pattern",
    )
    # cnt_parser.add_argument('--use_hetero', type=str,
    #                     help='view graph as heterogeneous graph')
    cnt_parser.add_argument(
        "--objective",
        type=str,
        help="which obejective is the model going to learn, choosing from canonical/graphlet",
    )
    cnt_parser.add_argument(
        "--relabel_mode",
        help="which mode to use to relabel node index, if do not wish to relabel, set to None",
    )

    cnt_parser.set_defaults(
        # experiment
        # count_type= 'multitask',
        count_type="motif",
        objective="canonical",
        embs_path="",
        num_cpu=1,
        # n_neighborhoods =64*900, # more query (6\7\8)
        # n_neighborhoods = 64*100,
        n_neighborhoods=2048,
        val_size=64 * 100,
        # val_size=64,
        model_path="ckpt/general/baseline/DIAMNet/GIN_DIAMNet_345_syn_1827",
        # ckpt/general/motif/sage_345_synXL_qs_triTQ_hetero_epo300.pt
        # model_path="ckpt/general/baseline/LRP/LRP_345_synXL_qs",
        # model_path="ckpt/general/baseline/DIAMNet/SAGE_DIAMNet_345_syn_qs",
        # model_path = 'ckpt/general/trans/large_query/LRP_345_syn2048_qs_FROM_LRP_345_synXL_qs_epo50.pt',
        use_log=True,
        use_norm=False,
        # use_hetero = True,
        # training
        batch_size=4,
        weight_decay=0.0,
        lr=1e-3,
        num_epoch=300,
        dataset="Syn",
        gpu="cuda:0",
        # relabel_mode="decreasing_degree"
        relabel_mode=None,
    )


def parse_optimizer_baseline(parser):
    opt_parser = parser.add_argument_group()
    opt_parser.add_argument("--opt", dest="opt", type=str, help="Type of optimizer")
    opt_parser.add_argument(
        "--opt-scheduler",
        dest="opt_scheduler",
        type=str,
        help="Type of optimizer scheduler. By default none",
    )
    opt_parser.add_argument(
        "--opt-restart",
        dest="opt_restart",
        type=int,
        help="Number of epochs before restart (by default set to 0 which means no restart)",
    )
    opt_parser.add_argument(
        "--opt-decay-step",
        dest="opt_decay_step",
        type=int,
        help="Number of epochs before decay",
    )
    opt_parser.add_argument(
        "--opt-decay-rate",
        dest="opt_decay_rate",
        type=float,
        help="Learning rate decay ratio",
    )
    opt_parser.add_argument("--lr", dest="lr", type=float, help="Learning rate.")
    opt_parser.add_argument(
        "--clip", dest="clip", type=float, help="Gradient clipping."
    )
    opt_parser.add_argument(
        "--weight_decay", type=float, help="Optimizer weight decay."
    )


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
    enc_parser.add_argument(
        "--use_canonical",
        action="store_true",
        help="whether to use canonical partition",
    )

    enc_parser.set_defaults(
        neigh_conv_type="SAGE",
        neigh_layer_num=8,
        neigh_input_dim=1,
        neigh_hidden_dim=64,
        neigh_dropout=0.0,
        neigh_model_path="ckpt/DeSCo/Syn_1827/neigh",
        neigh_epoch_num=300,
        neigh_batch_size=512,
        depth=4,
        use_hetero=True,
        use_tconv=True,
        use_node_feature=False,
        zero_node_feat=False,
        neigh_weight_decay=0.0,
        neigh_lr=1e-4,
        use_canonical=True,
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
        gossip_model_path="ckpt/DeSCo/Syn_1827/gossip",
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
    opt_parser.add_argument(
        "--valid_dataset", type=str, help="name of the validation dataset"
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
        train_dataset="Syn_1827",
        valid_dataset="Syn_1827",
        test_dataset="MUTAG",
        gpu=0,
        num_cpu=8,
        output_dir=None,
        # neigh_checkpoint="ckpt/kdd23/Syn_1827/neighborhood/lightning_logs/version_0/checkpoints/epoch=299-step=35100.ckpt",
        # gossip_checkpoint="ckpt/kdd23/Syn_1827/gossip/lightning_logs/version_2/checkpoints/epoch=99-step=1500.ckpt",
        neigh_checkpoint=None,
        gossip_checkpoint=None,
        train_neigh=False,
        train_gossip=False,
        test_gossip=False,
    )

    # return the keys of the parser
    return opt_parser._group_actions
