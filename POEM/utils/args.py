import argparse
from pathlib import Path
from sconf import Config
from domainbed import hparams_registry
from domainbed.lib import misc


def parse_argument():
    parser = argparse.ArgumentParser(description="Domain generalization")
    parser.add_argument("name", type=str)
    parser.add_argument("configs", nargs="*")
    parser.add_argument("--data_dir", type=str, default="datadir/")
    parser.add_argument("--dataset", type=str, default="EEG")
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument(
        "--trial_seed",
        type=int,
        default=0,
        help="Trial number (used for seeding split_dataset and random_hparams).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for everything else")
    parser.add_argument(
        "--steps", type=int, default=None, help="Number of steps. Default is dataset-dependent."
    )
    parser.add_argument(
        "--checkpoint_freq",
        type=int,
        default=None,
        help="Checkpoint every N steps. Default is dataset-dependent.",
    )
    parser.add_argument("--test_envs", type=int, nargs="+", default=None)  # sketch in PACS
    parser.add_argument("--holdout_fraction", type=float, default=0.2)
    parser.add_argument("--model_save", default=None, type=int, help="Model save start step")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--tb_freq", default=10)
    parser.add_argument("--debug", action="store_true", help="Run w/ debug mode")
    parser.add_argument("--show", action="store_true", help="Show args and hparams w/o run")
# parser.add_argument("--bool_angle", type=bool, default=False, help="bool angle loss")
    parser.add_argument("--bool_angle", action="store_true", help="bool angle loss")
    parser.add_argument(
        "--domain_swad", type=bool, default=True, help="bool swad to domain network"
    )
    parser.add_argument("--bool_swad", action="store_true", help="bool swad to networks")
    parser.add_argument("--bool_task", action="store_true", help="bool task to networks")
    parser.add_argument("--lr", type=float, default=0.0, help="learning rate")
    parser.add_argument(
        "--evalmode",
        default="fast",
        help="[fast, all]. if fast, ignore train_in datasets in evaluation time.",
    )
    parser.add_argument("--prebuild_loader", action="store_true", help="Pre-build eval loaders")
    args, left_argv = parser.parse_known_args()
    # setup hparams
    
    hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)

    keys = ["./utils/config.yaml"] + args.configs
    keys = [open(key, encoding="utf8") for key in keys]
    hparams = Config(*keys, default=hparams)
    hparams.argv_update(left_argv)
    hparams["bool_angle"] = args.bool_angle
    hparams["bool_task"] = args.bool_task
    if args.lr > 0:
        hparams["lr"] = args.lr

    # setup debug
    if args.debug:
        args.checkpoint_freq = 5
        args.steps = 10
        args.name += "_debug"

    timestamp = misc.timestamp()
    args.unique_name = f"{timestamp}_{args.name}"

    # path setup
    args.work_dir = Path(".")
    args.data_dir = Path(args.data_dir)

    args.out_root = args.work_dir / Path("log") / args.dataset
    args.out_dir = args.out_root / args.unique_name
    args.out_dir.mkdir(exist_ok=True, parents=True)

    return args, hparams

