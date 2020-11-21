import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from klue_baseline import KLUE_TASKS
from klue_baseline.utils import Command, LoggingCallback

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def add_general_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help=f"Run one of the task in {list(KLUE_TASKS.keys())}",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--gpus",
        default=None,
        nargs="+",
        type=int,
        help="Select specific GPU allocated for this, it is by default [] meaning none",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision instead of 32-bit",
    )
    parser.add_argument(
        "--num_sanity_val_steps",
        type=int,
        default=2,
        help="Sanity check validation steps (default 2 steps)",
    )
    parser.add_argument("--n_tpu_cores", dest="tpu_cores", type=int)
    parser.add_argument("--max_grad_norm", dest="gradient_clip_val", default=1.0, type=float, help="Max gradient norm")
    parser.add_argument(
        "--gradient_accumulation_steps",
        dest="accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--metric_key", type=str, default="loss", help="The name of monitoring metric")
    parser.add_argument(
        "--patience",
        default=5,
        type=int,
        help="The number of validation epochs with no improvement after which training will be stopped.",
    )
    parser.add_argument(
        "--early_stopping_mode",
        choices=["min", "max"],
        default="max",
        type=str,
        help="In min mode, training will stop when the quantity monitored has stopped decreasing; in max mode it will stop when the quantity monitored has stopped increasing;",
    )
    return parser


def make_klue_trainer(
    args: argparse.Namespace,
    extra_callbacks: List = [],
    checkpoint_callback: Optional[pl.Callback] = None,
    logging_callback: Optional[pl.Callback] = None,
    **extra_train_kwargs,
) -> pl.Trainer:
    pl.seed_everything(args.seed)

    # Logging
    csv_logger = CSVLogger(args.output_dir, name=args.task)
    args.output_dir = csv_logger.log_dir

    if logging_callback is None:
        logging_callback = LoggingCallback()

    # add custom checkpoints
    metric_key = f"valid/{args.metric_key}"
    if checkpoint_callback is None:
        filename_for_metric = "{" + metric_key + ":.2f}"

        checkpoint_callback = ModelCheckpoint(
            dirpath=Path(args.output_dir).joinpath("checkpoint"),
            monitor=metric_key,
            filename="{epoch:02d}-{step}=" + filename_for_metric,
            save_top_k=1,
            mode="max",
        )
    early_stopping_callback = EarlyStopping(monitor=metric_key, patience=args.patience, mode=args.early_stopping_mode)
    extra_callbacks.append(early_stopping_callback)

    train_params: Dict[str, Any] = {}
    if args.fp16:
        train_params["precision"] = 16

    # Set GPU & Data Parallel
    args.num_gpus = 0 if args.gpus is None else len(args.gpus)
    if args.num_gpus > 1:
        train_params["accelerator"] = "dp"
    train_params["val_check_interval"] = 0.25  # check validation set 4 times during a training epoch
    train_params["num_sanity_val_steps"] = args.num_sanity_val_steps
    train_params["accumulate_grad_batches"] = args.accumulate_grad_batches
    train_params["profiler"] = extra_train_kwargs.get("profiler", None)

    return pl.Trainer.from_argparse_args(
        args,
        weights_summary=None,
        callbacks=[logging_callback] + extra_callbacks,
        logger=csv_logger,
        checkpoint_callback=checkpoint_callback,
        **train_params,
    )


def log_args(args: argparse.Namespace) -> None:
    args_dict = vars(args)
    max_len = max([len(k) for k in args_dict.keys()])
    fmt_string = "\t%" + str(max_len) + "s : %s"
    logger.info("Arguments:")
    for key, value in args_dict.items():
        logger.info(fmt_string, key, value)


def main() -> None:
    command = sys.argv[1].lower()

    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "command",
        type=str,
        help=f"Whether to run klue with command ({Command.tolist()})",
    )
    if command in ["--help", "-h"]:
        parser.parse_known_args()
    elif command not in Command.tolist():
        raise ValueError(f"command is positional argument. command list: {Command.tolist()}")

    # Parser (general -> data -> model)
    parser = add_general_args(parser, os.getcwd())
    parsed, _ = parser.parse_known_args()
    task_name = parsed.task

    task = KLUE_TASKS.get(task_name, None)

    if not task:
        raise ValueError(f"task_name is positional argument. task list: {list(KLUE_TASKS.keys())}")

    parser = task.processor_type.add_specific_args(parser, os.getcwd())
    parser = task.model_type.add_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    log_args(args)

    trainer = make_klue_trainer(args)
    task.setup(args, command)

    if command == Command.Train:
        logger.info("Start to run the full optimization routine.")
        trainer.fit(**task.to_dict())

        # load the best checkpoint automatically
        trainer.get_model().eval_dataset_type = "valid"
        val_results = trainer.test(test_dataloaders=task.val_loader, verbose=False)[0]
        print("-" * 80)

        output_val_results_file = os.path.join(args.output_dir, "val_results.txt")
        with open(output_val_results_file, "w") as writer:
            for k, v in val_results.items():
                writer.write(f"{k} = {v}\n")
                print(f" - {k} : {v}")
        print("-" * 80)

    elif command == Command.Evaluate:
        trainer.test(task.model, test_dataloaders=task.val_loader)
    elif command == Command.Test:
        trainer.test(task.model, test_dataloaders=task.test_loader)


if __name__ == "__main__":
    main()
