from typing import Dict

import numpy as np

from .arxiv import get_arxiv_2000, get_arxiv_full
from .benchmarks import SUPPORTED_TASK_MAP
from .c4 import get_c4_data
from .fineweb import get_fineweb_data
from .fineweb_edu import get_fineweb_edu_data
from .openwebtext2 import get_openwebtext2_data
from .reader import DataReader
from .redpajama import get_redpajama_data, get_redpajamav2_data
from .shakespeare import get_shakespeare_data
from .slimpajama import get_slimpajama_data
from .token_bin import get_token_bin_data
from .wikitext import get_wikitext_data


def get_dataset(args) -> Dict[str, np.ndarray]:
    """Fetch the right dataset given by the args.dataset parameter. The logic for each dataset is
    contained in its own python file. The expected format at the moment is a dictionary of np.memmap
    containing two keys: 'train' and 'val', corresponding to the tokenized training and validation data.
    """
    if args.dataset == "token-bin":
        return get_token_bin_data(args.train_data_path, args.val_data_path)
    if args.dataset == "wikitext":
        return get_wikitext_data(args.datasets_dir)
    if args.dataset == "shakespeare-char":
        return get_shakespeare_data(args.datasets_dir)
    if args.dataset == "arxiv2000":
        return get_arxiv_2000(args.datasets_dir)
    if args.dataset == "arxiv":
        return get_arxiv_full(args.datasets_dir)
    if args.dataset == "arxiv+wiki":
        arxiv_data = get_arxiv_full(args.datasets_dir)
        wiki_data = get_wikitext_data(args.datasets_dir)
        train_data = np.concatenate((arxiv_data["train"], wiki_data["train"]))
        val_data = np.concatenate((arxiv_data["val"], wiki_data["val"]))
        return {"train": train_data, "val": val_data}
    if args.dataset == "openwebtext2":
        return get_openwebtext2_data(args.datasets_dir)
    if args.dataset == "redpajama":
        return get_redpajama_data(args.datasets_dir)
    if args.dataset == "redpajamav2":
        return get_redpajamav2_data(args.datasets_dir)
    if args.dataset == "slimpajama":
        return get_slimpajama_data(args.datasets_dir)
    if args.dataset == "fineweb":
        return get_fineweb_data(args.datasets_dir)
    if args.dataset == "finewebedu":
        return get_fineweb_edu_data(args.datasets_dir)
    if args.dataset == "c4":
        return get_c4_data(args.datasets_dir)
    if args.dataset in SUPPORTED_TASK_MAP:
        return get_benchmark_task(args.dataset)
    else:
        raise NotImplementedError(f"Unknow dataset key '{args.dataset}'")


def get_benchmark_task(name, **kwargs):
    """Fetch the right benchmark task given by the name parameter. The logic for each task is
    contained in its own python file.
    """
    try:
        fn = SUPPORTED_TASK_MAP[name]
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{name}'. Supported: {sorted(SUPPORTED_TASK_MAP.keys())}"
        )
    return fn(**kwargs)
