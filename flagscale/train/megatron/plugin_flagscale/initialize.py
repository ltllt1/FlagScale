import time

import torch
import torch_npu

def _compile_dependencies():
    if torch.distributed.get_rank() == 0:
        start_time = time.time()
        print('> compiling dataset index builder ...')
        from megatron.core.datasets.utils import compile_helpers

        compile_helpers()
        print(
            '>>> done with dataset index builder. Compilation time: {:.3f} seconds'.format(time.time() - start_time),
            flush=True,
        )