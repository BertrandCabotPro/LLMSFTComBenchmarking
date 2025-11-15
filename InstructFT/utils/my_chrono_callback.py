from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
import torch

from .chrono import Chronometer

###############################
# Author : Bertrand CABOT from IDRIS(CNRS)
#
# #######################



class MyChronoCallback(pl.Callback):

    def __init__(
        self,
        rank: int = 0,
        grad_acc: int= 1,
    ):
        self.rank = rank
        self.chronometer = Chronometer(rank, grad_acc)


    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self.rank == 0: print(f'Pre-loop Model MaxMemory for GPU:{self.rank} {torch.cuda.max_memory_allocated()/2**30} GBytes')      
        self.chronometer.start()
        self.chronometer.dataload()

    def on_train_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int
    ) -> None:
        self.chronometer.dataload()
        self.chronometer.training()
        self.chronometer.forward()


    def on_before_backward(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", loss: torch.Tensor) -> None:
        self.chronometer.forward()
        self.chronometer.backward()
    

    def on_train_batch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.chronometer.backward()
        self.chronometer.training()
        self.chronometer.dataload()


    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.chronometer.display()
        if self.rank == 0: print(f'MaxMemory for GPU:{self.rank} {torch.cuda.max_memory_allocated()/2**30} GBytes')
        