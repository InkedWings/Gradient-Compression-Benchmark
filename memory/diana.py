import torch
from memory.memory import Memory

class DianaMemory(Memory):

    def __init__(self):
        super().__init__(cumulative_grads={}, residuals={})

    def __str__(self):
        return "diana_memory"

    def compensate(self, tensor, name, worker):
        """Update the tensor with the residuals."""
        return tensor

    def getproto(self, name, worker):
        idx = name + str(worker)
        if idx in self.residuals:
            proto = self.residuals[idx]
        else:
            raise RuntimeError('Empty ProtoType!!!')
        return proto

    def update(self, full_grad, name, worker, compressor):
        idx = name + str(worker)

        if idx in self.residuals:
            diff = full_grad - self.residuals[idx]
            comp_diff, ctx = compressor.compress(diff, name)
            decomp_diff = compressor.decompress(comp_diff, ctx)
            proto = self.residuals[idx] + decomp_diff
            self.residuals[idx] = torch.clone(proto).detach()
        else:
            comp_grad, ctx = compressor.compress(full_grad, name)
            decomp_grad = compressor.decompress(comp_grad, ctx)
            self.residuals[idx] = torch.clone(decomp_grad).detach()