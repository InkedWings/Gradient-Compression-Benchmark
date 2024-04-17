from memory.none import NoneMemory
from memory.residual import ResidualMemory
from memory.dgc import DGCMemory
from memory.diana import DianaMemory

def memory_chooser(args, gradient_clipping=0.25):
    """method for selecting memory method
        from command line argument."""

    inp = args.memory

    if inp == 'none':
        return NoneMemory()

    if inp == 'residual':
        return ResidualMemory()

    if inp == 'dgc':
        return DGCMemory(args.dgc_momentum, gradient_clipping)
    
    if inp == 'diana':
        return DianaMemory()

    else:
        raise ValueError('memory argument invalid')
