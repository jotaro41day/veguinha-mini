"""Portable copies of Brian2 Cython random buffers, never process addresses."""
import ctypes
import numpy as np
from brian2 import get_device
from brian2.codegen.generators.cython_generator import _BUFFER_SIZE


def capture():
    device=get_device();state={}
    for name in ('rand','randn'):
        pointer=ctypes.c_size_t(int(getattr(device,name+'_buffer')[0])).value
        index=int(getattr(device,name+'_buffer_index')[0])
        values=(np.ctypeslib.as_array((ctypes.c_double*_BUFFER_SIZE).from_address(pointer)).copy()
                if pointer and index else None)
        state[name]=dict(index=index,values=values)
    return state


def restore(state):
    device=get_device();libc=ctypes.CDLL(None)
    libc.malloc.argtypes=[ctypes.c_size_t];libc.malloc.restype=ctypes.c_void_p
    libc.free.argtypes=[ctypes.c_void_p];libc.free.restype=None
    for name in ('rand','randn'):
        pointer=getattr(device,name+'_buffer');index=getattr(device,name+'_buffer_index')
        saved=state.get(name) if state else None
        if saved is None or not saved['index']:
            # Old files contain only an address, not the remaining random values.
            # Keep this process's valid allocation and regenerate at its next draw.
            index[:]=0
            continue
        values=np.asarray(saved['values'],dtype=np.float64)
        if values.shape!=(_BUFFER_SIZE,) or not 0<saved['index']<_BUFFER_SIZE:
            raise ValueError('buffer aleatório incompatível com Brian2 instalado')
        allocation=libc.malloc(values.nbytes)
        if not allocation:raise MemoryError('buffer aleatório')
        ctypes.memmove(allocation,values.ctypes.data,values.nbytes)
        if int(pointer[0]):libc.free(ctypes.c_size_t(int(pointer[0])).value)
        # Android may tag malloc pointers in the high byte; preserve their bits
        # in Brian's signed np.intp slot instead of converting as a C long.
        pointer[0]=ctypes.c_ssize_t(allocation).value;index[0]=saved['index']
