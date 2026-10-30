# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
# distutils: extra_compile_args=-O3 -ffast-math -fno-finite-math-only
"""Euler .5ms motor-only diagnostic loop; independently checked against Brian2."""
import numpy as np
cimport numpy as cnp


def simulate(double[:] initial_v,double[:,::1] noise,double[:,::1] pattern,
             int n,double tonic,double noise_gain,double[:] thresholds):
    cdef cnp.ndarray[cnp.float64_t,ndim=1] va=np.array(initial_v,copy=True)
    cdef cnp.ndarray[cnp.float64_t,ndim=1] ua=.2*np.array(initial_v,copy=True)
    cdef cnp.ndarray[cnp.float64_t,ndim=1] sa=np.zeros(64)
    cdef double[:] v=va,u=ua,s=sa
    cdef cnp.ndarray[cnp.int32_t,ndim=2] counts=np.zeros((1000,n),dtype=np.int32)
    cdef int k,j,phase
    cdef double oldv,oldu,newv,newu,current
    cdef list ts=[],ids=[]
    for k in range(40000):
        phase=k%760
        for j in range(n):
            oldv=v[j];oldu=u[j]
            current=tonic+s[j]+noise_gain*noise[k//2,j]
            newu=oldu+.5*.02*(.2*oldv-oldu)
            newv=oldv+.5*(.04*oldv*oldv+5*oldv+140-oldu+current)
            s[j]=s[j]-.5*s[j]/5
            if newv>=thresholds[j]:
                counts[k//40,j]+=1;ts.append(k*.0005);ids.append(j)
                newv=-65;newu+=8
            v[j]=newv;u[j]=newu
            s[j]+=pattern[phase,j]
    return counts,np.asarray(ts),np.asarray(ids,dtype=int)
