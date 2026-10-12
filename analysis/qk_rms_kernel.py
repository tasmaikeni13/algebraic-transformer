"""Experimental Pallas RMS normalization with local gain-gradient reduction."""

from functools import partial

import jax
from jax import lax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from src.sota.algebraic import rms_norm as _native_norm


def _forward_kernel(x_ref, gain_ref, y_ref):
    x = x_ref[...].astype(jnp.float32)
    tau = lax.rsqrt(jnp.mean(x*x, axis=-1, keepdims=True)+1e-5)
    y_ref[...] = (x*tau*gain_ref[...]).astype(y_ref.dtype)


def _backward_kernel(x_ref, gain_ref, do_ref, dx_ref, dg_ref, *, faithful=False):
    x = x_ref[...].astype(jnp.float32)
    do = do_ref[...].astype(jnp.float32)
    rad = jnp.mean(x*x, axis=-1, keepdims=True)+1e-5
    tau = lax.rsqrt(rad)
    y = x*tau
    dg = do*gain_ref[...]
    if faithful:
        dtau = jnp.sum(x*dg,axis=-1,keepdims=True)
        radial = (dtau*(-.5*(tau/rad)))/x.shape[-1]
        dx = (dg*tau+radial*x)+x*radial
    else:
        dx = tau*(dg-y*jnp.mean(dg*y, axis=-1, keepdims=True))
    dx_ref[...] = dx.astype(dx_ref.dtype)
    dg_ref[0] = jnp.broadcast_to(jnp.sum(do*y, axis=0, keepdims=True), dg_ref.shape[1:])


def _gain_kernel(x_ref, do_ref, dg_ref):
    x=x_ref[...].astype(jnp.float32)
    do=do_ref[...].astype(jnp.float32)
    tau=lax.rsqrt(jnp.mean(x*x,axis=-1,keepdims=True)+1e-5)
    partial=jnp.sum(do*(x*tau),axis=0,keepdims=True)
    dg_ref[0]=jnp.broadcast_to(partial,dg_ref.shape[1:])


def make_norm(block=512, forward_native=False, faithful_backward=False, gain_partials_only=False):
    def forward(x, gain):
        if forward_native or gain_partials_only:
            return _native_norm(x,gain)
        D = x.shape[-1]
        xf = x.reshape(-1,D)
        N = xf.shape[0]
        if N % block:
            raise ValueError("prototype row count must divide the block")
        y = pl.pallas_call(_forward_kernel, grid=(N//block,),
                in_specs=[pl.BlockSpec((block,D),lambda i:(i,0)),
                          pl.BlockSpec((1,D),lambda i:(0,0))],
                out_specs=pl.BlockSpec((block,D),lambda i:(i,0)),
                out_shape=jax.ShapeDtypeStruct(xf.shape,x.dtype),
                compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",))
            )(xf,gain[None,:])
        return y.reshape(x.shape)
    @jax.custom_vjp
    def norm(x, gain):
        return forward(x,gain)
    def fwd(x,gain):
        return forward(x,gain),(x,gain)
    def bwd(res,do):
        x,gain = res
        D = x.shape[-1]
        xf,dof = x.reshape(-1,D),do.reshape(-1,D)
        N = xf.shape[0]
        if gain_partials_only:
            _, pullback=jax.vjp(_native_norm,x,gain)
            dx,_=pullback(do)
            dg=pl.pallas_call(_gain_kernel,grid=(N//block,),
                    in_specs=[pl.BlockSpec((block,D),lambda i:(i,0)),
                              pl.BlockSpec((block,D),lambda i:(i,0))],
                    out_specs=pl.BlockSpec((1,8,D),lambda i:(i,0,0)),
                    out_shape=jax.ShapeDtypeStruct((N//block,8,D),jnp.float32),
                    compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",))
                )(xf,dof)
            return dx,jnp.sum(dg[:,0,:],axis=0).astype(gain.dtype)
        dx,dg = pl.pallas_call(partial(_backward_kernel,faithful=faithful_backward),grid=(N//block,),
                in_specs=[pl.BlockSpec((block,D),lambda i:(i,0)),
                          pl.BlockSpec((1,D),lambda i:(0,0)),
                          pl.BlockSpec((block,D),lambda i:(i,0))],
                out_specs=[pl.BlockSpec((block,D),lambda i:(i,0)),
                           pl.BlockSpec((1,8,D),lambda i:(i,0,0))],
                out_shape=[jax.ShapeDtypeStruct(xf.shape,x.dtype),
                           jax.ShapeDtypeStruct((N//block,8,D),jnp.float32)],
                compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",))
            )(xf,gain[None,:],dof)
        return dx.reshape(x.shape),jnp.sum(dg[:,0,:],axis=0).astype(gain.dtype)
    norm.defvjp(fwd,bwd)
    return norm
