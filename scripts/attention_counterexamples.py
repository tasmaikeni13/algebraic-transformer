#!/usr/bin/env python3
"""Measure cases where octic attention lacks a uniform noise advantage."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='1'
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import jax
import jax.numpy as jnp
from src.attention import algebraic_softmax,_normalized_forward
from tests.reference_attention import attention,normalized_attention,softmax
from scripts.attention_records import environment,write_json
from scripts.primitives_experiments import summary


def counterexamples():
    rng=np.random.default_rng(42);x=rng.normal(size=(10000,64));noise=rng.normal(0,.05,size=x.shape)
    da=np.linalg.norm(attention(x+noise)-attention(x),axis=-1)
    db=np.linalg.norm(softmax(x+noise)-softmax(x),axis=-1)
    j=np.asarray(jax.jacrev(lambda z:_normalized_forward(z,0.)[0])(jnp.zeros(2)))
    raw=np.asarray(jax.jacrev(lambda z:algebraic_softmax(z,0.))(jnp.zeros(2)))
    tied=np.ones(64);delta=np.r_[.05,-.05,np.zeros(62)]
    tied_gain=float(np.linalg.norm(softmax(tied+delta)-softmax(tied))/np.linalg.norm(attention(tied+delta)-attention(tied)))
    return {'seed':42,'trials':10000,'length':64,'noise_sigma':.05,
            'scores':'independent N(0,1) raw scores','noise_metric':'L2 displacement with identical raw-coordinate Gaussian noise',
            'noise_alg':summary(da),'noise_softmax':summary(db),'noise_suppression_ratio':float(db.mean()/da.mean()),
            'required_suppression_ratio':100,'noise_ratio_above_100':bool(db.mean()/da.mean()>=100),
            'normalized_zero_jacobian':j.tolist(),'normalized_zero_spectral_norm':float(np.linalg.norm(j,2)),
            'raw_zero_jacobian':raw.tolist(),'tied_scores_unit_length64_noise_ratio':tied_gain,
            'sharpness_exact_float':float(51841+23184*np.sqrt(5)),
            'two_token_distribution_counterexample':{'scores':[2.,0.],'algebraic':attention(np.array([2.,0.])).tolist(),
                                                     'softmax':softmax(np.array([2.,0.])).tolist()}}

if __name__=='__main__':
    r={'environment':environment(),'command':sys.argv,**counterexamples()}
    write_json(ROOT/'results/attention/counterexamples.json',r)
    print('Noise ratio above 100:',r['noise_ratio_above_100'],flush=True)
