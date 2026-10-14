#!/usr/bin/env python3
"""Fit and certify rational evaluations of sqrt(1+x*x) for the original link.

Coefficients are rounded to float32 before exact rational Bernstein analysis.
Bounds cover real P/Q evaluation, not TPU approximate reciprocal or rounding.
"""

import argparse
from fractions import Fraction as Q
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"src")]

import numpy as np
from analysis.attention_polynomial import bernstein,product


def fit(bound,pdegree,qdegree):
    t=np.linspace(0.,bound*bound,8193)
    root=np.sqrt(1+t)
    rho_negative=1/(root+np.sqrt(t))
    matrix=np.stack([t**i for i in range(1,pdegree+1)]+[-root*t**i for i in range(1,qdegree+1)],axis=1)
    weight=1/rho_negative
    for _ in range(3):
        coefficients=np.linalg.lstsq(matrix*weight[:,None],(root-1)*weight,rcond=None)[0]
        p=np.r_[1,coefficients[:pdegree]].astype(np.float32)
        q=np.r_[1,coefficients[pdegree:]].astype(np.float32)
        weight=1/(rho_negative*np.polynomial.polynomial.polyval(t,q))
    return p.tolist(),q.tolist()


def sqrt_enclosure(value,bits=24):
    # The approximate seed only locates an integer; the final endpoints are
    # checked and corrected entirely with exact integer/rational arithmetic.
    scale=2**bits
    lower=int(math.sqrt(float(value))*scale)
    while Q(lower,scale)**2>value:lower-=1
    while Q(lower+1,scale)**2<=value:lower+=1
    return Q(lower,scale),Q(lower+1,scale)


def certificate(p,q,bound,intervals=256):
    p=[Q(float(np.float32(c))) for c in p]
    q=[Q(float(np.float32(c))) for c in q]
    domain=Q(bound)**2
    residual=product(p,p)
    squared=product(product(q,q),[Q(1),Q(1)])
    residual += [Q(0)]*(len(squared)-len(residual))
    for i,c in enumerate(squared):residual[i]-=c
    rho_min=1/(Q(bound)+sqrt_enclosure(1+domain)[1])
    rows=[]
    for i in range(intervals):
        a,b=domain*Q(i,intervals),domain*Q(i+1,intervals)
        bp,bq,br=bernstein(p,a,b),bernstein(q,a,b),bernstein(residual,a,b)
        pmin,qmin,qmax=min(bp),min(bq),max(bq)
        if pmin<=0 or qmin<=0:raise ValueError("positive root numerator/denominator not certified")
        root_min=sqrt_enclosure(1+a)[0]
        error=max(abs(min(br)),abs(max(br)))/(qmin*qmin*(pmin/qmax+root_min))
        rows.append(dict(a=str(a),b=str(b),p_lower=str(pmin),q_lower=str(qmin),
                         squared_residual_abs=str(max(abs(min(br)),abs(max(br)))),
                         sqrt_abs_error=str(error)))
    root_error=max(Q(row["sqrt_abs_error"]) for row in rows)
    rho_error=root_error/rho_min
    if rho_error>=1:raise ValueError("positive approximate rho not certified")
    low,high=(1-rho_error)**16,(1+rho_error)**16
    tv=((1+rho_error)**8-(1-rho_error)**8)/((1+rho_error)**8+(1-rho_error)**8)
    return dict(method="exact Fraction arithmetic; rational squared residual and Bernstein convex hull",
                p_exact_f32=[str(c) for c in p],q_exact_f32=[str(c) for c in q],
                domain_abs_x=bound,intervals=rows,
                real_bounds=dict(root_absolute_error=float(root_error),rho_relative_error=float(rho_error),
                                 weight_ratio_lower=float(low),weight_ratio_upper=float(high),
                                 weight_max_relative_error=float(max(1-low,high-1)),
                                 normalized_TV_sharp_bound=float(tv),
                                 inverse_root_relative_error=float(root_error/(1-root_error))),
                scope="real rational evaluation; excludes approximate reciprocal and machine arithmetic")


def row_errors(p,q,x,sink=0.):
    root=np.sqrt(1+x*x)
    exact_rho=np.where(x>=0,np.abs(x)+root,1/(np.abs(x)+root))
    approximate_root=np.polynomial.polynomial.polyval(x*x,p)/np.polynomial.polynomial.polyval(x*x,q)
    w=exact_rho**16;wh=(x+approximate_root)**16
    sr=math.sqrt(1+sink*sink)
    ar=np.polynomial.polynomial.polyval(sink*sink,p)/np.polynomial.polynomial.polyval(sink*sink,q)
    w=np.concatenate((w,np.full((len(x),1),(sink+sr)**16)),axis=1)
    wh=np.concatenate((wh,np.full((len(x),1),(sink+ar)**16)),axis=1)
    probability=w/w.sum(1,keepdims=True);estimated=wh/wh.sum(1,keepdims=True)
    tv=.5*np.abs(estimated-probability).sum(1)
    g=np.random.default_rng(659).normal(size=probability.shape);g[:,-1]=0.
    slope=16/np.concatenate((root,np.full((len(x),1),sr)),axis=1)
    sh=16/np.concatenate((approximate_root,np.full((len(x),1),ar)),axis=1)
    vjp=slope*probability*(g-(g*probability).sum(1,keepdims=True))
    vjph=sh*estimated*(g-(g*estimated).sum(1,keepdims=True))
    error=np.linalg.norm(vjph-vjp,axis=1)/np.maximum(np.linalg.norm(vjp,axis=1),1e-100)
    return dict(rows=len(x),keys_plus_sink=x.shape[-1]+1,TV_mean=float(tv.mean()),TV_max=float(tv.max()),
                VJP_relative_l2_mean=float(error.mean()),VJP_relative_l2_max=float(error.max()))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bound",type=float,default=33/32)
    parser.add_argument("--degrees",nargs="+",default=["11","12","21","22"])
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    report=dict(bound=args.bound,model_adopted=False,coefficients={},certificates={},trials={},adversarial={})
    for degree in args.degrees:
        p,q=fit(args.bound,int(degree[0]),int(degree[1]))
        report["coefficients"][degree]=dict(p=p,q=q)
        report["certificates"][degree]=certificate(p,q,args.bound)
        trials=[]
        for seed in [661,673,677,683,691,701,709,719]:
            for std in [.125,.25,.5]:
                x=np.clip(np.random.default_rng(seed).normal(0,std,(64,2048)),-args.bound,args.bound)
                trials.append(dict(seed=seed,std=std,**row_errors(p,q,x)))
        report["trials"][degree]=trials
        report["adversarial"][degree]=[dict(center=center,sink=sink,**row_errors(p,q,np.full((1,2048),center),sink))
                                           for center in [-args.bound,-1.,0.,1.,args.bound] if abs(center)<=args.bound
                                           for sink in [0.,center]]
        print(degree,report["certificates"][degree]["real_bounds"],flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")


if __name__=="__main__":main()
