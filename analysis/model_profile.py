#!/usr/bin/env python3
"""Run the real trainer's optimizer-step profile with an optional QK prototype.

Arguments other than --qk-rms-block go to algebraic.train unchanged. The
prototype substitutes RMS normalization only for rank-four D=64 Q/K arrays.
Its formula and learned gains remain the same. It is not model dispatch.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument("--qk-rms-block",type=int,default=0)
    p.add_argument("--forward-native",action="store_true")
    p.add_argument("--faithful-backward",action="store_true")
    p.add_argument("--gain-partials-only",action="store_true")
    p.add_argument("--native-qk",action="store_true")
    extra,rest=p.parse_known_args()
    from algebraic import algebraic as A
    from algebraic import train
    from algebraic import model
    original=A.rms_norm
    original_qk=model._qk_norm
    if extra.native_qk:
        model._qk_norm=lambda x,gain,cfg:original(x,gain,cfg.eps)
    if extra.qk_rms_block:
        from analysis.qk_rms_kernel import make_norm
        candidate=make_norm(extra.qk_rms_block,extra.forward_native,extra.faithful_backward,extra.gain_partials_only)
        def norm(x,gain,cfg):
            if x.ndim==4 and x.shape[-1]==64 and gain is not None and cfg.eps==1e-5:
                return candidate(x,gain)
            return original(x,gain,cfg.eps)
        model._qk_norm=norm
    args=train.get_args(rest)
    if not args.profile_steps:
        raise ValueError("this wrapper requires --profile-steps")
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    (out/"prototype.json").write_text(json.dumps({
        "qk_rms_block":extra.qk_rms_block,"prototype":bool(extra.qk_rms_block),
        "forward_native":extra.forward_native,"faithful_backward":extra.faithful_backward,
        "gain_partials_only":extra.gain_partials_only,
        "native_qk":extra.native_qk,
        "deployed_model_unchanged":True,
        "source_sha256":{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
            ("analysis/model_profile.py","analysis/qk_rms_kernel.py","src/algebraic/kernels/qk_norm.py","src/algebraic/model.py",
             "src/algebraic/algebraic.py","src/algebraic/kernels/flash.py")}},indent=2)+"\n")
    try:
        return train.main(rest)
    finally:
        model._qk_norm=original_qk


if __name__=="__main__":
    raise SystemExit(main())
