"""Independent diagnostic of the local repairs prescribed by Attention study section 6."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import hashlib
import numpy as np
from tests.reference_attention import attention,softmax
from scripts.primitives_experiments import summary
from scripts.attention_records import environment,write_json
rng=np.random.default_rng(46);x=rng.normal(size=(2048,128));noise=rng.normal(0,.05,size=x.shape)
rows=[]
for dk in (1,16,32,64,128,256,512,1024,4096):
    scale=1/np.sqrt(dk);a=x*scale;an=(x+noise)*scale
    q=softmax(a);db=np.linalg.norm(softmax(an)-q,axis=-1)
    for sink in (.25,.5,.75,1.):
        p=attention(a,sink);da=np.linalg.norm(attention(an,sink)-p,axis=-1)
        w1=np.abs(np.cumsum(p-q,axis=-1)[:,:-1]).mean(-1)
        gain=float(db.mean()/da.mean())
        rows.append({'dk':dk,'scale':float(scale),'sink':sink,'noise_ratio':gain,'w1':summary(w1),'passed':gain>=100 and np.mean(w1)<=.05})
r={'environment':environment(),'command':sys.argv,'diagnostic_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
   'seed':46,'trials':2048,'length':128,'noise_sigma':.05,'protocol':'Both raw scores and common pre-scaling noise multiplied by rsqrt(dk), for both arms',
   'rows':rows,'passed':any(r['passed'] for r in rows)}
write_json(ROOT/'results/attention/iterations/scale-sink-sweep.json',r)
print('Any passing repair:',r['passed'],'noise ratio range:',min(r['noise_ratio'] for r in rows),max(r['noise_ratio'] for r in rows))
