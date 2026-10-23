"""Small, CPU-first experiment utilities. No training runs at import time."""
import argparse,json,os,random
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

def tensor(x): return torch.as_tensor(np.asarray(x),dtype=torch.float32)
def normalize(train,test):
    scaler=MinMaxScaler().fit(train.reshape(-1,train.shape[-1]))
    def transform(x): return scaler.transform(x.reshape(-1,x.shape[-1])).reshape(x.shape).astype('float32')
    return transform(train),transform(test),scaler

def require(root,relative):
    p=Path(root)/relative
    if not p.is_file(): raise FileNotFoundError(f'Missing input: {p}. See DATA.md. Raw data are not in this repository.')
    return p

def fit(model,loss_fn,n,args,after_epoch=None):
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    history=[]
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint,map_location='cpu',weights_only=True))
        return history
    for epoch in range(args.epochs):
        model.train();values=[]
        indices=torch.randperm(n)
        for idx in indices.split(args.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss,parts=loss_fn(idx)
            if not torch.isfinite(loss):raise FloatingPointError(f'Non-finite loss at epoch {epoch+1}: {parts}')
            loss.backward()
            grads=[p.grad for p in model.parameters() if p.grad is not None]
            if not grads or not all(torch.isfinite(g).all() for g in grads):raise FloatingPointError('Missing or non-finite gradients')
            torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            optimizer.step()
            values.append({'loss':float(loss.detach()),**{k:float(v.detach()) if torch.is_tensor(v) else float(v) for k,v in parts.items()}})
        row={k:sum(v[k] for v in values)/len(values) for k in values[0]};row['epoch']=epoch+1;history.append(row)
        if after_epoch: after_epoch(epoch,history)
        print(json.dumps(row),flush=True)
    return history

def finish(model,args,history,pred,target,extra=None,scalers=None):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    pred=np.asarray(pred);target=np.asarray(target)
    if pred.shape!=target.shape:raise ValueError(f'Prediction shape {pred.shape} != target shape {target.shape}')
    if not np.isfinite(pred).all():raise FloatingPointError('Non-finite held-out predictions')
    report={'data_kind':'synthetic_demo' if args.demo else 'local_research_data',
            'scope':'Short-run execution check; not a reproduction of published performance.',
            'seed':args.seed,'epochs':args.epochs if not args.checkpoint else 0,
            'max_samples':args.max_samples,'prediction_shape':list(pred.shape),
            'heldout_mse':float(np.mean((pred-target)**2)),
            'history':history,**(extra or {})}
    torch.save(model.state_dict(),out/'model.pt')
    np.savez_compressed(out/'predictions.npz',prediction=pred,target=target)
    if scalers:
        arrays={}
        for name,s in scalers.items():
            for attr in ['scale_','min_','data_min_','data_max_']: arrays[name+'_'+attr]=getattr(s,attr)
        np.savez(out/'preprocessing.npz',**arrays)
    (out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='history'}),flush=True)
    return report

def main(run):
    p=argparse.ArgumentParser(description='Portable research-code experiment. Defaults are short execution checks.')
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--demo',action='store_true');g.add_argument('--data-root',type=Path)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--max-samples',type=int,default=128,help='Cap samples/windows; trajectory models use smaller documented caps.')
    p.add_argument('--seed',type=int,default=22);p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--output',type=Path,default=Path('outputs/run'));p.add_argument('--checkpoint',type=Path)
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or a.max_samples<16:p.error('epochs and batch-size must be positive; max-samples must be >=16')
    os.environ.setdefault('KERAS_BACKEND','torch')
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    protocol={'demo':a.demo,'data_root':str(a.data_root.resolve()) if a.data_root else None,
              'seed':a.seed,'max_samples':a.max_samples}
    try:
        if a.checkpoint:
            saved_path=a.checkpoint.parent/'protocol.json'
            if not saved_path.is_file():raise ValueError('Checkpoint requires its sibling protocol.json from this runner')
            if json.loads(saved_path.read_text())!=protocol:
                raise ValueError('Checkpoint replay requires the same data root, demo flag, seed and max-samples; it does not resume training')
        run(a)
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    except (FileNotFoundError,ValueError) as e:p.exit(2,str(e)+'\n')
