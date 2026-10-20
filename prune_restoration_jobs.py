"""List expired terminal jobs; --apply removes only their explicitly recorded renders.

Uploads, analysis, untracked historical outputs and running jobs are never deleted.
"""
import argparse,json,time
from pathlib import Path

def expired_jobs(output,days=30,now=None):
    output=Path(output).resolve()
    if days<1:raise ValueError('Retention must be at least one day')
    cutoff=(time.time() if now is None else now)-days*86400
    jobs=(output/'jobs').resolve();restored=(output/'restored').resolve()
    if jobs.parent!=output or restored.parent!=output:
        raise ValueError('Job/render directories must resolve directly inside the output directory')
    expired=[]
    for path in jobs.glob('*.json'):
        if path.is_symlink():continue
        try:record=json.loads(path.read_text(encoding='utf-8'))
        except (ValueError,OSError):continue
        if record.get('status') not in ('completed','needs_revision','cancelled','failed','interrupted'):continue
        if not isinstance(record.get('updated_at'),(int,float)) or record['updated_at']>=cutoff:continue
        files=[]
        for name in record.get('artifacts',[]):
            if not isinstance(name,str) or not name or '/' in name or '\\' in name or Path(name).name!=name:
                raise ValueError(f'Unsafe artifact in {path.name}')
            candidate=(restored/name).resolve()
            if candidate.parent!=restored:raise ValueError('Artifact escapes restored directory')
            if candidate.exists():files.append(candidate)
        expired.append((path,files))
    return expired

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path);parser.add_argument('--days',type=int,default=30)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args()
    selected=expired_jobs(args.output,args.days)
    print(json.dumps({'dry_run':not args.apply,'jobs':len(selected),
                      'files':[str(p) for record,files in selected for p in [record]+files]},indent=2))
    if args.apply:
        for record,files in selected:
            for path in files:path.unlink()
            record.unlink()
if __name__=='__main__':main()
