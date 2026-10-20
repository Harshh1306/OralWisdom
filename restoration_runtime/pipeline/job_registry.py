"""Durable local job state and cancellation for the single GPU worker."""
import json,re,time,os
from pathlib import Path
from threading import Event,RLock
from uuid import uuid4


class GenerationCancelled(RuntimeError):
    pass


class JobRegistry:
    def __init__(self,folder):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=True)
        self.lock=RLock();self.events={}
        for path in self.folder.glob('*.json'):
            try:
                record=json.loads(path.read_text(encoding='utf-8'))
                if record.get('status')=='running':
                    record.update(status='interrupted',error='The previous worker stopped before completion',updated_at=time.time())
                    self._write(record)
            except (ValueError,OSError):
                continue

    def _path(self,job_id):
        if not isinstance(job_id,str) or not re.fullmatch('[0-9a-f]{32}',job_id): raise ValueError('Invalid job ID')
        return self.folder/(job_id+'.json')

    def _write(self,record):
        path=self._path(record['job_id']);temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(record,indent=2),encoding='utf-8');temporary.replace(path)

    def start(self,image_stem,plan_id,job_id=None):
        job_id=job_id or uuid4().hex
        with self.lock:
            if self._path(job_id).exists(): raise ValueError('This job ID was already used')
            record=dict(job_id=job_id,image_stem=image_stem,plan_id=plan_id,status='running',
                created_at=time.time(),updated_at=time.time(),progress={'phase':'loading','step':0})
            self.events[job_id]=Event();self._write(record)
        return job_id

    def read(self,job_id):
        with self.lock:
            path=self._path(job_id)
            return json.loads(path.read_text(encoding='utf-8')) if path.exists() else None

    def update(self,job_id,**values):
        with self.lock:
            record=self.read(job_id)
            if record is None: raise ValueError('Unknown job')
            record.update(values,updated_at=time.time());self._write(record)
            if record['status']!='running': self.events.pop(job_id,None)

    def cancel(self,job_id):
        self._path(job_id)
        with self.lock:
            event=self.events.get(job_id)
            if event is None: return False
            event.set();self.update(job_id,cancel_requested=True)
            return True

    def callback(self,job_id,timeout_seconds=600):
        began=time.monotonic()
        def progress(arch,step,total):
            if self.events[job_id].is_set(): raise GenerationCancelled('Generation cancelled')
            if time.monotonic()-began>timeout_seconds: raise GenerationCancelled('Generation exceeded its configured time limit')
            if step==0 or step==total or step%4==0:
                self.update(job_id,progress=dict(phase=arch,step=step,total_steps=total))
        return progress
