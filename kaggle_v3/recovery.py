"""Bounded process recovery, OS-released locks, and persistent integrity halts."""
import json
import os
import queue
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def controller_lock(path):
    """Kernel releases this lock on process death. Never unlink an active lock inode."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, 'a+b'); acquired = False
    try:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0: f.write(b' '); f.flush()
            f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        acquired = True
        f.seek(0); previous = f.read().strip()
        if previous:
            old = json.loads(previous)
            if old.get('protocol') != 2:
                try: os.kill(int(old['pid']), 0)
                except ProcessLookupError: pass
                else: raise RuntimeError('An older controller may still be running; stop it first')
        f.seek(0); f.truncate()
        f.write(json.dumps(dict(protocol=2, pid=os.getpid())).encode()); f.flush(); os.fsync(f.fileno())
        yield
    except BlockingIOError:
        raise RuntimeError('Another controller is using this output directory') from None
    finally:
        if acquired:
            if os.name == 'nt':
                f.seek(0); msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else: fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def stop_process(process):
    if process.poll() is not None: return
    if os.name == 'nt':
        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
    else:
        os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL); process.wait()


def run(command, env=None, cwd=None, log=None, retries=0):
    """Retry only allowlisted transient errors; no retry of OOM/data/integrity errors."""
    deadline = float((env or os.environ).get('NVDA_DEADLINE', time.time()+6*3600))
    for attempt in range(retries+1):
        if time.time() >= deadline: raise TimeoutError('Research time budget exhausted')
        handle = open(log, 'a', encoding='utf-8') if log else None
        process = subprocess.Popen(list(map(str, command)), env=env, cwd=cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
            start_new_session=os.name != 'nt')
        messages = queue.Queue(); tail = []; heartbeat = time.monotonic()
        def reader():
            try:
                for line in process.stdout: messages.put(line)
            finally: messages.put(None)
        thread = threading.Thread(target=reader, daemon=True); thread.start()
        try:
            while True:
                if time.time() >= deadline: raise TimeoutError('Research time budget exhausted; checkpoints retained')
                try: line = messages.get(timeout=.25)
                except queue.Empty: line = ''
                if line is None: break
                if line:
                    print(line, end='', flush=True); tail = (tail+[line])[-80:]
                    if handle: handle.write(line); handle.flush()
                if time.monotonic()-heartbeat > 30:
                    print('Stage running; remaining budget:', int(deadline-time.time()), 'seconds', flush=True)
                    heartbeat = time.monotonic()
            code = process.wait(timeout=max(.1, deadline-time.time()))
        except BaseException:
            stop_process(process); raise
        finally:
            process.stdout.close()
            if handle: handle.close()
        if code == 0: return
        error = ''.join(tail).lower()
        transient = any(s in error for s in ('connection reset', 'timed out', 'temporary failure', 'nccl error'))
        fatal = any(s in error for s in ('out of memory', 'mismatch', 'non-finite', 'budget', 'coverage', 'corrupt'))
        if not transient or fatal or attempt >= retries:
            raise RuntimeError(f'Process exited {code}; immutable checkpoints retained. See stage log.')
        print('Recovering transient failure from the same checkpoint:', attempt+1, flush=True)
        time.sleep(min(2**attempt, 4))


def health(root):
    from core import read, rollback, verify_champion, dump
    root = Path(root); halt = root/'risk_halt.json'; registry = root/'champion.json'
    report = dict(halted=halt.exists(), registries={})
    for name in ('champion.json','paper_candidate.json'):
        registry=root/name; status=dict(artifact='absent',rollback=False)
        if registry.exists():
            try: verify_champion(registry); status['artifact']='verified'
            except Exception:
                status['artifact']='invalid'
                try: status['rollback']=rollback(registry,halt,'artifact_integrity_failure')
                except Exception: dump(halt,dict(halted=True,reason='rollback_failed'))
                report['halted']=True
        report['registries'][name]=status
    if halt.exists(): report['halt']=read(halt)
    return report
