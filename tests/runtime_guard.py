"""Root-side isolation helpers; no candidate modules are ever imported here."""
import os
from pathlib import Path
import signal
import stat
import subprocess
import ctypes
import errno
import platform

RUNTIME_UID=1001
AUDITOR_UID=1002


def restrict_worker_network():
    """Deny IP sockets in evaluation children without changing agent inference."""
    architectures={'x86_64':(41,0xc000003e),'aarch64':(198,0xc00000b7)}
    target=architectures.get(platform.machine())
    if target is None: raise RuntimeError('Unsupported evaluation sandbox architecture')
    syscall,architecture=target
    class Filter(ctypes.Structure):
        _fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint)]
    class Program(ctypes.Structure):
        _fields_=[('len',ctypes.c_ushort),('filter',ctypes.POINTER(Filter))]
    instructions=(Filter*11)(Filter(0x20,0,0,4),Filter(0x15,0,8,architecture),
        Filter(0x20,0,0,0),Filter(0x45,6,0,0x40000000),Filter(0x15,0,4,syscall),
        Filter(0x20,0,0,16),Filter(0x15,3,0,2),Filter(0x15,2,0,10),
        Filter(0x06,0,0,0x7fff0000),Filter(0x06,0,0,0x7fff0000),
        Filter(0x06,0,0,0x00050000|errno.EPERM))
    program=Program(len(instructions),instructions)
    libc=ctypes.CDLL(None,use_errno=True)
    if libc.prctl(38,1,0,0,0) or libc.prctl(22,2,ctypes.byref(program),0,0):
        raise RuntimeError('Could not restrict evaluation worker network')


def kill_uid(uid):
    for item in Path('/proc').iterdir():
        if item.name.isdigit():
            try:
                status=(item/'status').read_text()
                ids=next(line for line in status.splitlines() if line.startswith('Uid:')).split()[1:]
                if str(uid) in ids: os.kill(int(item.name),signal.SIGKILL)
            except (FileNotFoundError,ProcessLookupError,PermissionError): pass


def protected(path):
    current=Path(path)
    for item in [current,*current.parents]:
        st=item.stat()
        if st.st_uid!=0 or st.st_mode & 0o022:
            raise RuntimeError('Trusted runtime is writable by an unprivileged user: '+str(item))


def freeze_tree(path):
    path=Path(path)
    for current,dirs,files in os.walk(path,followlinks=False):
        for name in dirs+files:
            p=Path(current)/name
            st=p.lstat()
            if stat.S_ISLNK(st.st_mode):
                target=p.resolve(strict=True)
                if target.is_relative_to(path): continue
                if not any(target.is_relative_to(Path(prefix)) for prefix in ('/opt/venv','/opt/benchmark','/usr/local','/usr/lib')):
                    raise RuntimeError('Mutable external source/dependency symlink: '+str(p))
                protected(target)
            elif stat.S_ISDIR(st.st_mode):
                os.chown(p,0,0);os.chmod(p,0o755)
            elif stat.S_ISREG(st.st_mode):
                os.chown(p,0,0);os.chmod(p,0o555 if st.st_mode & 0o111 else 0o444)
            else:
                raise RuntimeError('Special file in candidate source/dependency tree: '+str(p))
    os.chown(path,0,0);os.chmod(path,0o755)


def prepare():
    kill_uid(1000)
    for path in ('/opt/venv','/opt/venv/bin','/opt/venv/bin/python','/usr/local/bin/python3.11',
                 '/opt/venv/lib/python3.11/site-packages','/opt/benchmark/protocol.json'):
        protected(path)
    freeze_tree('/app')
    for uid in (RUNTIME_UID,AUDITOR_UID): kill_uid(uid)


def execute(code=None,args=None,user='runtime',timeout=90,cwd='/tmp'):
    uid=RUNTIME_UID if user=='runtime' else AUDITOR_UID
    kill_uid(uid)
    command=['/usr/sbin/runuser','-u',user,'--','/opt/venv/bin/python','-I','-B']
    command += ['-']+(args or []) if code is not None else args
    environment={'PATH':'/opt/venv/bin:/usr/bin:/bin','LANG':'C.UTF-8','HOME':'/tmp',
                 'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1',
                 'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
    try:
        return subprocess.run(command,input=code,text=True,capture_output=True,cwd=cwd,
                              timeout=timeout,env=environment,start_new_session=True,
                              preexec_fn=restrict_worker_network)
    finally:
        kill_uid(uid)
