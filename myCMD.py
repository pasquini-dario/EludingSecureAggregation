import tempfile
import os, sys, glob
from subprocess import Popen, PIPE
import multiprocessing 
from multiprocessing import Pool
import functools
import time

NGPU = 16

def run(cmd_str, encoding='utf-8'):
    p = Popen([cmd_str], stdout=PIPE, stderr=PIPE, shell=True)
    out = p.communicate()
    stdout = out[0].decode(encoding).split('\n')
    stderr = out[1].decode()
    return stdout, stderr


ENVGPU = "CUDA_VISIBLE_DEVICES"

def _wrapper(inputs, cmd, nGPU, GPUidShift):
    name = multiprocessing.current_process().name
    i = GPUidShift + (int(name.split('-')[-1]) -1)
    #time.sleep(i * 5)
    assert i < NGPU
    my_env = os.environ
    my_env[ENVGPU] = str(i)
    print(ENVGPU, i)
    cmd = cmd % inputs
    print('(%d)' % i, cmd)
    o, err = run(cmd)
    print(i, 'STOPPED')
    if err:
        print('\n\n', err, '\n\n')
    return o

def runMultiGPU(cmd, TASKS, nGPU=16, GPUidShift=0):
    f = functools.partial(_wrapper, cmd=cmd, nGPU=nGPU, GPUidShift=GPUidShift)
    print('nGPU', nGPU)
    with Pool(nGPU) as pool:
         out = pool.map(f, TASKS)
    return out


def _wrapperCPU(inputs, cmd):
    name = multiprocessing.current_process().name
    i = int(name.split('-')[-1]) - 1
    cmd = cmd % inputs
    print('%s' % i, cmd)
    o, err = run(cmd)
    print(i, 'STOPPED')
    if err:
        print('\n\n', err, '\n\n')
    return o

def runMultiCPU(cmd, TASKS, nCPU=60):
    f = functools.partial(_wrapperCPU, cmd=cmd)
    with Pool(nCPU) as pool:
         out = pool.map(f, TASKS)
    return out

#------------------____________________--------------------______________________--------------------________________

def _wrapper_RR(inputs, cmd, nGPU, GPUidShift):
    name = multiprocessing.current_process().name
    i = (int(name.split('-')[-1]) -1)
    i = GPUidShift + (i % nGPU)
    #time.sleep(i * 5)
    assert i < NGPU
    my_env = os.environ
    my_env[ENVGPU] = str(i)
    #print(ENVGPU, i)
    cmd = cmd % inputs
    print('(%d)' % i, cmd, '\n')
    o, err = run(cmd)
    print(i, 'STOPPED')
    if err:
        print('\n\n', err, '\n\n')
    return o

def runMultiGPU_RR(cmd, TASKS, max_p_GPU, nGPU=16, GPUidShift=0):
    n_running_P = nGPU * max_p_GPU
    f = functools.partial(_wrapper_RR, cmd=cmd, nGPU=nGPU, GPUidShift=GPUidShift)
    with Pool(n_running_P) as pool:
         out = pool.map(f, TASKS)
    return out