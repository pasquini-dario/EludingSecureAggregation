import os, sys, glob
from tqdm import tqdm
import myCMD, myOS


EX = "python canary_attack_main.py %s %s"

OUTP = './results/'

try:
    CONF = sys.argv[1]
    num_run = int(sys.argv[2])
    nGPU = int(sys.argv[3])
    shift = int(sys.argv[4])
    id_shift = int(sys.argv[5])
except:
    print("USAGE: 'CONF  NUM_RUNS nGPUs GPUidShift EXP_id_shift")
    sys.exit(1)

X = []
for i in range(num_run):
    ex = (CONF, i+id_shift)
    X.append(ex)
print(*X, sep='\n')

myCMD.runMultiGPU(EX, X, nGPU, shift)