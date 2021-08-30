import sys
from pathlib import Path

file = Path(__file__).resolve()
file0 = file.parents[1]
file1 = file0 / 'config/configs'
sys.path.extend([str(file0), str(file1)])
