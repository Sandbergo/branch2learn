import os
from pathlib import Path

i = 50_000
for path in Path(f'branch2learn/data/samples/setcover/valid').glob('sample_*.pkl'):
    i += 1
    #print(i)
    #print(str(path)[49:-4])
    if int(str(path)[47:-4]) > 10_000:
        new_path = 'branch2learn/data/samples/setcover/train/sample_' + str(i) + '.pkl'
        #print(new_path)
        #exit(0)
        #print(new_path)
        #exit(0)
        os.rename(Path(str(path)), Path(new_path))
        