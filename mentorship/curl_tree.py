from matplotlib import pyplot as plt
import numpy as np
import json
from pathlib import Path
        
SCALE = 0.8
CURL = 45
    
from math import cos, sin
def rot(v, theta):
    theta = np.deg2rad(theta)
    mat = np.array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])
    return np.dot(mat, v)

def unfurl(X, base_vector=np.array([0,1]), origin=np.array([0,0]), depth=None, cdepth=0):
    vs = []
    
    if 'children' not in X:
        return []
    
    if depth is not None:
        if cdepth >= depth:
            return []
    
    girls = [c for c in X['children'] if c['gender'] == 'woman']
    boys = [c for c in X['children'] if c['gender'] == 'man']
    
    #print(origin, base_vector)
    
    #print(len(girls), len(boys))
    
    for ci, c in enumerate(girls):
        #new_v = rot( base_vector, -CURL * (ci+1) )# / len(girls) )
        new_v = rot( base_vector, -CURL * (ci+1) / len(girls) )
        new_v = new_v * SCALE
        #print( np.power((new_v**2).sum(), 0.5) )

        vs.append( (origin, origin+new_v) )
        vs += unfurl(c, new_v, origin+new_v, depth, cdepth+1)

    for ci, c in enumerate(boys):
        #new_v = rot( base_vector, CURL * (ci+1) )# / len(boys) )
        new_v = rot( base_vector, CURL * (ci+1) / len(boys) )
        new_v = new_v * SCALE
        #print( np.power((new_v**2).sum(), 0.5) )

        vs.append( (origin, origin+new_v) )
        vs += unfurl(c, new_v, origin+new_v, depth, cdepth+1)
            
    return vs

fns = list(Path('treeData/').glob("*.json"))

plt.figure(figsize=(25,25))

for i, treefn in enumerate(fns):
    plt.subplot(6,6,i+1)
    with open(treefn) as f:
        tree = json.loads(f.read())
        for s, e in unfurl(tree, depth=None):
            plt.plot(
                [s[0],e[0]],
                [s[1],e[1]],
                color='black',
                alpha=0.3
            )
        plt.xticks([],[])
        plt.yticks([],[])
        plt.title(treefn.name)
        
plt.savefig('curl_tree.png')