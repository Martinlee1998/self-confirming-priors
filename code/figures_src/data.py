import json
from pathlib import Path
HERE = Path(__file__).resolve().parent
f3 = json.loads((HERE / 'fig3.json').read_text(encoding='utf-8'))
f45 = json.loads((HERE / 'fig45.json').read_text(encoding='utf-8'))
STUDY1={'swaps':{k:f3[k] for k in ['AB','BC','CD','DE']},'composite':f3['composite'],
        'means':{'Canonical':0.530,'Swap-AB':0.191,'Swap-BC':0.201,'Swap-CD':0.4685,'Swap-DE':0.527}}
RUNS=['C1-r0','C1-r1','C2-r0','C2-r1','C3-r0','C3-r1','C4-r0','C4-r1']
P1_OBS=[32,36,0,0,35,29,5,7]; P1_REC=[0.258]*2+[0]*2+[0.258]*2+[0]*2
P1_EFF=[0.230,0.164,0,0,0.272,0.244,0,0]; P1_CTRL=[0.197,0,0.258,0]
P1_WORLD=f45['p1_NminusF']; P2_WORLD=f45['p2_NminusF']
X_OBS=[26,25,29,23,29,28,22,25]; X_PBC=[0.750,0.743,0.769,0.727,0.769,0.763,0.719,0.743]
X_PCB=[1-v for v in X_PBC]
X_EFF=[0.348,0.324,0.318,0.316,0.338,0.352,0.334,0.330]; X_CTRL=[0.336,0.317,0.345,0.332]
X_WORLD=f45['X_minus_N']
DEPTH={'O and X':[0.00125,0.02,0.31125,0.64875,0.01875,0],'F and N':[0.00125,0.9975,0,0.00125,0,0]}
UNC=[ # label, cond (m,lo,hi), run boot, ctrl boot, controller means
 ('Study 2, P1: N - F',(0.11375,0.10002,0.12748),(0.033,0.19825),(0.0,0.23325),[0.197,0,0.258,0]),
 ('Study 2, P2: N - F',(0,0,0),(0,0),(0,0),[0,0,0,0]),
 ('Study 3: X - N',(0.3325,0.31972,0.34528),(0.31575,0.3485),(0.3145,0.34925),X_CTRL)]
ES_P2_OBS=[8,8,1,2,8,5,5,3]; ES_P2_EFF=[0.174,0.130,0,0,0.142,0,0,0]
ES_P1_EFF=[0,0,0,0,-0.284,0,-0.446,0]
ES_MEANS={'P2':(0.05575,0.04797,0.06353),'P1':(-0.09125,-0.09992,-0.08258)}
REPAIRED={'P2':{'N':0,'ES':3},'P1':{'N':5,'ES':3}}
# comparator: (group, relation, name, mean, lo, hi, classes per run: 'r' repaired,'p' partial,'u' unrepaired)
COMP=[
 ('Study 4 seed sets','P1','Frontier (ES)',-0.09125,-0.09992,-0.08258,'rruuuruu'),
 ('Study 4 seed sets','P1','SEA-A',-0.22975,-0.24431,-0.21519,'uuuuppuu'),
 ('Comparator seed sets','P1','Random',-0.100,-0.1145,-0.0855,'uuuuuuuu'),
 ('Comparator seed sets','P1','Uncertainty',-0.100,-0.1145,-0.0855,'uuuuuuuu'),
 ('Comparator seed sets','P1','DI',-0.100,-0.1145,-0.0855,'uuuuuuuu'),
 ('Study 4 seed sets','P2','Frontier (ES)',0.05575,0.04797,0.06353,'ppuupuuu'),
 ('Study 4 seed sets','P2','SEA-A',-0.11425,-0.1254,-0.1031,'uuuuuuuu'),
 ('Comparator seed sets','P2','Random',0,0,0,'uuuuuuuu'),
 ('Comparator seed sets','P2','Uncertainty',0,0,0,'uuuuuuuu'),
 ('Comparator seed sets','P2','DI',0,0,0,'uuuuuuuu')]
SEL={'P1':{'pairs':['A/B','B/C','B/D','B/E','C/D','C/E'],
           'M':[[0.21,0.25,0.26,0.26,0.00,0.01],[0.25,0.21,0.27,0.25,0.00,0.01],[0.34,0.34,0.15,0.16,0.00,0.00]],
           'div':[('Random vs Uncertainty',0.05,306),('Random vs DI',0.60,72),('Uncertainty vs DI',0.54,69)],
           'obs':[6,4,0],'rep':[0,0,0]},
     'P2':{'pairs':['A/B','A/C','A/D','A/E','C/D','C/E'],
           'M':[[0.23,0.23,0.22,0.24,0.03,0.05],[0.12,0.25,0.31,0.30,0.02,0.01],[0.15,0.50,0.15,0.15,0.04,0.00]],
           'div':[('Random vs Uncertainty',0.12,196),('Random vs DI',0.64,74),('Uncertainty vs DI',0.70,69)],
           'obs':[0,0,0],'rep':[0,0,0]}}
