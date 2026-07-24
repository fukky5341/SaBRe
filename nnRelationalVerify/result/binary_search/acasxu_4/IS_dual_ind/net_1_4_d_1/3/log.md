## Execution arguments:
Dataset: Dataset.ACAS
Network: onnx/acasxu_op11/ACASXU_1_4.onnx
Epsilon: None
Initial delta epsilon: 1
Time budget: 1200 seconds
Threshold: 0.07398656999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144)
1: (-0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303)
2: (-0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905)
3: (-0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251)
4: (-0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503)

## BASE Result
execution time: IAR + LP analysis = 2.01 + 0.98 = 2.99 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0754965, upper bound: 0.0754965


# Binary Search by BASE starts (time budget: 1197.01 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate diff: 0.0909091


## IAR start
Binary search (step 0): status=Status.UNKNOWN, low=0.0000000, high=0.0909091, mid=0.0909091, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 1) starts
Candidate diff: 0.0454545


## IAR start
Binary search (step 1): status=Status.UNKNOWN, low=0.0000000, high=0.0454545, mid=0.0454545, abs_max=0.09181444346904755
rel_dist={0: [-0.074817289964232, 0.07481728996423198]}

## Binary search (step 2) starts
Candidate diff: 0.0227273


## IAR start
Binary search (step 2): status=Status.UNKNOWN, low=0.0000000, high=0.0227273, mid=0.0227273, abs_max=0.09181444346904755
rel_dist={0: [-0.07418591474870181, 0.07418591474870184]}

## Binary search (step 3) starts
Candidate diff: 0.0113636


## IAR start
Binary search (step 3): status=Status.VERIFIED, low=0.0113636, high=0.0227273, mid=0.0113636, abs_max=0.09181444346904755
rel_dist={0: [-0.07385188413801798, 0.07385188413801799]}

## Binary search (step 4) starts
Candidate diff: 0.0170455


## IAR start
Binary search (step 4): status=Status.UNKNOWN, low=0.0113636, high=0.0170455, mid=0.0170455, abs_max=0.09181444346904755
rel_dist={0: [-0.07401889944335989, 0.07401889944335993]}

## Binary search (step 5) starts
Candidate diff: 0.0142045


## IAR start
Binary search (step 5): status=Status.VERIFIED, low=0.0142045, high=0.0170455, mid=0.0142045, abs_max=0.09181444346904755
rel_dist={0: [-0.07393539179068895, 0.07393539179068898]}

## Binary search (step 6) starts
Candidate diff: 0.0156250


## IAR start
Binary search (step 6): status=Status.VERIFIED, low=0.0156250, high=0.0170455, mid=0.0156250, abs_max=0.09181444346904755
rel_dist={0: [-0.0739771456033364, 0.07397714560333646]}

## Binary search (step 7) starts
Candidate diff: 0.0163352


## IAR start
Binary search (step 7): status=Status.UNKNOWN, low=0.0156250, high=0.0163352, mid=0.0163352, abs_max=0.09181444346904755
rel_dist={0: [-0.07399802255072409, 0.07399802255072413]}

## Binary search (step 8) starts
Candidate diff: 0.0159801


## IAR start
Binary search (step 8): status=Status.UNKNOWN, low=0.0156250, high=0.0159801, mid=0.0159801, abs_max=0.09181444346904755
rel_dist={0: [-0.0739875840496543, 0.07398758404965433]}

## Binary search (step 9) starts
Candidate diff: 0.0158026


## IAR start
Binary search (step 9): status=Status.VERIFIED, low=0.0158026, high=0.0159801, mid=0.0158026, abs_max=0.09181444346904755
rel_dist={0: [-0.07398236482649538, 0.0739823648264954]}

## Binary search (step 10) starts
Candidate diff: 0.0158913


## IAR start
Binary search (step 10): status=Status.VERIFIED, low=0.0158913, high=0.0159801, mid=0.0158913, abs_max=0.09181444346904755
rel_dist={0: [-0.07398497446545078, 0.07398497446545083]}

## Binary search (step 11) starts
Candidate diff: 0.0159357


## IAR start
Binary search (step 11): status=Status.VERIFIED, low=0.0159357, high=0.0159801, mid=0.0159357, abs_max=0.09181444346904755
rel_dist={0: [-0.07398627925755258, 0.07398627925755258]}

## Binary search (step 12) starts
Candidate diff: 0.0159579


## IAR start
Binary search (step 12): status=Status.UNKNOWN, low=0.0159357, high=0.0159579, mid=0.0159579, abs_max=0.09181444346904755
rel_dist={0: [-0.07398693168097939, 0.07398693168097939]}

## Binary search (step 13) starts
Candidate diff: 0.0159468


## IAR start
Binary search (step 13): status=Status.UNKNOWN, low=0.0159357, high=0.0159468, mid=0.0159468, abs_max=0.09181444346904755
rel_dist={0: [-0.07398660546926598, 0.07398660546926597]}

## Binary search (step 14) starts
Candidate diff: 0.0159413


## IAR start
Binary search (step 14): status=Status.VERIFIED, low=0.0159413, high=0.0159468, mid=0.0159413, abs_max=0.09181444346904755
rel_dist={0: [-0.07398644236340933, 0.07398644236340929]}

## Binary search (step 15) starts
Candidate diff: 0.0159440


## IAR start
Binary search (step 15): status=Status.VERIFIED, low=0.0159440, high=0.0159468, mid=0.0159440, abs_max=0.09181444346904755
rel_dist={0: [-0.0739865239437136, 0.07398652394371358]}

## Binary search (step 16) starts
Candidate diff: 0.0159454


## IAR start
Binary search (step 16): status=Status.VERIFIED, low=0.0159454, high=0.0159468, mid=0.0159454, abs_max=0.09181444346904755
rel_dist={0: [-0.07398656467911387, 0.07398656467911383]}

## Binary search (step 17) starts
Candidate diff: 0.0159461


## IAR start
Binary search (step 17): status=Status.UNKNOWN, low=0.0159454, high=0.0159461, mid=0.0159461, abs_max=0.09181444346904755
rel_dist={0: [-0.07398658510156589, 0.07398658510156586]}

## Binary Search Result
Binary search time: 52.09 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01594543504552348


# Individual Split (IS_dual_ind) starts
Time budget: 1144.92 seconds

## Binary search (step 0) starts
Candidate diff: 0.0988818


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750495, upper bound: 0.0745360
time: 0.31 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0751743
time: 0.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.77 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.0750495, upper bound: 0.0745360
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.77
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0751743

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0096398, 0.0840788, -0.0122537, 0.0795023, -0.0891421, 0.0963325
1: -0.0483434, 0.1096082, -0.0481836, 0.1045535, -0.1528969, 0.1577918
2: -0.0175393, 0.1624446, -0.0215126, 0.1537826, -0.1713219, 0.1839572
3: -0.1008122, 0.1153759, -0.0898233, 0.1119772, -0.2127894, 0.2051992
4: -0.0483946, 0.1864471, -0.0514254, 0.1716636, -0.2200582, 0.2378725

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0743580, upper bound: 0.0743580
time: 0.29 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0743580, upper bound: 0.0745360
time: 0.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0080926, 0.0739747, -0.0122710, 0.0795435, -0.0876361, 0.0862456
1: -0.0402663, 0.0988584, -0.0482301, 0.1046001, -0.1448664, 0.1470885
2: -0.0162963, 0.1430283, -0.0215323, 0.1538582, -0.1701545, 0.1645606
3: -0.0793115, 0.1036161, -0.0898915, 0.1120336, -0.1913451, 0.1935076
4: -0.0462731, 0.1545721, -0.0514701, 0.1717802, -0.2180533, 0.2060422

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0745360, upper bound: 0.0749940
time: 0.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0745360, upper bound: 0.0749940
time: 0.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.62 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.62
Output dim: 0, lower bound: -0.0743580, upper bound: 0.0743580
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.62
Output dim: 0, lower bound: -0.0743580, upper bound: 0.0745360
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.62
Output dim: 0, lower bound: -0.0745360, upper bound: 0.0749940
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.62
Output dim: 0, lower bound: -0.0745360, upper bound: 0.0749940

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0096398, 0.0840788, -0.0096398, 0.0840788, -0.0937186, 0.0937186
1: -0.0483434, 0.1096082, -0.0483434, 0.1096082, -0.1579516, 0.1579516
2: -0.0175393, 0.1624446, -0.0175393, 0.1624446, -0.1799838, 0.1799838
3: -0.1008122, 0.1153759, -0.1008122, 0.1153759, -0.2161881, 0.2161881
4: -0.0483946, 0.1864471, -0.0483946, 0.1864471, -0.2348417, 0.2348417

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741097, upper bound: 0.0725347
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0729613
time: 0.29 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0096398, 0.0840788, -0.0080926, 0.0739747, -0.0836145, 0.0921715
1: -0.0483434, 0.1096082, -0.0402663, 0.0988584, -0.1472017, 0.1498745
2: -0.0175393, 0.1624446, -0.0162963, 0.1430283, -0.1605676, 0.1787409
3: -0.1008122, 0.1153759, -0.0793115, 0.1036161, -0.2044283, 0.1946874
4: -0.0483946, 0.1864471, -0.0462731, 0.1545721, -0.2029667, 0.2327202

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741097, upper bound: 0.0732822
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0741465
time: 0.32 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0080926, 0.0739747, -0.0096398, 0.0840788, -0.0921715, 0.0836145
1: -0.0402663, 0.0988584, -0.0483434, 0.1096082, -0.1498745, 0.1472017
2: -0.0162963, 0.1430283, -0.0175393, 0.1624446, -0.1787409, 0.1605676
3: -0.0793115, 0.1036161, -0.1008122, 0.1153759, -0.1946874, 0.2044283
4: -0.0462731, 0.1545721, -0.0483946, 0.1864471, -0.2327202, 0.2029667

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0720345
time: 0.29 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0739891
time: 0.30 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0080926, 0.0739747, -0.0080926, 0.0739747, -0.0820673, 0.0820673
1: -0.0402663, 0.0988584, -0.0402663, 0.0988584, -0.1391247, 0.1391247
2: -0.0162963, 0.1430283, -0.0162963, 0.1430283, -0.1593246, 0.1593246
3: -0.0793115, 0.1036161, -0.0793115, 0.1036161, -0.1829276, 0.1829276
4: -0.0462731, 0.1545721, -0.0462731, 0.1545721, -0.2008452, 0.2008452

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0725892
time: 0.32 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0751743
time: 0.32 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.63 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -0.0741097, upper bound: 0.0725347
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.63
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0729613
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -0.0741097, upper bound: 0.0732822
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0741465
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0720345
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0739891
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0725892
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0751743

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0031931, 0.0596317, -0.0096398, 0.0840788, -0.0808857, 0.0692715
1: -0.0159592, 0.0935836, -0.0483434, 0.1096082, -0.1255674, 0.1419269
2: -0.0004775, 0.1070566, -0.0175393, 0.1624446, -0.1629221, 0.1245959
3: -0.0561146, 0.0980010, -0.1008122, 0.1153759, -0.1714906, 0.1988132
4: -0.0290360, 0.1065627, -0.0483946, 0.1864471, -0.2154831, 0.1549573

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725347, upper bound: 0.0725347
time: 0.30 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725347, upper bound: 0.0725347
time: 0.30 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0031931, 0.0596317, -0.0080926, 0.0739747, -0.0707816, 0.0677244
1: -0.0159592, 0.0935836, -0.0402663, 0.0988584, -0.1148175, 0.1338498
2: -0.0004775, 0.1070566, -0.0162963, 0.1430283, -0.1435058, 0.1233529
3: -0.0561146, 0.0980010, -0.0793115, 0.1036161, -0.1597307, 0.1773125
4: -0.0290360, 0.1065627, -0.0462731, 0.1545721, -0.1836081, 0.1528358

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0732821
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0732821
time: 0.32 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0056816, 0.0768258, -0.0080926, 0.0739747, -0.0796563, 0.0849185
1: -0.0367180, 0.1016865, -0.0402663, 0.0988584, -0.1355763, 0.1419528
2: -0.0131781, 0.1492988, -0.0162963, 0.1430283, -0.1562063, 0.1655951
3: -0.0882968, 0.1038739, -0.0793115, 0.1036161, -0.1919129, 0.1831854
4: -0.0414514, 0.1655897, -0.0462731, 0.1545721, -0.1960235, 0.2118628

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0741465
time: 0.30 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0741465
time: 0.30 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0083338, 0.0470844, -0.0096398, 0.0840788, -0.0757450, 0.0567242
1: -0.0002709, 0.0768870, -0.0483434, 0.1096082, -0.1098791, 0.1252303
2: 0.0037108, 0.0803992, -0.0175393, 0.1624446, -0.1587338, 0.0979384
3: -0.0257773, 0.0755912, -0.1008122, 0.1153759, -0.1411532, 0.1764034
4: -0.0258763, 0.0645177, -0.0483946, 0.1864471, -0.2123234, 0.1129123

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732821, upper bound: 0.0720345
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732821, upper bound: 0.0720345
time: 0.34 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0096398, 0.0840788, -0.0890281, 0.0780929
1: -0.0301443, 0.0939154, -0.0483434, 0.1096082, -0.1397525, 0.1422587
2: -0.0130560, 0.1317879, -0.0175393, 0.1624446, -0.1755006, 0.1493272
3: -0.0685425, 0.0983886, -0.1008122, 0.1153759, -0.1839184, 0.1992007
4: -0.0412280, 0.1366114, -0.0483946, 0.1864471, -0.2276751, 0.1850060

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732821, upper bound: 0.0739891
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732821, upper bound: 0.0739891
time: 0.31 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0083338, 0.0470844, -0.0080926, 0.0739747, -0.0656408, 0.0551770
1: -0.0002709, 0.0768870, -0.0402663, 0.0988584, -0.0991293, 0.1171533
2: 0.0037108, 0.0803992, -0.0162963, 0.1430283, -0.1393175, 0.0966955
3: -0.0257773, 0.0755912, -0.0793115, 0.1036161, -0.1293934, 0.1549027
4: -0.0258763, 0.0645177, -0.0462731, 0.1545721, -0.1804484, 0.1107908

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0725892
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0725892
time: 0.32 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0080926, 0.0739747, -0.0789240, 0.0765457
1: -0.0301443, 0.0939154, -0.0402663, 0.0988584, -0.1290027, 0.1341817
2: -0.0130560, 0.1317879, -0.0162963, 0.1430283, -0.1560843, 0.1480842
3: -0.0685425, 0.0983886, -0.0793115, 0.1036161, -0.1721586, 0.1777001
4: -0.0412280, 0.1366114, -0.0462731, 0.1545721, -0.1958002, 0.1828845

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0751743
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0751743
time: 0.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.68 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0725347, upper bound: 0.0725347
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0725347, upper bound: 0.0725347
IS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0732821
IS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0732821
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0741465
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0741465
IS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0732821, upper bound: 0.0720345
IS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0732821, upper bound: 0.0720345
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0732821, upper bound: 0.0739891
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0732821, upper bound: 0.0739891
IS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0725892
IS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0725892
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0751743
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.68
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0751743

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0056816, 0.0768258, 0.0083338, 0.0470844, -0.0527660, 0.0684920
1: -0.0367180, 0.1016865, -0.0002709, 0.0768870, -0.1136049, 0.1019575
2: -0.0131781, 0.1492988, 0.0037108, 0.0803992, -0.0935772, 0.1455880
3: -0.0882968, 0.1038739, -0.0257773, 0.0755912, -0.1638880, 0.1296511
4: -0.0414514, 0.1655897, -0.0258763, 0.0645177, -0.1059691, 0.1914660

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 47

Time for candidate selection: 0.68 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0704852, upper bound: 0.0729268
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0726668
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0056816, 0.0768258, -0.0049493, 0.0684531, -0.0741346, 0.0817752
1: -0.0367180, 0.1016865, -0.0301443, 0.0939154, -0.1306333, 0.1318308
2: -0.0131781, 0.1492988, -0.0130560, 0.1317879, -0.1449660, 0.1623548
3: -0.0882968, 0.1038739, -0.0685425, 0.0983886, -0.1866854, 0.1724163
4: -0.0414514, 0.1655897, -0.0412280, 0.1366114, -0.1780628, 0.2068177

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 47

Time for candidate selection: 0.68 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0704852, upper bound: 0.0729268
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0726668
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
time: 0.32 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, 0.0031931, 0.0596317, -0.0645811, 0.0652600
1: -0.0301443, 0.0939154, -0.0159592, 0.0935836, -0.1237278, 0.1098745
2: -0.0130560, 0.1317879, -0.0004775, 0.1070566, -0.1201126, 0.1322655
3: -0.0685425, 0.0983886, -0.0561146, 0.0980010, -0.1665435, 0.1545032
4: -0.0412280, 0.1366114, -0.0290360, 0.1065627, -0.1477907, 0.1656474

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48

Time for candidate selection: 0.72 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0715099, upper bound: 0.0738147
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731685, upper bound: 0.0737314
time: 0.35 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0056816, 0.0768258, -0.0817752, 0.0741346
1: -0.0301443, 0.0939154, -0.0367180, 0.1016865, -0.1318308, 0.1306333
2: -0.0130560, 0.1317879, -0.0131781, 0.1492988, -0.1623548, 0.1449660
3: -0.0685425, 0.0983886, -0.0882968, 0.1038739, -0.1724163, 0.1866854
4: -0.0412280, 0.1366114, -0.0414514, 0.1655897, -0.2068177, 0.1780628

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48

Time for candidate selection: 0.69 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0715099, upper bound: 0.0738147
time: 0.33 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731685, upper bound: 0.0737314
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, 0.0083338, 0.0470844, -0.0520337, 0.0601192
1: -0.0301443, 0.0939154, -0.0002709, 0.0768870, -0.1070312, 0.0941863
2: -0.0130560, 0.1317879, 0.0037108, 0.0803992, -0.0934552, 0.1280771
3: -0.0685425, 0.0983886, -0.0257773, 0.0755912, -0.1441337, 0.1241658
4: -0.0412280, 0.1366114, -0.0258763, 0.0645177, -0.1057457, 0.1624877

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48

Time for candidate selection: 0.69 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0725221
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0049493, 0.0684531, -0.0734024, 0.0734024
1: -0.0301443, 0.0939154, -0.0301443, 0.0939154, -0.1240597, 0.1240597
2: -0.0130560, 0.1317879, -0.0130560, 0.1317879, -0.1448440, 0.1448440
3: -0.0685425, 0.0983886, -0.0685425, 0.0983886, -0.1669310, 0.1669310
4: -0.0412280, 0.1366114, -0.0412280, 0.1366114, -0.1778394, 0.1778394

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48

Time for candidate selection: 0.68 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0725221
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
time: 0.33 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.28 seconds
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0726668
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0726668
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 0, lower bound: -0.0715099, upper bound: 0.0738147
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 0, lower bound: -0.0731685, upper bound: 0.0737314
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 0, lower bound: -0.0715099, upper bound: 0.0738147
IS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 0, lower bound: -0.0731685, upper bound: 0.0737314
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0725221
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, 0.0083338, 0.0470844, -0.0517736, 0.0668406
1: -0.0346557, 0.1001197, -0.0002709, 0.0768870, -0.1115426, 0.1003906
2: -0.0124673, 0.1463497, 0.0037108, 0.0803992, -0.0928665, 0.1426390
3: -0.0855657, 0.1017813, -0.0257773, 0.0755912, -0.1611568, 0.1275585
4: -0.0402376, 0.1609547, -0.0258763, 0.0645177, -0.1047553, 0.1868310

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.71 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0726245
time: 0.31 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0740310
time: 0.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, -0.0049493, 0.0684531, -0.0731423, 0.0801238
1: -0.0346557, 0.1001197, -0.0301443, 0.0939154, -0.1285711, 0.1302640
2: -0.0124673, 0.1463497, -0.0130560, 0.1317879, -0.1442553, 0.1594058
3: -0.0855657, 0.1017813, -0.0685425, 0.0983886, -0.1839542, 0.1703237
4: -0.0402376, 0.1609547, -0.0412280, 0.1366114, -0.1768490, 0.2021827

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48

Time for candidate selection: 0.71 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712917, upper bound: 0.0717108
time: 0.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0712917, upper bound: 0.0740311
time: 0.33 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0083338, 0.0470844, -0.0529624, 0.0536615
1: -0.0304515, 0.0843779, -0.0002709, 0.0768870, -0.1073385, 0.0846488
2: -0.0112683, 0.1210694, 0.0037108, 0.0803992, -0.0916675, 0.1173586
3: -0.0502359, 0.0748101, -0.0257773, 0.0755912, -0.1258271, 0.1005873
4: -0.0342595, 0.1180273, -0.0258763, 0.0645177, -0.0987772, 0.1439036

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.70 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0049493, 0.0684531, -0.0743310, 0.0669446
1: -0.0304515, 0.0843779, -0.0301443, 0.0939154, -0.1243669, 0.1145222
2: -0.0112683, 0.1210694, -0.0130560, 0.1317879, -0.1430563, 0.1341254
3: -0.0502359, 0.0748101, -0.0685425, 0.0983886, -0.1486244, 0.1433525
4: -0.0342595, 0.1180273, -0.0412280, 0.1366114, -0.1708709, 0.1592553

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48

Time for candidate selection: 0.71 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0748467
time: 0.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.29 seconds
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0726245
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0740310
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.0712917, upper bound: 0.0717108
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.0712917, upper bound: 0.0740311
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.29
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0748467

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, 0.0089527, 0.0463036, -0.0509928, 0.0662218
1: -0.0346557, 0.1001197, 0.0023787, 0.0763386, -0.1109943, 0.0977410
2: -0.0124673, 0.1463497, 0.0041397, 0.0773552, -0.0898226, 0.1422101
3: -0.0855657, 0.1017813, -0.0231655, 0.0750298, -0.1605954, 0.1249468
4: -0.0402376, 0.1609547, -0.0254443, 0.0596935, -0.0999311, 0.1863990

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 47

Time for candidate selection: 0.69 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0727707
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
time: 0.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, -0.0039670, 0.0673659, -0.0720552, 0.0791415
1: -0.0346557, 0.1001197, -0.0276524, 0.0931255, -0.1277812, 0.1277721
2: -0.0124673, 0.1463497, -0.0123972, 0.1293659, -0.1418332, 0.1587470
3: -0.0855657, 0.1017813, -0.0660703, 0.0977019, -0.1832676, 0.1678516
4: -0.0402376, 0.1609547, -0.0406218, 0.1327123, -0.1729499, 0.2015765

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 47

Time for candidate selection: 0.71 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702994, upper bound: 0.0726668
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740311
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.69 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.71 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.73 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.72 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467
time: 0.37 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.37 seconds
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0727707
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0702994, upper bound: 0.0726668
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740311
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.37
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, 0.0089527, 0.0463036, -0.0509928, 0.0662218
1: -0.0346557, 0.1001197, 0.0023787, 0.0763386, -0.1109943, 0.0977410
2: -0.0124673, 0.1463497, 0.0041397, 0.0773552, -0.0898226, 0.1422101
3: -0.0855657, 0.1017813, -0.0231655, 0.0750298, -0.1605954, 0.1249468
4: -0.0402376, 0.1609547, -0.0254443, 0.0596935, -0.0999311, 0.1863990

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.81 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0726245
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, -0.0039670, 0.0673659, -0.0720552, 0.0791415
1: -0.0346557, 0.1001197, -0.0276524, 0.0931255, -0.1277812, 0.1277721
2: -0.0124673, 0.1463497, -0.0123972, 0.1293659, -0.1418332, 0.1587470
3: -0.0855657, 0.1017813, -0.0660703, 0.0977019, -0.1832676, 0.1678516
4: -0.0402376, 0.1609547, -0.0406218, 0.1327123, -0.1729499, 0.2015765

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48

Time for candidate selection: 0.72 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0717108
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740310
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.73 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722226, upper bound: 0.0748239
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.74 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
time: 0.34 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.73 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467
time: 0.37 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.46 seconds
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.46
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0726245
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.46
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
IS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.46
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0717108
IS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.46
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740310
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.46
Output dim: 0, lower bound: -0.0722226, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.46
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.46
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.46
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.46
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.46
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.46
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.46
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, 0.0089527, 0.0463036, -0.0509928, 0.0662218
1: -0.0346557, 0.1001197, 0.0023787, 0.0763386, -0.1109943, 0.0977410
2: -0.0124673, 0.1463497, 0.0041397, 0.0773552, -0.0898226, 0.1422101
3: -0.0855657, 0.1017813, -0.0231655, 0.0750298, -0.1605954, 0.1249468
4: -0.0402376, 0.1609547, -0.0254443, 0.0596935, -0.0999311, 0.1863990

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 47

Time for candidate selection: 0.73 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0727707
time: 0.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, -0.0039670, 0.0673659, -0.0720552, 0.0791415
1: -0.0346557, 0.1001197, -0.0276524, 0.0931255, -0.1277812, 0.1277721
2: -0.0124673, 0.1463497, -0.0123972, 0.1293659, -0.1418332, 0.1587470
3: -0.0855657, 0.1017813, -0.0660703, 0.0977019, -0.1832676, 0.1678516
4: -0.0402376, 0.1609547, -0.0406218, 0.1327123, -0.1729499, 0.2015765

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 47

Time for candidate selection: 0.74 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702994, upper bound: 0.0726668
time: 0.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740311
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.75 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.78 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.77 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.77 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.75 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467
time: 0.37 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 3.56 seconds
IS_A1_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0727707
IS_A1_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
IS_A1_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0702994, upper bound: 0.0726668
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740311
IS_A2_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
IS_A2_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
IS_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
IS_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
IS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, 0.0089527, 0.0463036, -0.0509928, 0.0662218
1: -0.0346557, 0.1001197, 0.0023787, 0.0763386, -0.1109943, 0.0977410
2: -0.0124673, 0.1463497, 0.0041397, 0.0773552, -0.0898226, 0.1422101
3: -0.0855657, 0.1017813, -0.0231655, 0.0750298, -0.1605954, 0.1249468
4: -0.0402376, 0.1609547, -0.0254443, 0.0596935, -0.0999311, 0.1863990

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0726245
time: 0.33 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
time: 0.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, -0.0039670, 0.0673659, -0.0720552, 0.0791415
1: -0.0346557, 0.1001197, -0.0276524, 0.0931255, -0.1277812, 0.1277721
2: -0.0124673, 0.1463497, -0.0123972, 0.1293659, -0.1418332, 0.1587470
3: -0.0855657, 0.1017813, -0.0660703, 0.0977019, -0.1832676, 0.1678516
4: -0.0402376, 0.1609547, -0.0406218, 0.1327123, -0.1729499, 0.2015765

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0717108
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740310
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722226, upper bound: 0.0748239
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722226, upper bound: 0.0748239
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.78 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.80 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.77 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.80 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467
time: 0.44 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 3.77 seconds
IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0726245
IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0717108
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740310
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0722226, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0722226, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.77
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, 0.0089527, 0.0463036, -0.0509928, 0.0662218
1: -0.0346557, 0.1001197, 0.0023787, 0.0763386, -0.1109943, 0.0977410
2: -0.0124673, 0.1463497, 0.0041397, 0.0773552, -0.0898226, 0.1422101
3: -0.0855657, 0.1017813, -0.0231655, 0.0750298, -0.1605954, 0.1249468
4: -0.0402376, 0.1609547, -0.0254443, 0.0596935, -0.0999311, 0.1863990

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 47

Time for candidate selection: 0.78 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0727707
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, -0.0039670, 0.0673659, -0.0720552, 0.0791415
1: -0.0346557, 0.1001197, -0.0276524, 0.0931255, -0.1277812, 0.1277721
2: -0.0124673, 0.1463497, -0.0123972, 0.1293659, -0.1418332, 0.1587470
3: -0.0855657, 0.1017813, -0.0660703, 0.0977019, -0.1832676, 0.1678516
4: -0.0402376, 0.1609547, -0.0406218, 0.1327123, -0.1729499, 0.2015765

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 47

Time for candidate selection: 0.79 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702994, upper bound: 0.0726668
time: 0.42 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740311
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.81 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.80 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.80 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.78 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.77 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.80 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.81 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.87 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.80 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.81 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058174, 0.0608798, -0.0667578, 0.0678127
1: -0.0304515, 0.0843779, -0.0302412, 0.0840455, -0.1144970, 0.1146191
2: -0.0112683, 0.1210694, -0.0112275, 0.1192306, -0.1304989, 0.1322969
3: -0.0502359, 0.0748101, -0.0482706, 0.0746668, -0.1249026, 0.1230807
4: -0.0342595, 0.1180273, -0.0342433, 0.1149402, -0.1491997, 0.1522706

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.82 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.84 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467
time: 0.44 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 3.92 seconds
IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0727707
IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0702994, upper bound: 0.0726668
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740311
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745387
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.92
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, 0.0089527, 0.0463036, -0.0509928, 0.0662218
1: -0.0346557, 0.1001197, 0.0023787, 0.0763386, -0.1109943, 0.0977410
2: -0.0124673, 0.1463497, 0.0041397, 0.0773552, -0.0898226, 0.1422101
3: -0.0855657, 0.1017813, -0.0231655, 0.0750298, -0.1605954, 0.1249468
4: -0.0402376, 0.1609547, -0.0254443, 0.0596935, -0.0999311, 0.1863990

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.85 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0726245
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
time: 0.43 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, -0.0039670, 0.0673659, -0.0720552, 0.0791415
1: -0.0346557, 0.1001197, -0.0276524, 0.0931255, -0.1277812, 0.1277721
2: -0.0124673, 0.1463497, -0.0123972, 0.1293659, -0.1418332, 0.1587470
3: -0.0855657, 0.1017813, -0.0660703, 0.0977019, -0.1832676, 0.1678516
4: -0.0402376, 0.1609547, -0.0406218, 0.1327123, -0.1729499, 0.2015765

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48

Time for candidate selection: 0.80 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0717108
time: 0.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740310
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.81 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722226, upper bound: 0.0748239
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.85 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.84 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722226, upper bound: 0.0748239
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.82 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.81 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722226, upper bound: 0.0748239
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.84 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.83 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722226, upper bound: 0.0748239
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.83 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.80 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.82 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.82 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.84 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.81 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058174, 0.0608798, -0.0667578, 0.0678127
1: -0.0304515, 0.0843779, -0.0302412, 0.0840455, -0.1144970, 0.1146191
2: -0.0112683, 0.1210694, -0.0112275, 0.1192306, -0.1304989, 0.1322969
3: -0.0502359, 0.0748101, -0.0482706, 0.0746668, -0.1249026, 0.1230807
4: -0.0342595, 0.1180273, -0.0342433, 0.1149402, -0.1491997, 0.1522706

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.82 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.82 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467
time: 0.44 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 3.97 seconds
IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0726245
IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0717108
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740310
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0722226, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0722226, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0722226, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0722226, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.97
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, 0.0089527, 0.0463036, -0.0509928, 0.0662218
1: -0.0346557, 0.1001197, 0.0023787, 0.0763386, -0.1109943, 0.0977410
2: -0.0124673, 0.1463497, 0.0041397, 0.0773552, -0.0898226, 0.1422101
3: -0.0855657, 0.1017813, -0.0231655, 0.0750298, -0.1605954, 0.1249468
4: -0.0402376, 0.1609547, -0.0254443, 0.0596935, -0.0999311, 0.1863990

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 47

Time for candidate selection: 0.82 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0727707
time: 0.39 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
time: 0.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0046892, 0.0751745, -0.0039670, 0.0673659, -0.0720552, 0.0791415
1: -0.0346557, 0.1001197, -0.0276524, 0.0931255, -0.1277812, 0.1277721
2: -0.0124673, 0.1463497, -0.0123972, 0.1293659, -0.1418332, 0.1587470
3: -0.0855657, 0.1017813, -0.0660703, 0.0977019, -0.1832676, 0.1678516
4: -0.0402376, 0.1609547, -0.0406218, 0.1327123, -0.1729499, 0.2015765

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 47

Time for candidate selection: 0.84 seconds

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702994, upper bound: 0.0726668
time: 0.38 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740311
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.84 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.92 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.82 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.83 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.88 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.88 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.87 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.88 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.88 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.95 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
time: 0.49 seconds

## Summary of splitting at layer (split count: 12)
- Time for IS candidates: 4.32 seconds
IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0727707
IS_A1_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0702994, upper bound: 0.0726668
IS_A1_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740311
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0729333
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0748239
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0722851, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.32
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467
Binary search (step 0): status=Status.UNKNOWN, low=0.0159454, high=0.0988818, mid=0.0988818, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 1) starts
Candidate diff: 0.0574136


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0745848, upper bound: 0.0740205
time: 0.31 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0748532, upper bound: 0.0748532
time: 0.33 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.83 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.0745848, upper bound: 0.0740205
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.83
Output dim: 0, lower bound: -0.0748532, upper bound: 0.0748532

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0096398, 0.0840788, -0.0109180, 0.0766336, -0.0862734, 0.0949968
1: -0.0483434, 0.1096082, -0.0445345, 0.1015407, -0.1498840, 0.1541427
2: -0.0175393, 0.1624446, -0.0199811, 0.1481837, -0.1657229, 0.1824257
3: -0.1008122, 0.1153759, -0.0846509, 0.1086355, -0.2094477, 0.2000268
4: -0.0483946, 0.1864471, -0.0485491, 0.1629775, -0.2113721, 0.2349962

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0735765
time: 0.39 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0738308, upper bound: 0.0735773
time: 0.38 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0080926, 0.0739747, -0.0122710, 0.0795435, -0.0876361, 0.0862456
1: -0.0402663, 0.0988584, -0.0482301, 0.1046001, -0.1448664, 0.1470885
2: -0.0162963, 0.1430283, -0.0215323, 0.1538582, -0.1701545, 0.1645606
3: -0.0793115, 0.1036161, -0.0898915, 0.1120336, -0.1913451, 0.1935076
4: -0.0462731, 0.1545721, -0.0514701, 0.1717802, -0.2180533, 0.2060422

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740205, upper bound: 0.0745753
time: 0.30 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740205, upper bound: 0.0748532
time: 0.30 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.89 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.89
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0735765
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.89
Output dim: 0, lower bound: -0.0738308, upper bound: 0.0735773
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.89
Output dim: 0, lower bound: -0.0740205, upper bound: 0.0745753
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.89
Output dim: 0, lower bound: -0.0740205, upper bound: 0.0748532

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0080926, 0.0739747, -0.0096398, 0.0840788, -0.0921715, 0.0836145
1: -0.0402663, 0.0988584, -0.0483434, 0.1096082, -0.1498745, 0.1472017
2: -0.0162963, 0.1430283, -0.0175393, 0.1624446, -0.1787409, 0.1605676
3: -0.0793115, 0.1036161, -0.1008122, 0.1153759, -0.1946874, 0.2044283
4: -0.0462731, 0.1545721, -0.0483946, 0.1864471, -0.2327202, 0.2029667

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0735765, upper bound: 0.0720345
time: 0.32 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0735773, upper bound: 0.0738308
time: 0.32 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0080926, 0.0739747, -0.0080926, 0.0739747, -0.0820673, 0.0820673
1: -0.0402663, 0.0988584, -0.0402663, 0.0988584, -0.1391247, 0.1391247
2: -0.0162963, 0.1430283, -0.0162963, 0.1430283, -0.1593246, 0.1593246
3: -0.0793115, 0.1036161, -0.0793115, 0.1036161, -0.1829276, 0.1829276
4: -0.0462731, 0.1545721, -0.0462731, 0.1545721, -0.2008452, 0.2008452

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0735765, upper bound: 0.0725892
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0735773, upper bound: 0.0748532
time: 0.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.70 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.70
Output dim: 0, lower bound: -0.0735765, upper bound: 0.0720345
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.70
Output dim: 0, lower bound: -0.0735773, upper bound: 0.0738308
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.70
Output dim: 0, lower bound: -0.0735765, upper bound: 0.0725892
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 0, lower bound: -0.0735773, upper bound: 0.0748532

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0080926, 0.0739747, -0.0789240, 0.0765457
1: -0.0301443, 0.0939154, -0.0402663, 0.0988584, -0.1290027, 0.1341817
2: -0.0130560, 0.1317879, -0.0162963, 0.1430283, -0.1560843, 0.1480842
3: -0.0685425, 0.0983886, -0.0793115, 0.1036161, -0.1721586, 0.1777001
4: -0.0412280, 0.1366114, -0.0462731, 0.1545721, -0.1958002, 0.1828845

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731040, upper bound: 0.0748532
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731040, upper bound: 0.0748532
time: 0.34 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.74 seconds
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731040, upper bound: 0.0748532
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.74
Output dim: 0, lower bound: -0.0731040, upper bound: 0.0748532

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, 0.0083338, 0.0470844, -0.0520337, 0.0601192
1: -0.0301443, 0.0939154, -0.0002709, 0.0768870, -0.1070312, 0.0941863
2: -0.0130560, 0.1317879, 0.0037108, 0.0803992, -0.0934552, 0.1280771
3: -0.0685425, 0.0983886, -0.0257773, 0.0755912, -0.1441337, 0.1241658
4: -0.0412280, 0.1366114, -0.0258763, 0.0645177, -0.1057457, 0.1624877

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48

Time for candidate selection: 0.69 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0722994
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0049493, 0.0684531, -0.0734024, 0.0734024
1: -0.0301443, 0.0939154, -0.0301443, 0.0939154, -0.1240597, 0.1240597
2: -0.0130560, 0.1317879, -0.0130560, 0.1317879, -0.1448440, 0.1448440
3: -0.0685425, 0.0983886, -0.0685425, 0.0983886, -0.1669310, 0.1669310
4: -0.0412280, 0.1366114, -0.0412280, 0.1366114, -0.1778394, 0.1778394

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48

Time for candidate selection: 0.71 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0722994
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
time: 0.39 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.43 seconds
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.43
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0722994
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.43
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.43
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0083338, 0.0470844, -0.0529624, 0.0536615
1: -0.0304515, 0.0843779, -0.0002709, 0.0768870, -0.1073385, 0.0846488
2: -0.0112683, 0.1210694, 0.0037108, 0.0803992, -0.0916675, 0.1173586
3: -0.0502359, 0.0748101, -0.0257773, 0.0755912, -0.1258271, 0.1005873
4: -0.0342595, 0.1180273, -0.0258763, 0.0645177, -0.0987772, 0.1439036

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.72 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.34 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.35 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0042697, 0.0674990, -0.0733770, 0.0662650
1: -0.0304515, 0.0843779, -0.0295562, 0.0923321, -0.1227836, 0.1139341
2: -0.0112683, 0.1210694, -0.0122676, 0.1304387, -0.1417070, 0.1333370
3: -0.0502359, 0.0748101, -0.0668858, 0.0940056, -0.1442414, 0.1416959
4: -0.0342595, 0.1180273, -0.0402565, 0.1348467, -0.1691062, 0.1582838

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48

Time for candidate selection: 0.72 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745673
time: 0.39 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.46 seconds
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.46
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745673

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.74 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.73 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.72 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.73 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0745673
time: 0.40 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.53 seconds
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.53
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.53
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0745673

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.73 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.77 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0745673
time: 0.44 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.62 seconds
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0745673

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.73 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.74 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.75 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.73 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.75 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.74 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.74 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0745673
time: 0.38 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 3.54 seconds
IS_A2_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
IS_A2_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.54
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0745673

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.72 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.36 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.77 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.74 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.74 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.75 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.77 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0745673
time: 0.45 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 3.72 seconds
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.72
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0745673

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.77 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.78 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.78 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.78 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.78 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.78 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.80 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058174, 0.0608798, -0.0667578, 0.0678127
1: -0.0304515, 0.0843779, -0.0302412, 0.0840455, -0.1144970, 0.1146191
2: -0.0112683, 0.1210694, -0.0112275, 0.1192306, -0.1304989, 0.1322969
3: -0.0502359, 0.0748101, -0.0482706, 0.0746668, -0.1249026, 0.1230807
4: -0.0342595, 0.1180273, -0.0342433, 0.1149402, -0.1491997, 0.1522706

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.80 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.81 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0745673
time: 0.38 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 3.81 seconds
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.81
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0745673

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.81 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.40 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.80 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.82 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.83 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.85 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.84 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.83 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.84 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.83 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.82 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.39 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.84 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.90 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.89 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058174, 0.0608798, -0.0667578, 0.0678127
1: -0.0304515, 0.0843779, -0.0302412, 0.0840455, -0.1144970, 0.1146191
2: -0.0112683, 0.1210694, -0.0112275, 0.1192306, -0.1304989, 0.1322969
3: -0.0502359, 0.0748101, -0.0482706, 0.0746668, -0.1249026, 0.1230807
4: -0.0342595, 0.1180273, -0.0342433, 0.1149402, -0.1491997, 0.1522706

Time for backsubstitution: 2.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.88 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.85 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.85 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0745673
time: 0.49 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 4.06 seconds
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 4.06
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0745673

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.85 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.85 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.87 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.88 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.88 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.91 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.93 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.92 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.91 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.93 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
time: 0.48 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.92 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.93 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.92 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.93 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.93 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058174, 0.0608798, -0.0667578, 0.0678127
1: -0.0304515, 0.0843779, -0.0302412, 0.0840455, -0.1144970, 0.1146191
2: -0.0112683, 0.1210694, -0.0112275, 0.1192306, -0.1304989, 0.1322969
3: -0.0502359, 0.0748101, -0.0482706, 0.0746668, -0.1249026, 0.1230807
4: -0.0342595, 0.1180273, -0.0342433, 0.1149402, -0.1491997, 0.1522706

Time for backsubstitution: 2.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.93 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.92 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.92 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.43 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.93 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.43 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058174, 0.0608798, -0.0667578, 0.0678127
1: -0.0304515, 0.0843779, -0.0302412, 0.0840455, -0.1144970, 0.1146191
2: -0.0112683, 0.1210694, -0.0112275, 0.1192306, -0.1304989, 0.1322969
3: -0.0502359, 0.0748101, -0.0482706, 0.0746668, -0.1249026, 0.1230807
4: -0.0342595, 0.1180273, -0.0342433, 0.1149402, -0.1491997, 0.1522706

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.87 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.90 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058174, 0.0608798, -0.0667578, 0.0678127
1: -0.0304515, 0.0843779, -0.0302412, 0.0840455, -0.1144970, 0.1146191
2: -0.0112683, 0.1210694, -0.0112275, 0.1192306, -0.1304989, 0.1322969
3: -0.0502359, 0.0748101, -0.0482706, 0.0746668, -0.1249026, 0.1230807
4: -0.0342595, 0.1180273, -0.0342433, 0.1149402, -0.1491997, 0.1522706

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.90 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.90 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058174, 0.0608798, -0.0667578, 0.0678127
1: -0.0304515, 0.0843779, -0.0302412, 0.0840455, -0.1144970, 0.1146191
2: -0.0112683, 0.1210694, -0.0112275, 0.1192306, -0.1304989, 0.1322969
3: -0.0502359, 0.0748101, -0.0482706, 0.0746668, -0.1249026, 0.1230807
4: -0.0342595, 0.1180273, -0.0342433, 0.1149402, -0.1491997, 0.1522706

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.91 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.90 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.92 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0745673
time: 0.45 seconds

## Summary of splitting at layer (split count: 12)
- Time for IS candidates: 4.27 seconds
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0724970
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0745673
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0719047, upper bound: 0.0745356
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0722994
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 4.27
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0745673

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0097754, 0.0448622, -0.0507401, 0.0522199
1: -0.0304515, 0.0843779, 0.0012435, 0.0714976, -0.1019491, 0.0831345
2: -0.0112683, 0.1210694, 0.0055359, 0.0778081, -0.0890764, 0.1155335
3: -0.0502359, 0.0748101, -0.0235429, 0.0696618, -0.1198977, 0.0983530
4: -0.0342595, 0.1180273, -0.0193165, 0.0613490, -0.0956085, 0.1373438

Time for backsubstitution: 2.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.89 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.89 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.88 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.88 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.42 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.88 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.45 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.89 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.91 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.89 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0745496
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.47 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0078467, 0.0443021, -0.0501801, 0.0541486
1: -0.0304515, 0.0843779, -0.0047341, 0.0745015, -0.1049530, 0.0891120
2: -0.0112683, 0.1210694, 0.0048839, 0.0745222, -0.0857905, 0.1161855
3: -0.0502359, 0.0748101, -0.0191891, 0.0687331, -0.1189690, 0.0939991
4: -0.0342595, 0.1180273, -0.0225027, 0.0567436, -0.0910031, 0.1405299

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.91 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716437, upper bound: 0.0745496
time: 0.46 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0105107, 0.0439873, -0.0498653, 0.0514846
1: -0.0304515, 0.0843779, 0.0025599, 0.0697696, -0.1002211, 0.0818180
2: -0.0112683, 0.1210694, 0.0063427, 0.0763993, -0.0876676, 0.1147266
3: -0.0502359, 0.0748101, -0.0214248, 0.0672601, -0.1174960, 0.0962349
4: -0.0342595, 0.1180273, -0.0185084, 0.0591982, -0.0934577, 0.1365357

Time for backsubstitution: 2.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48

Time for candidate selection: 0.90 seconds
Binary search (step 1): status=Status.UNKNOWN, low=0.0159454, high=0.0574136, mid=0.0574136, abs_max=0.09181444346904755
rel_dist={0: [-0.07506273618631137, 0.07506273618631135]}

## Binary search (step 2) starts
Candidate diff: 0.0366795


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740144, upper bound: 0.0735263
time: 0.35 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0743721, upper bound: 0.0743721
time: 0.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.92 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.92
Output dim: 0, lower bound: -0.0740144, upper bound: 0.0735263
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.92
Output dim: 0, lower bound: -0.0743721, upper bound: 0.0743721

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0096398, 0.0840788, -0.0100369, 0.0748344, -0.0844742, 0.0941158
1: -0.0483434, 0.1096082, -0.0420080, 0.0998038, -0.1481472, 0.1516162
2: -0.0175393, 0.1624446, -0.0189352, 0.1444444, -0.1619836, 0.1813798
3: -0.1008122, 0.1153759, -0.0811189, 0.1068460, -0.2076582, 0.1964948
4: -0.0483946, 0.1864471, -0.0466999, 0.1570692, -0.2054638, 0.2331470

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0730577
time: 0.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0736694, upper bound: 0.0730613
time: 0.38 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0080926, 0.0739747, -0.0117705, 0.0786946, -0.0867872, 0.0857452
1: -0.0402663, 0.0988584, -0.0471804, 0.1036548, -0.1439211, 0.1460388
2: -0.0162963, 0.1430283, -0.0208958, 0.1523376, -0.1686339, 0.1639241
3: -0.0793115, 0.1036161, -0.0884371, 0.1106943, -0.1900058, 0.1920531
4: -0.0462731, 0.1545721, -0.0506829, 0.1693146, -0.2155877, 0.2052550

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0735263, upper bound: 0.0740144
time: 0.36 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0735263, upper bound: 0.0740144
time: 0.37 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.92 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.92
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0730577
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.92
Output dim: 0, lower bound: -0.0736694, upper bound: 0.0730613
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 0, lower bound: -0.0735263, upper bound: 0.0740144
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.92
Output dim: 0, lower bound: -0.0735263, upper bound: 0.0740144

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0080926, 0.0739747, -0.0094406, 0.0807850, -0.0888777, 0.0834153
1: -0.0402663, 0.0988584, -0.0478688, 0.1056063, -0.1458726, 0.1467272
2: -0.0162963, 0.1430283, -0.0173479, 0.1585741, -0.1748704, 0.1603762
3: -0.0793115, 0.1036161, -0.0986593, 0.1095327, -0.1888442, 0.2022754
4: -0.0462731, 0.1545721, -0.0452935, 0.1803576, -0.2266306, 0.1998657

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730577, upper bound: 0.0720345
time: 0.38 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730613, upper bound: 0.0736694
time: 0.39 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0080926, 0.0739747, -0.0080926, 0.0739747, -0.0820673, 0.0820673
1: -0.0402663, 0.0988584, -0.0402663, 0.0988584, -0.1391247, 0.1391247
2: -0.0162963, 0.1430283, -0.0162963, 0.1430283, -0.1593246, 0.1593246
3: -0.0793115, 0.1036161, -0.0793115, 0.1036161, -0.1829276, 0.1829276
4: -0.0462731, 0.1545721, -0.0462731, 0.1545721, -0.2008452, 0.2008452

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730577, upper bound: 0.0725892
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730613, upper bound: 0.0743721
time: 0.37 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.02 seconds
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.0730577, upper bound: 0.0720345
IS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.0730613, upper bound: 0.0736694
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.0730577, upper bound: 0.0725892
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.02
Output dim: 0, lower bound: -0.0730613, upper bound: 0.0743721

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0075458, 0.0730978, -0.0780472, 0.0759988
1: -0.0301443, 0.0939154, -0.0385353, 0.0980235, -0.1281678, 0.1324507
2: -0.0130560, 0.1317879, -0.0157788, 0.1412918, -0.1543478, 0.1475668
3: -0.0685425, 0.0983886, -0.0775842, 0.1028128, -0.1713553, 0.1759728
4: -0.0412280, 0.1366114, -0.0455512, 0.1517030, -0.1929311, 0.1821626

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728806, upper bound: 0.0742395
time: 0.35 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0728806, upper bound: 0.0743721
time: 0.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.94 seconds
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 0, lower bound: -0.0728806, upper bound: 0.0742395
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 0, lower bound: -0.0728806, upper bound: 0.0743721

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, 0.0083338, 0.0470844, -0.0520337, 0.0601192
1: -0.0301443, 0.0939154, -0.0002709, 0.0768870, -0.1070312, 0.0941863
2: -0.0130560, 0.1317879, 0.0037108, 0.0803992, -0.0934552, 0.1280771
3: -0.0685425, 0.0983886, -0.0257773, 0.0755912, -0.1441337, 0.1241658
4: -0.0412280, 0.1366114, -0.0258763, 0.0645177, -0.1057457, 0.1624877

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48

Time for candidate selection: 0.75 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0719937
time: 0.36 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0738459
time: 0.38 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0049493, 0.0684531, -0.0734024, 0.0734024
1: -0.0301443, 0.0939154, -0.0301443, 0.0939154, -0.1240597, 0.1240597
2: -0.0130560, 0.1317879, -0.0130560, 0.1317879, -0.1448440, 0.1448440
3: -0.0685425, 0.0983886, -0.0685425, 0.0983886, -0.1669310, 0.1669310
4: -0.0412280, 0.1366114, -0.0412280, 0.1366114, -0.1778394, 0.1778394

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48

Time for candidate selection: 0.74 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0719937
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0740747
time: 0.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.56 seconds
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0719937
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0738459
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.56
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.56
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0034954, 0.0665585, -0.0724365, 0.0654907
1: -0.0304515, 0.0843779, -0.0289045, 0.0906926, -0.1211441, 0.1132824
2: -0.0112683, 0.1210694, -0.0113797, 0.1291437, -0.1404120, 0.1324491
3: -0.0502359, 0.0748101, -0.0651047, 0.0892128, -0.1394486, 0.1399148
4: -0.0342595, 0.1180273, -0.0391575, 0.1331348, -0.1673943, 0.1571848

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48

Time for candidate selection: 0.73 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0740747
time: 0.36 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.49 seconds
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.49
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.49
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.73 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.36 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.47 seconds
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.47
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.47
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.42 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.64 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.64
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.64
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.74 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.38 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 3.56 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.56
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.77 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.40 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.38 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 3.57 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 3.57
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 3.57
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.38 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 3.58 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 3.58
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 3.58
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.40 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 3.66 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 12, time: 3.66
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 3.66
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.76 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.46 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.42 seconds

## Summary of splitting at layer (split count: 12)
- Time for IS candidates: 3.70 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 3.70
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 3.70
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.45 seconds

## Summary of splitting at layer (split count: 13)
- Time for IS candidates: 3.77 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 14, time: 3.77
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 14, time: 3.77
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.41 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.40 seconds

## Summary of splitting at layer (split count: 14)
- Time for IS candidates: 3.68 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 15, time: 3.68
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 15, time: 3.68
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.79 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.44 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.41 seconds

## Summary of splitting at layer (split count: 15)
- Time for IS candidates: 3.71 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 16, time: 3.71
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 16, time: 3.71
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.81 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.42 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.41 seconds

## Summary of splitting at layer (split count: 16)
- Time for IS candidates: 3.71 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 17, time: 3.71
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 17, time: 3.71
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.82 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.45 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.43 seconds

## Summary of splitting at layer (split count: 17)
- Time for IS candidates: 3.76 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 18, time: 3.76
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 18, time: 3.76
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.84 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.47 seconds

## Summary of splitting at layer (split count: 18)
- Time for IS candidates: 3.92 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 19, time: 3.92
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 19, time: 3.92
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.49 seconds

## Summary of splitting at layer (split count: 19)
- Time for IS candidates: 4.02 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 20, time: 4.02
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 20, time: 4.02
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.48 seconds

## Summary of splitting at layer (split count: 20)
- Time for IS candidates: 3.96 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 21, time: 3.96
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 21, time: 3.96
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.45 seconds

## Summary of splitting at layer (split count: 21)
- Time for IS candidates: 3.93 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 22, time: 3.93
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 22, time: 3.93
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.87 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.49 seconds

## Summary of splitting at layer (split count: 22)
- Time for IS candidates: 3.99 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 23, time: 3.99
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 23, time: 3.99
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.85 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.48 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.46 seconds

## Summary of splitting at layer (split count: 23)
- Time for IS candidates: 3.93 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 24, time: 3.93
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 24, time: 3.93
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.85 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.47 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.47 seconds

## Summary of splitting at layer (split count: 24)
- Time for IS candidates: 3.89 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 25, time: 3.89
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 25, time: 3.89
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.85 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.48 seconds

## Summary of splitting at layer (split count: 25)
- Time for IS candidates: 3.92 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 26, time: 3.92
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 26, time: 3.92
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.47 seconds

## Summary of splitting at layer (split count: 26)
- Time for IS candidates: 3.94 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 27, time: 3.94
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 27, time: 3.94
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.86 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.47 seconds

## Summary of splitting at layer (split count: 27)
- Time for IS candidates: 3.96 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 28, time: 3.96
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 28, time: 3.96
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.88 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.49 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.46 seconds

## Summary of splitting at layer (split count: 28)
- Time for IS candidates: 3.95 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 29, time: 3.95
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 29, time: 3.95
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.89 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738464
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.48 seconds

## Summary of splitting at layer (split count: 29)
- Time for IS candidates: 4.00 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 30, time: 4.00
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738464
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 30, time: 4.00
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.90 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.47 seconds

## Summary of splitting at layer (split count: 30)
- Time for IS candidates: 3.99 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 31, time: 3.99
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 31, time: 3.99
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.90 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.52 seconds

## Summary of splitting at layer (split count: 31)
- Time for IS candidates: 4.04 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 32, time: 4.04
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 32, time: 4.04
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.90 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.49 seconds

## Summary of splitting at layer (split count: 32)
- Time for IS candidates: 4.05 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 33, time: 4.05
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 33, time: 4.05
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.91 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.53 seconds

## Summary of splitting at layer (split count: 33)
- Time for IS candidates: 4.10 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 34, time: 4.10
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 34, time: 4.10
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.92 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.50 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.50 seconds

## Summary of splitting at layer (split count: 34)
- Time for IS candidates: 4.08 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 35, time: 4.08
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 35, time: 4.08
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.90 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.52 seconds

## Summary of splitting at layer (split count: 35)
- Time for IS candidates: 4.00 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 36, time: 4.00
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 36, time: 4.00
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.90 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.51 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.50 seconds

## Summary of splitting at layer (split count: 36)
- Time for IS candidates: 3.97 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 37, time: 3.97
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 37, time: 3.97
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.89 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.53 seconds

## Summary of splitting at layer (split count: 37)
- Time for IS candidates: 4.03 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 38, time: 4.03
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 38, time: 4.03
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.91 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.52 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.50 seconds

## Summary of splitting at layer (split count: 38)
- Time for IS candidates: 4.03 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 39, time: 4.03
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 39, time: 4.03
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.94 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.54 seconds

## Summary of splitting at layer (split count: 39)
- Time for IS candidates: 4.09 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 40, time: 4.09
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 40, time: 4.09
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.97 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.54 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.56 seconds

## Summary of splitting at layer (split count: 40)
- Time for IS candidates: 4.16 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 41, time: 4.16
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 41, time: 4.16
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.96 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.56 seconds

## Summary of splitting at layer (split count: 41)
- Time for IS candidates: 4.25 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 42, time: 4.25
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 42, time: 4.25
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 0.96 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.53 seconds

## Summary of splitting at layer (split count: 42)
- Time for IS candidates: 4.19 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 43, time: 4.19
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 43, time: 4.19
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 0.96 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.57 seconds

## Summary of splitting at layer (split count: 43)
- Time for IS candidates: 4.29 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 44, time: 4.29
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 44, time: 4.29
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 1.03 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.56 seconds

## Summary of splitting at layer (split count: 44)
- Time for IS candidates: 4.49 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 45, time: 4.49
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 45, time: 4.49
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 1.03 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.62 seconds

## Summary of splitting at layer (split count: 45)
- Time for IS candidates: 4.57 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 46, time: 4.57
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 46, time: 4.57
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 1.03 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.57 seconds

## Summary of splitting at layer (split count: 46)
- Time for IS candidates: 4.53 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 47, time: 4.53
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 47, time: 4.53
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 1.04 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.63 seconds

## Summary of splitting at layer (split count: 47)
- Time for IS candidates: 4.61 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 48, time: 4.61
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 48, time: 4.61
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.30 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 1.05 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.60 seconds

## Summary of splitting at layer (split count: 48)
- Time for IS candidates: 4.57 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 49, time: 4.57
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 49, time: 4.57
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 1.05 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.62 seconds

## Summary of splitting at layer (split count: 49)
- Time for IS candidates: 4.63 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 50, time: 4.63
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 50, time: 4.63
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 1.07 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.62 seconds

## Summary of splitting at layer (split count: 50)
- Time for IS candidates: 4.65 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 51, time: 4.65
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 51, time: 4.65
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 1.03 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.65 seconds

## Summary of splitting at layer (split count: 51)
- Time for IS candidates: 4.64 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 52, time: 4.64
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 52, time: 4.64
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 1.00 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.59 seconds

## Summary of splitting at layer (split count: 52)
- Time for IS candidates: 4.40 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 53, time: 4.40
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 53, time: 4.40
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 1.00 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.66 seconds

## Summary of splitting at layer (split count: 53)
- Time for IS candidates: 4.42 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 54, time: 4.42
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 54, time: 4.42
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 1.02 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.63 seconds

## Summary of splitting at layer (split count: 54)
- Time for IS candidates: 4.48 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 55, time: 4.48
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 55, time: 4.48
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 1.02 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.65 seconds

## Summary of splitting at layer (split count: 55)
- Time for IS candidates: 4.53 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 56, time: 4.53
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 56, time: 4.53
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 1.04 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.64 seconds

## Summary of splitting at layer (split count: 56)
- Time for IS candidates: 4.53 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 57, time: 4.53
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 57, time: 4.53
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 1.04 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.66 seconds

## Summary of splitting at layer (split count: 57)
- Time for IS candidates: 4.54 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 58, time: 4.54
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 58, time: 4.54
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 1.04 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.63 seconds

## Summary of splitting at layer (split count: 58)
- Time for IS candidates: 4.49 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 59, time: 4.49
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 59, time: 4.49
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 1.05 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.66 seconds

## Summary of splitting at layer (split count: 59)
- Time for IS candidates: 4.50 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 60, time: 4.50
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 60, time: 4.50
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 1.05 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.67 seconds

## Summary of splitting at layer (split count: 60)
- Time for IS candidates: 4.50 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 61, time: 4.50
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 61, time: 4.50
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 1.05 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.69 seconds

## Summary of splitting at layer (split count: 61)
- Time for IS candidates: 4.60 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 62, time: 4.60
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 62, time: 4.60
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 1.06 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.68 seconds

## Summary of splitting at layer (split count: 62)
- Time for IS candidates: 4.58 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 63, time: 4.58
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 63, time: 4.58
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 1.07 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.68 seconds

## Summary of splitting at layer (split count: 63)
- Time for IS candidates: 4.63 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 64, time: 4.63
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 64, time: 4.63
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 1.09 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.68 seconds

## Summary of splitting at layer (split count: 64)
- Time for IS candidates: 4.70 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 65, time: 4.70
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 65, time: 4.70
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 1.07 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.70 seconds

## Summary of splitting at layer (split count: 65)
- Time for IS candidates: 4.69 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 66, time: 4.69
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 66, time: 4.69
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 1.08 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.71 seconds

## Summary of splitting at layer (split count: 66)
- Time for IS candidates: 4.69 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 67, time: 4.69
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 67, time: 4.69
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 1.10 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.70 seconds

## Summary of splitting at layer (split count: 67)
- Time for IS candidates: 4.77 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 68, time: 4.77
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 68, time: 4.77
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 1.10 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.72 seconds

## Summary of splitting at layer (split count: 68)
- Time for IS candidates: 4.75 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 69, time: 4.75
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 69, time: 4.75
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48

Time for candidate selection: 1.11 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747
time: 0.71 seconds

## Summary of splitting at layer (split count: 69)
- Time for IS candidates: 4.80 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 70, time: 4.80
Output dim: 0, lower bound: -0.0716816, upper bound: 0.0738463
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 70, time: 4.80
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 48

Time for candidate selection: 1.17 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747
time: 0.75 seconds

## Summary of splitting at layer (split count: 70)
- Time for IS candidates: 4.93 seconds
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 71, time: 4.93
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0719937
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 71, time: 4.93
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0740747

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.45 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0159454, high=0.0366795, mid=0.0366795, abs_max=0.09181444346904755
rel_dist={0: [-0.0745849892195982, 0.07458498921959819]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01594543504552348
execution time: 1145.70 seconds
