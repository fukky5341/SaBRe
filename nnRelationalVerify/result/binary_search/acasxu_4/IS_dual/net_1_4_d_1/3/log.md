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
execution time: IAR + LP analysis = 1.94 + 0.88 = 2.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0754965, upper bound: 0.0754965


# Binary Search by BASE starts (time budget: 1197.18 seconds, max iter: 100)

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
Binary search time: 50.94 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.01594543504552348


# Individual Split (IS_dual) starts
Time budget: 1146.24 seconds

## Binary search (step 0) starts
Candidate diff: 0.0988818


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754965, upper bound: 0.0736647
time: 0.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754287, upper bound: 0.0754287
time: 0.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.79 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0754965, upper bound: 0.0736647
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.79
Output dim: 0, lower bound: -0.0754287, upper bound: 0.0754287

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0056712, 0.0494482, -0.0122710, 0.0795435, -0.0738723, 0.0617192
1: -0.0048309, 0.0792722, -0.0482301, 0.1046001, -0.1094310, 0.1275024
2: 0.0013121, 0.0852126, -0.0215323, 0.1538582, -0.1525461, 0.1067449
3: -0.0302970, 0.0797933, -0.0898915, 0.1120336, -0.1423306, 0.1696847
4: -0.0295759, 0.0721076, -0.0514701, 0.1717802, -0.2013561, 0.1235777

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0720345
time: 0.30 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0732197
time: 0.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0080689, 0.0722086, -0.0122710, 0.0795435, -0.0876124, 0.0844796
1: -0.0359785, 0.0976419, -0.0482301, 0.1046001, -0.1405786, 0.1458721
2: -0.0166659, 0.1399382, -0.0215323, 0.1538582, -0.1705241, 0.1614705
3: -0.0768474, 0.1041032, -0.0898915, 0.1120336, -0.1888810, 0.1939946
4: -0.0455294, 0.1494691, -0.0514701, 0.1717802, -0.2173096, 0.2009392

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0739891, upper bound: 0.0741465
time: 0.31 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0751743
time: 0.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.84 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0720345
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0732197
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 0, lower bound: -0.0739891, upper bound: 0.0741465
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.84
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0751743

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0056786, 0.0494340, -0.0096398, 0.0840788, -0.0784002, 0.0590738
1: -0.0048107, 0.0792583, -0.0483434, 0.1096082, -0.1144190, 0.1276017
2: 0.0013196, 0.0851824, -0.0175393, 0.1624446, -0.1611249, 0.1027216
3: -0.0302683, 0.0797658, -0.1008122, 0.1153759, -0.1456442, 0.1805780
4: -0.0295506, 0.0720598, -0.0483946, 0.1864471, -0.2159977, 0.1204543

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0736243, upper bound: 0.0720345
time: 0.32 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0736243, upper bound: 0.0720345
time: 0.30 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0056712, 0.0494482, -0.0080926, 0.0739747, -0.0683034, 0.0575409
1: -0.0048309, 0.0792722, -0.0402663, 0.0988584, -0.1036893, 0.1195385
2: 0.0013121, 0.0852126, -0.0162963, 0.1430283, -0.1417162, 0.1015089
3: -0.0302970, 0.0797933, -0.0793115, 0.1036161, -0.1339131, 0.1591048
4: -0.0295759, 0.0721076, -0.0462731, 0.1545721, -0.1841480, 0.1183806

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0732197
time: 0.30 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0732197
time: 0.31 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -0.0055032, 0.0746784, -0.0122537, 0.0795023, -0.0850055, 0.0869322
1: -0.0363166, 0.1003191, -0.0481836, 0.1045535, -0.1408701, 0.1485027
2: -0.0130474, 0.1460492, -0.0215126, 0.1537826, -0.1668300, 0.1675618
3: -0.0863938, 0.1024000, -0.0898233, 0.1119772, -0.1983710, 0.1922233
4: -0.0407893, 0.1603383, -0.0514254, 0.1716636, -0.2124529, 0.2117637

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0729613
time: 0.31 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0741465
time: 0.32 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0122710, 0.0795435, -0.0844928, 0.0807240
1: -0.0301443, 0.0939154, -0.0482301, 0.1046001, -0.1347444, 0.1421455
2: -0.0130560, 0.1317879, -0.0215323, 0.1538582, -0.1669142, 0.1533202
3: -0.0685425, 0.0983886, -0.0898915, 0.1120336, -0.1805761, 0.1882800
4: -0.0412280, 0.1366114, -0.0514701, 0.1717802, -0.2130082, 0.1880815

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0739891
time: 0.31 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0751743
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.82 seconds
IS_A1_B1_B1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 0, lower bound: -0.0736243, upper bound: 0.0720345
IS_A1_B1_B2, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 0, lower bound: -0.0736243, upper bound: 0.0720345
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0732197
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0732197
IS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0729613
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0741465
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0739891
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0751743

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0055032, 0.0746784, -0.0080926, 0.0739747, -0.0794779, 0.0827711
1: -0.0363166, 0.1003191, -0.0402663, 0.0988584, -0.1351749, 0.1405854
2: -0.0130474, 0.1460492, -0.0162963, 0.1430283, -0.1560757, 0.1623455
3: -0.0863938, 0.1024000, -0.0793115, 0.1036161, -0.1900099, 0.1817115
4: -0.0407893, 0.1603383, -0.0462731, 0.1545721, -0.1953614, 0.2066113

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0741465
time: 0.31 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0741465
time: 0.30 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0096398, 0.0840788, -0.0890281, 0.0780929
1: -0.0301443, 0.0939154, -0.0483434, 0.1096082, -0.1397525, 0.1422587
2: -0.0130560, 0.1317879, -0.0175393, 0.1624446, -0.1755006, 0.1493272
3: -0.0685425, 0.0983886, -0.1008122, 0.1153759, -0.1839184, 0.1992007
4: -0.0412280, 0.1366114, -0.0483946, 0.1864471, -0.2276751, 0.1850060

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0739891
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0739891
time: 0.35 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0080926, 0.0739747, -0.0789240, 0.0765457
1: -0.0301443, 0.0939154, -0.0402663, 0.0988584, -0.1290027, 0.1341817
2: -0.0130560, 0.1317879, -0.0162963, 0.1430283, -0.1560843, 0.1480842
3: -0.0685425, 0.0983886, -0.0793115, 0.1036161, -0.1721586, 0.1777001
4: -0.0412280, 0.1366114, -0.0462731, 0.1545721, -0.1958002, 0.1828845

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0751743
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0751743
time: 0.33 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.90 seconds
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0741465
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0741465
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0739891
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0739891
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0751743
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.90
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0751743

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0055032, 0.0746784, 0.0083338, 0.0470844, -0.0525876, 0.0663446
1: -0.0363166, 0.1003191, -0.0002709, 0.0768870, -0.1132035, 0.1005900
2: -0.0130474, 0.1460492, 0.0037108, 0.0803992, -0.0934466, 0.1423384
3: -0.0863938, 0.1024000, -0.0257773, 0.0755912, -0.1619850, 0.1281773
4: -0.0407893, 0.1603383, -0.0258763, 0.0645177, -0.1053070, 0.1862146

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7

Time for candidate selection: 1.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0726668
time: 0.34 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
time: 0.36 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0055032, 0.0746784, -0.0049493, 0.0684531, -0.0739563, 0.0796278
1: -0.0363166, 0.1003191, -0.0301443, 0.0939154, -0.1302319, 0.1304634
2: -0.0130474, 0.1460492, -0.0130560, 0.1317879, -0.1448353, 0.1591052
3: -0.0863938, 0.1024000, -0.0685425, 0.0983886, -0.1847824, 0.1709425
4: -0.0407893, 0.1603383, -0.0412280, 0.1366114, -0.1774007, 0.2015663

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 1.18 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0726668
time: 0.32 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
time: 0.33 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, 0.0031931, 0.0596317, -0.0645811, 0.0652600
1: -0.0301443, 0.0939154, -0.0159592, 0.0935836, -0.1237278, 0.1098745
2: -0.0130560, 0.1317879, -0.0004775, 0.1070566, -0.1201126, 0.1322655
3: -0.0685425, 0.0983886, -0.0561146, 0.0980010, -0.1665435, 0.1545032
4: -0.0412280, 0.1366114, -0.0290360, 0.1065627, -0.1477907, 0.1656474

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 1.19 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711760, upper bound: 0.0717345
time: 0.32 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731986, upper bound: 0.0739534
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0056816, 0.0768258, -0.0817752, 0.0741346
1: -0.0301443, 0.0939154, -0.0367180, 0.1016865, -0.1318308, 0.1306333
2: -0.0130560, 0.1317879, -0.0131781, 0.1492988, -0.1623548, 0.1449660
3: -0.0685425, 0.0983886, -0.0882968, 0.1038739, -0.1724163, 0.1866854
4: -0.0412280, 0.1366114, -0.0414514, 0.1655897, -0.2068177, 0.1780628

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 47
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 1.18 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0722610, upper bound: 0.0721618
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0731986, upper bound: 0.0739534
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, 0.0083338, 0.0470844, -0.0520337, 0.0601192
1: -0.0301443, 0.0939154, -0.0002709, 0.0768870, -0.1070312, 0.0941863
2: -0.0130560, 0.1317879, 0.0037108, 0.0803992, -0.0934552, 0.1280771
3: -0.0685425, 0.0983886, -0.0257773, 0.0755912, -0.1441337, 0.1241658
4: -0.0412280, 0.1366114, -0.0258763, 0.0645177, -0.1057457, 0.1624877

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 1.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0707840, upper bound: 0.0724653
time: 0.34 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731042, upper bound: 0.0750589
time: 0.35 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0049493, 0.0684531, -0.0734024, 0.0734024
1: -0.0301443, 0.0939154, -0.0301443, 0.0939154, -0.1240597, 0.1240597
2: -0.0130560, 0.1317879, -0.0130560, 0.1317879, -0.1448440, 0.1448440
3: -0.0685425, 0.0983886, -0.0685425, 0.0983886, -0.1669310, 0.1669310
4: -0.0412280, 0.1366114, -0.0412280, 0.1366114, -0.1778394, 0.1778394

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 1.19 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0725221
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
time: 0.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.98 seconds
IS_A2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.98
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0726668
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
IS_A2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.98
Output dim: 0, lower bound: -0.0702543, upper bound: 0.0726668
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
IS_A2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.98
Output dim: 0, lower bound: -0.0711760, upper bound: 0.0717345
IS_A2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.98
Output dim: 0, lower bound: -0.0731986, upper bound: 0.0739534
IS_A2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.98
Output dim: 0, lower bound: -0.0722610, upper bound: 0.0721618
IS_A2_A2_B1_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.98
Output dim: 0, lower bound: -0.0731986, upper bound: 0.0739534
IS_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.98
Output dim: 0, lower bound: -0.0707840, upper bound: 0.0724653
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -0.0731042, upper bound: 0.0750589
IS_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.98
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0725221
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.98
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467

## BFS IS instance: IS_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0045362, 0.0732913, 0.0083338, 0.0470844, -0.0516205, 0.0649575
1: -0.0342618, 0.0992907, -0.0002709, 0.0768870, -0.1111488, 0.0995616
2: -0.0123451, 0.1432838, 0.0037108, 0.0803992, -0.0927442, 0.1395730
3: -0.0837640, 0.1012131, -0.0257773, 0.0755912, -0.1593552, 0.1269904
4: -0.0400739, 0.1559919, -0.0258763, 0.0645177, -0.1045916, 0.1818682

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 47

## Relational analysis of IS_A2_A1_B2_B1_A2_A1

### Relational analysis result of IS_A2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719988, upper bound: 0.0740311
time: 0.36 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A1_B2_B1_A2_A1

### Relational analysis result of IS_A2_A1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0713983, upper bound: 0.0714113
time: 0.33 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710581, upper bound: 0.0718189
time: 0.32 seconds

## BFS IS instance: IS_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045362, 0.0732913, -0.0049493, 0.0684531, -0.0729892, 0.0782406
1: -0.0342618, 0.0992907, -0.0301443, 0.0939154, -0.1281772, 0.1294349
2: -0.0123451, 0.1432838, -0.0130560, 0.1317879, -0.1441330, 0.1563398
3: -0.0837640, 0.1012131, -0.0685425, 0.0983886, -0.1821526, 0.1697556
4: -0.0400739, 0.1559919, -0.0412280, 0.1366114, -0.1766853, 0.1972199

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 47
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712917, upper bound: 0.0717108
time: 0.33 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0712917, upper bound: 0.0740311
time: 0.34 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0083338, 0.0470844, -0.0510514, 0.0590321
1: -0.0276524, 0.0931255, -0.0002709, 0.0768870, -0.1045393, 0.0933965
2: -0.0123972, 0.1293659, 0.0037108, 0.0803992, -0.0927964, 0.1256551
3: -0.0660703, 0.0977019, -0.0257773, 0.0755912, -0.1416615, 0.1234792
4: -0.0406218, 0.1327123, -0.0258763, 0.0645177, -0.1051394, 0.1585886

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0721072, upper bound: 0.0747092
time: 0.34 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0715436, upper bound: 0.0747313
time: 0.34 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0049493, 0.0684531, -0.0743310, 0.0669446
1: -0.0304515, 0.0843779, -0.0301443, 0.0939154, -0.1243669, 0.1145222
2: -0.0112683, 0.1210694, -0.0130560, 0.1317879, -0.1430563, 0.1341254
3: -0.0502359, 0.0748101, -0.0685425, 0.0983886, -0.1486244, 0.1433525
4: -0.0342595, 0.1180273, -0.0412280, 0.1366114, -0.1708709, 0.1592553

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0748467
time: 0.34 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.18 seconds
IS_A2_A1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.0713983, upper bound: 0.0714113
IS_A2_A1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.0710581, upper bound: 0.0718189
IS_A2_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.0712917, upper bound: 0.0717108
IS_A2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.0712917, upper bound: 0.0740311
IS_A2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.0721072, upper bound: 0.0747092
IS_A2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.0715436, upper bound: 0.0747313
IS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.18
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0748467

## BFS IS instance: IS_A2_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045362, 0.0732913, -0.0039670, 0.0673659, -0.0719021, 0.0772583
1: -0.0342618, 0.0992907, -0.0276524, 0.0931255, -0.1273873, 0.1269430
2: -0.0123451, 0.1432838, -0.0123972, 0.1293659, -0.1417110, 0.1556810
3: -0.0837640, 0.1012131, -0.0660703, 0.0977019, -0.1814659, 0.1672834
4: -0.0400739, 0.1559919, -0.0406218, 0.1327123, -0.1727862, 0.1966137

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702994, upper bound: 0.0726668
time: 0.34 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740311
time: 0.35 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0097754, 0.0448622, -0.0488292, 0.0575906
1: -0.0276524, 0.0931255, 0.0012435, 0.0714976, -0.0991500, 0.0918821
2: -0.0123972, 0.1293659, 0.0055359, 0.0778081, -0.0902053, 0.1238300
3: -0.0660703, 0.0977019, -0.0235429, 0.0696618, -0.1357321, 0.1212448
4: -0.0406218, 0.1327123, -0.0193165, 0.0613490, -0.1019708, 0.1520288

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0718083, upper bound: 0.0723385
time: 0.33 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.32 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720385, upper bound: 0.0746514
time: 0.33 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0078467, 0.0443021, -0.0482691, 0.0595193
1: -0.0276524, 0.0931255, -0.0047341, 0.0745015, -0.1021538, 0.0978596
2: -0.0123972, 0.1293659, 0.0048839, 0.0745222, -0.0869194, 0.1244820
3: -0.0660703, 0.0977019, -0.0191891, 0.0687331, -0.1348034, 0.1168910
4: -0.0406218, 0.1327123, -0.0225027, 0.0567436, -0.0973654, 0.1552150

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711634, upper bound: 0.0746760
time: 0.33 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0714749, upper bound: 0.0746703
time: 0.33 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711015, upper bound: 0.0745032
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743948
time: 0.34 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.35 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716905, upper bound: 0.0747509
time: 0.34 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467
time: 0.35 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.83 seconds
IS_A2_A1_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.83
Output dim: 0, lower bound: -0.0702994, upper bound: 0.0726668
IS_A2_A1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740311
IS_A2_A2_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 0, lower bound: -0.0720385, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 0, lower bound: -0.0711634, upper bound: 0.0746760
IS_A2_A2_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 0, lower bound: -0.0714749, upper bound: 0.0746703
IS_A2_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743948
IS_A2_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.83
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.83
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467

## BFS IS instance: IS_A2_A1_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045362, 0.0732913, -0.0039670, 0.0673659, -0.0719021, 0.0772583
1: -0.0342618, 0.0992907, -0.0276524, 0.0931255, -0.1273873, 0.1269430
2: -0.0123451, 0.1432838, -0.0123972, 0.1293659, -0.1417110, 0.1556810
3: -0.0837640, 0.1012131, -0.0660703, 0.0977019, -0.1814659, 0.1672834
4: -0.0400739, 0.1559919, -0.0406218, 0.1327123, -0.1727862, 0.1966137

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0717108
time: 0.34 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740310
time: 0.35 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0119214, 0.0440228, -0.0479898, 0.0554446
1: -0.0276524, 0.0931255, 0.0005230, 0.0695521, -0.0972045, 0.0926026
2: -0.0123972, 0.1293659, 0.0096441, 0.0773689, -0.0897661, 0.1197219
3: -0.0660703, 0.0977019, -0.0194770, 0.0624470, -0.1285173, 0.1171789
4: -0.0406218, 0.1327123, -0.0188315, 0.0573395, -0.0979612, 0.1515437

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.33 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.33 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0100352, 0.0447484, -0.0487154, 0.0573308
1: -0.0276524, 0.0931255, 0.0015434, 0.0713148, -0.0989672, 0.0915821
2: -0.0123972, 0.1293659, 0.0057138, 0.0774657, -0.0898629, 0.1236522
3: -0.0660703, 0.0977019, -0.0229581, 0.0693600, -0.1354303, 0.1206600
4: -0.0406218, 0.1327123, -0.0192196, 0.0607713, -0.1013930, 0.1519319

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720385, upper bound: 0.0746514
time: 0.32 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0089256, 0.0436134, -0.0475805, 0.0584403
1: -0.0276524, 0.0931255, -0.0056828, 0.0719095, -0.0995619, 0.0988083
2: -0.0123972, 0.1293659, 0.0094991, 0.0732964, -0.0856936, 0.1198668
3: -0.0660703, 0.0977019, -0.0168171, 0.0634604, -0.1295307, 0.1145190
4: -0.0406218, 0.1327123, -0.0218024, 0.0532283, -0.0938501, 0.1545147

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0746760
time: 0.35 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0080052, 0.0442183, -0.0481854, 0.0593608
1: -0.0276524, 0.0931255, -0.0045802, 0.0743399, -0.1019923, 0.0977057
2: -0.0123972, 0.1293659, 0.0050475, 0.0742267, -0.0866239, 0.1243185
3: -0.0660703, 0.0977019, -0.0188669, 0.0684462, -0.1345165, 0.1165688
4: -0.0406218, 0.1327123, -0.0224565, 0.0562747, -0.0968964, 0.1551688

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0727767
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0727767
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0024844, 0.0581586, -0.0021452, 0.0612259, -0.0637102, 0.0603038
1: -0.0264296, 0.0815245, -0.0263871, 0.0842858, -0.1107154, 0.1079116
2: -0.0051643, 0.1094249, -0.0090670, 0.1218928, -0.1270570, 0.1184920
3: -0.0423236, 0.0763173, -0.0604222, 0.0902384, -0.1325620, 0.1367395
4: -0.0359877, 0.1030843, -0.0321517, 0.1232046, -0.1591924, 0.1352361

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743949
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743949
time: 0.33 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0021452, 0.0612259, -0.0666225, 0.0636251
1: -0.0295746, 0.0838206, -0.0263871, 0.0842858, -0.1138604, 0.1102077
2: -0.0105933, 0.1197037, -0.0090670, 0.1218928, -0.1324860, 0.1287707
3: -0.0481883, 0.0739372, -0.0604222, 0.0902384, -0.1384267, 0.1343594
4: -0.0338624, 0.1156039, -0.0321517, 0.1232046, -0.1570670, 0.1477557

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0721539, upper bound: 0.0742207
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467
time: 0.37 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.55 seconds
IS_A2_A1_B2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0717108
IS_A2_A1_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740310
IS_A2_A2_B2_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0720385, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
IS_A2_A2_B2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0746760
IS_A2_A2_B2_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0727767
IS_A2_A2_B2_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0727767
IS_A2_A2_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743949
IS_A2_A2_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743949
IS_A2_A2_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0721539, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_A2_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467

## BFS IS instance: IS_A2_A1_B2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045362, 0.0732913, -0.0039670, 0.0673659, -0.0719021, 0.0772583
1: -0.0342618, 0.0992907, -0.0276524, 0.0931255, -0.1273873, 0.1269430
2: -0.0123451, 0.1432838, -0.0123972, 0.1293659, -0.1417110, 0.1556810
3: -0.0837640, 0.1012131, -0.0660703, 0.0977019, -0.1814659, 0.1672834
4: -0.0400739, 0.1559919, -0.0406218, 0.1327123, -0.1727862, 0.1966137

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702994, upper bound: 0.0726668
time: 0.34 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740311
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0119214, 0.0440228, -0.0479898, 0.0554446
1: -0.0276524, 0.0931255, 0.0005230, 0.0695521, -0.0972045, 0.0926026
2: -0.0123972, 0.1293659, 0.0096441, 0.0773689, -0.0897661, 0.1197219
3: -0.0660703, 0.0977019, -0.0194770, 0.0624470, -0.1285173, 0.1171789
4: -0.0406218, 0.1327123, -0.0188315, 0.0573395, -0.0979612, 0.1515437

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
time: 0.33 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.35 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0089256, 0.0436134, -0.0475805, 0.0584403
1: -0.0276524, 0.0931255, -0.0056828, 0.0719095, -0.0995619, 0.0988083
2: -0.0123972, 0.1293659, 0.0094991, 0.0732964, -0.0856936, 0.1198668
3: -0.0660703, 0.0977019, -0.0168171, 0.0634604, -0.1295307, 0.1145190
4: -0.0406218, 0.1327123, -0.0218024, 0.0532283, -0.0938501, 0.1545147

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0741902
time: 0.34 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0100352, 0.0447484, -0.0487154, 0.0573308
1: -0.0276524, 0.0931255, 0.0015434, 0.0713148, -0.0989672, 0.0915821
2: -0.0123972, 0.1293659, 0.0057138, 0.0774657, -0.0898629, 0.1236522
3: -0.0660703, 0.0977019, -0.0229581, 0.0693600, -0.1354303, 0.1206600
4: -0.0406218, 0.1327123, -0.0192196, 0.0607713, -0.1013930, 0.1519319

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0717476, upper bound: 0.0723384
time: 0.32 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720385, upper bound: 0.0746514
time: 0.33 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0080052, 0.0442183, -0.0481854, 0.0593608
1: -0.0276524, 0.0931255, -0.0045802, 0.0743399, -0.1019923, 0.0977057
2: -0.0123972, 0.1293659, 0.0050475, 0.0742267, -0.0866239, 0.1243185
3: -0.0660703, 0.0977019, -0.0188669, 0.0684462, -0.1345165, 0.1165688
4: -0.0406218, 0.1327123, -0.0224565, 0.0562747, -0.0968964, 0.1551688

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040268, 0.0608585, 0.0089256, 0.0436134, -0.0476403, 0.0519329
1: -0.0240528, 0.0833467, -0.0056828, 0.0719095, -0.0959624, 0.0890295
2: -0.0102135, 0.1185798, 0.0094991, 0.0732964, -0.0835099, 0.1090807
3: -0.0473803, 0.0734831, -0.0168171, 0.0634604, -0.1108407, 0.0903002
4: -0.0333277, 0.1139941, -0.0218024, 0.0532283, -0.0865560, 0.1357965

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.34 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.33 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0024844, 0.0581586, -0.0021452, 0.0612259, -0.0637102, 0.0603038
1: -0.0264296, 0.0815245, -0.0263871, 0.0842858, -0.1107154, 0.1079116
2: -0.0051643, 0.1094249, -0.0090670, 0.1218928, -0.1270570, 0.1184920
3: -0.0423236, 0.0763173, -0.0604222, 0.0902384, -0.1325620, 0.1367395
4: -0.0359877, 0.1030843, -0.0321517, 0.1232046, -0.1591924, 0.1352361

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0708850, upper bound: 0.0721222
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0708850, upper bound: 0.0721222
time: 0.35 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0024844, 0.0581586, -0.0058780, 0.0619953, -0.0644797, 0.0640366
1: -0.0264296, 0.0815245, -0.0304515, 0.0843779, -0.1108075, 0.1119760
2: -0.0051643, 0.1094249, -0.0112683, 0.1210694, -0.1262336, 0.1206933
3: -0.0423236, 0.0763173, -0.0502359, 0.0748101, -0.1171336, 0.1265531
4: -0.0359877, 0.1030843, -0.0342595, 0.1180273, -0.1540150, 0.1373438

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711798, upper bound: 0.0726242
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711798, upper bound: 0.0743949
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0021452, 0.0612259, -0.0666225, 0.0636251
1: -0.0295746, 0.0838206, -0.0263871, 0.0842858, -0.1138604, 0.1102077
2: -0.0105933, 0.1197037, -0.0090670, 0.1218928, -0.1324860, 0.1287707
3: -0.0481883, 0.0739372, -0.0604222, 0.0902384, -0.1384267, 0.1343594
4: -0.0338624, 0.1156039, -0.0321517, 0.1232046, -0.1570670, 0.1477557

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716324, upper bound: 0.0724500
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716324, upper bound: 0.0724658
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0058780, 0.0619953, -0.0673919, 0.0673578
1: -0.0295746, 0.0838206, -0.0304515, 0.0843779, -0.1139525, 0.1142721
2: -0.0105933, 0.1197037, -0.0112683, 0.1210694, -0.1316626, 0.1309720
3: -0.0481883, 0.0739372, -0.0502359, 0.0748101, -0.1229984, 0.1241731
4: -0.0338624, 0.1156039, -0.0342595, 0.1180273, -0.1518897, 0.1498634

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0713638, upper bound: 0.0741086
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
time: 0.34 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0742207
time: 0.35 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711015, upper bound: 0.0745032
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716324, upper bound: 0.0724500
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716905, upper bound: 0.0747509
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467
time: 0.37 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 6.19 seconds
IS_A2_A1_B2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0702994, upper bound: 0.0726668
IS_A2_A1_B2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740311
IS_A2_A2_B2_B1_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
IS_A2_A2_B2_B1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
IS_A2_A2_B2_B1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0720385, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0708850, upper bound: 0.0721222
IS_A2_A2_B2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0708850, upper bound: 0.0721222
IS_A2_A2_B2_B2_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0711798, upper bound: 0.0726242
IS_A2_A2_B2_B2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0711798, upper bound: 0.0743949
IS_A2_A2_B2_B2_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0716324, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0716324, upper bound: 0.0724658
IS_A2_A2_B2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0716324, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 6.19
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467

## BFS IS instance: IS_A2_A1_B2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045362, 0.0732913, -0.0039670, 0.0673659, -0.0719021, 0.0772583
1: -0.0342618, 0.0992907, -0.0276524, 0.0931255, -0.1273873, 0.1269430
2: -0.0123451, 0.1432838, -0.0123972, 0.1293659, -0.1417110, 0.1556810
3: -0.0837640, 0.1012131, -0.0660703, 0.0977019, -0.1814659, 0.1672834
4: -0.0400739, 0.1559919, -0.0406218, 0.1327123, -0.1727862, 0.1966137

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0717108
time: 0.34 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740310
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0036633, 0.0672161, 0.0119214, 0.0440228, -0.0476861, 0.0552947
1: -0.0272224, 0.0929019, 0.0005230, 0.0695521, -0.0967745, 0.0923789
2: -0.0120373, 0.1289738, 0.0096441, 0.0773689, -0.0894062, 0.1193298
3: -0.0653339, 0.0973650, -0.0194770, 0.0624470, -0.1277809, 0.1168420
4: -0.0405064, 0.1320211, -0.0188315, 0.0573395, -0.0978459, 0.1508526

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040268, 0.0608585, 0.0089256, 0.0436134, -0.0476403, 0.0519329
1: -0.0240528, 0.0833467, -0.0056828, 0.0719095, -0.0959624, 0.0890295
2: -0.0102135, 0.1185798, 0.0094991, 0.0732964, -0.0835099, 0.1090807
3: -0.0473803, 0.0734831, -0.0168171, 0.0634604, -0.1108407, 0.0903002
4: -0.0333277, 0.1139941, -0.0218024, 0.0532283, -0.0865560, 0.1357965

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0119214, 0.0440228, -0.0479898, 0.0554446
1: -0.0276524, 0.0931255, 0.0005230, 0.0695521, -0.0972045, 0.0926026
2: -0.0123972, 0.1293659, 0.0096441, 0.0773689, -0.0897661, 0.1197219
3: -0.0660703, 0.0977019, -0.0194770, 0.0624470, -0.1285173, 0.1171789
4: -0.0406218, 0.1327123, -0.0188315, 0.0573395, -0.0979612, 0.1515437

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0100352, 0.0447484, -0.0487154, 0.0573308
1: -0.0276524, 0.0931255, 0.0015434, 0.0713148, -0.0989672, 0.0915821
2: -0.0123972, 0.1293659, 0.0057138, 0.0774657, -0.0898629, 0.1236522
3: -0.0660703, 0.0977019, -0.0229581, 0.0693600, -0.1354303, 0.1206600
4: -0.0406218, 0.1327123, -0.0192196, 0.0607713, -0.1013930, 0.1519319

Time for backsubstitution: 2.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720385, upper bound: 0.0746514
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0089256, 0.0436134, -0.0475805, 0.0584403
1: -0.0276524, 0.0931255, -0.0056828, 0.0719095, -0.0995619, 0.0988083
2: -0.0123972, 0.1293659, 0.0094991, 0.0732964, -0.0856936, 0.1198668
3: -0.0660703, 0.0977019, -0.0168171, 0.0634604, -0.1295307, 0.1145190
4: -0.0406218, 0.1327123, -0.0218024, 0.0532283, -0.0938501, 0.1545147

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0080052, 0.0442183, -0.0481854, 0.0593608
1: -0.0276524, 0.0931255, -0.0045802, 0.0743399, -0.1019923, 0.0977057
2: -0.0123972, 0.1293659, 0.0050475, 0.0742267, -0.0866239, 0.1243185
3: -0.0660703, 0.0977019, -0.0188669, 0.0684462, -0.1345165, 0.1165688
4: -0.0406218, 0.1327123, -0.0224565, 0.0562747, -0.0968964, 0.1551688

Time for backsubstitution: 2.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0727767
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0727767
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040268, 0.0608585, 0.0130052, 0.0433223, -0.0473491, 0.0478533
1: -0.0240528, 0.0833467, 0.0019701, 0.0680764, -0.0921292, 0.0813766
2: -0.0102135, 0.1185798, 0.0118987, 0.0761565, -0.0863701, 0.1066811
3: -0.0473803, 0.0734831, -0.0178691, 0.0586825, -0.1060628, 0.0913523
4: -0.0333277, 0.1139941, -0.0175245, 0.0554511, -0.0887788, 0.1315185

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040268, 0.0608585, 0.0089256, 0.0436134, -0.0476403, 0.0519329
1: -0.0240528, 0.0833467, -0.0056828, 0.0719095, -0.0959624, 0.0890295
2: -0.0102135, 0.1185798, 0.0094991, 0.0732964, -0.0835099, 0.1090807
3: -0.0473803, 0.0734831, -0.0168171, 0.0634604, -0.1108407, 0.0903002
4: -0.0333277, 0.1139941, -0.0218024, 0.0532283, -0.0865560, 0.1357965

Time for backsubstitution: 2.14 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701906, upper bound: 0.0720962
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711634, upper bound: 0.0746760
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0024844, 0.0581586, -0.0053966, 0.0614798, -0.0639642, 0.0635552
1: -0.0264296, 0.0815245, -0.0295746, 0.0838206, -0.1102502, 0.1110991
2: -0.0051643, 0.1094249, -0.0105933, 0.1197037, -0.1248679, 0.1200182
3: -0.0423236, 0.0763173, -0.0481883, 0.0739372, -0.1162608, 0.1245056
4: -0.0359877, 0.1030843, -0.0338624, 0.1156039, -0.1515917, 0.1369467

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0721222
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0742207
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0058780, 0.0619953, -0.0673919, 0.0673578
1: -0.0295746, 0.0838206, -0.0304515, 0.0843779, -0.1139525, 0.1142721
2: -0.0105933, 0.1197037, -0.0112683, 0.1210694, -0.1316626, 0.1309720
3: -0.0481883, 0.0739372, -0.0502359, 0.0748101, -0.1229984, 0.1241731
4: -0.0338624, 0.1156039, -0.0342595, 0.1180273, -0.1518897, 0.1498634

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0012393, 0.0595270, -0.0654050, 0.0632346
1: -0.0304515, 0.0843779, -0.0242316, 0.0825824, -0.1130340, 0.1086095
2: -0.0112683, 0.1210694, -0.0079983, 0.1184319, -0.1297002, 0.1290677
3: -0.0502359, 0.0748101, -0.0566117, 0.0876408, -0.1378767, 0.1314218
4: -0.0342595, 0.1180273, -0.0306006, 0.1174072, -0.1516666, 0.1486279

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712027, upper bound: 0.0722356
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720780, upper bound: 0.0741778
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467
time: 0.42 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 5.95 seconds
IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0717108
IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740310
IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0720385, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A2, status: Status.VERIFIED, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0727767
IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0727767
IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0701906, upper bound: 0.0720962
IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0711634, upper bound: 0.0746760
IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0721222
IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0712027, upper bound: 0.0722356
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0720780, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0720818, upper bound: 0.0745388
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.95
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467

## BFS IS instance: IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0045362, 0.0732913, -0.0039670, 0.0673659, -0.0719021, 0.0772583
1: -0.0342618, 0.0992907, -0.0276524, 0.0931255, -0.1273873, 0.1269430
2: -0.0123451, 0.1432838, -0.0123972, 0.1293659, -0.1417110, 0.1556810
3: -0.0837640, 0.1012131, -0.0660703, 0.0977019, -0.1814659, 0.1672834
4: -0.0400739, 0.1559919, -0.0406218, 0.1327123, -0.1727862, 0.1966137

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0702994, upper bound: 0.0726668
time: 0.36 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740311
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0036633, 0.0672161, 0.0119214, 0.0440228, -0.0476861, 0.0552947
1: -0.0272224, 0.0929019, 0.0005230, 0.0695521, -0.0967745, 0.0923789
2: -0.0120373, 0.1289738, 0.0096441, 0.0773689, -0.0894062, 0.1193298
3: -0.0653339, 0.0973650, -0.0194770, 0.0624470, -0.1277809, 0.1168420
4: -0.0405064, 0.1320211, -0.0188315, 0.0573395, -0.0978459, 0.1508526

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0036633, 0.0672161, 0.0089256, 0.0436134, -0.0472768, 0.0582905
1: -0.0272224, 0.0929019, -0.0056828, 0.0719095, -0.0991319, 0.0985847
2: -0.0120373, 0.1289738, 0.0094991, 0.0732964, -0.0853337, 0.1194747
3: -0.0653339, 0.0973650, -0.0168171, 0.0634604, -0.1287943, 0.1141820
4: -0.0405064, 0.1320211, -0.0218024, 0.0532283, -0.0937347, 0.1538236

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0741902
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040268, 0.0608585, 0.0130052, 0.0433223, -0.0473491, 0.0478533
1: -0.0240528, 0.0833467, 0.0019701, 0.0680764, -0.0921292, 0.0813766
2: -0.0102135, 0.1185798, 0.0118987, 0.0761565, -0.0863701, 0.1066811
3: -0.0473803, 0.0734831, -0.0178691, 0.0586825, -0.1060628, 0.0913523
4: -0.0333277, 0.1139941, -0.0175245, 0.0554511, -0.0887788, 0.1315185

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040268, 0.0608585, 0.0089256, 0.0436134, -0.0476403, 0.0519329
1: -0.0240528, 0.0833467, -0.0056828, 0.0719095, -0.0959624, 0.0890295
2: -0.0102135, 0.1185798, 0.0094991, 0.0732964, -0.0835099, 0.1090807
3: -0.0473803, 0.0734831, -0.0168171, 0.0634604, -0.1108407, 0.0903002
4: -0.0333277, 0.1139941, -0.0218024, 0.0532283, -0.0865560, 0.1357965

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0119214, 0.0440228, -0.0479898, 0.0554446
1: -0.0276524, 0.0931255, 0.0005230, 0.0695521, -0.0972045, 0.0926026
2: -0.0123972, 0.1293659, 0.0096441, 0.0773689, -0.0897661, 0.1197219
3: -0.0660703, 0.0977019, -0.0194770, 0.0624470, -0.1285173, 0.1171789
4: -0.0406218, 0.1327123, -0.0188315, 0.0573395, -0.0979612, 0.1515437

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0089256, 0.0436134, -0.0475805, 0.0584403
1: -0.0276524, 0.0931255, -0.0056828, 0.0719095, -0.0995619, 0.0988083
2: -0.0123972, 0.1293659, 0.0094991, 0.0732964, -0.0856936, 0.1198668
3: -0.0660703, 0.0977019, -0.0168171, 0.0634604, -0.1295307, 0.1145190
4: -0.0406218, 0.1327123, -0.0218024, 0.0532283, -0.0938501, 0.1545147

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0741902
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0100352, 0.0447484, -0.0487154, 0.0573308
1: -0.0276524, 0.0931255, 0.0015434, 0.0713148, -0.0989672, 0.0915821
2: -0.0123972, 0.1293659, 0.0057138, 0.0774657, -0.0898629, 0.1236522
3: -0.0660703, 0.0977019, -0.0229581, 0.0693600, -0.1354303, 0.1206600
4: -0.0406218, 0.1327123, -0.0192196, 0.0607713, -0.1013930, 0.1519319

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0717476, upper bound: 0.0723384
time: 0.33 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720385, upper bound: 0.0746514
time: 0.35 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0080052, 0.0442183, -0.0481854, 0.0593608
1: -0.0276524, 0.0931255, -0.0045802, 0.0743399, -0.1019923, 0.0977057
2: -0.0123972, 0.1293659, 0.0050475, 0.0742267, -0.0866239, 0.1243185
3: -0.0660703, 0.0977019, -0.0188669, 0.0684462, -0.1345165, 0.1165688
4: -0.0406218, 0.1327123, -0.0224565, 0.0562747, -0.0968964, 0.1551688

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038033, 0.0607403, 0.0130052, 0.0433223, -0.0471255, 0.0477351
1: -0.0237384, 0.0831125, 0.0019701, 0.0680764, -0.0918148, 0.0811424
2: -0.0098275, 0.1182602, 0.0118987, 0.0761565, -0.0859840, 0.1063615
3: -0.0466977, 0.0731392, -0.0178691, 0.0586825, -0.1053802, 0.0910084
4: -0.0332167, 0.1134348, -0.0175245, 0.0554511, -0.0886678, 0.1309593

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040268, 0.0608585, 0.0089256, 0.0436134, -0.0476403, 0.0519329
1: -0.0240528, 0.0833467, -0.0056828, 0.0719095, -0.0959624, 0.0890295
2: -0.0102135, 0.1185798, 0.0094991, 0.0732964, -0.0835099, 0.1090807
3: -0.0473803, 0.0734831, -0.0168171, 0.0634604, -0.1108407, 0.0903002
4: -0.0333277, 0.1139941, -0.0218024, 0.0532283, -0.0865560, 0.1357965

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711634, upper bound: 0.0746760
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0024844, 0.0581586, -0.0053966, 0.0614798, -0.0639642, 0.0635552
1: -0.0264296, 0.0815245, -0.0295746, 0.0838206, -0.1102502, 0.1110991
2: -0.0051643, 0.1094249, -0.0105933, 0.1197037, -0.1248679, 0.1200182
3: -0.0423236, 0.0763173, -0.0481883, 0.0739372, -0.1162608, 0.1245056
4: -0.0359877, 0.1030843, -0.0338624, 0.1156039, -0.1515917, 0.1369467

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0724500
time: 0.34 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0742207
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0017990, 0.0603538, -0.0657504, 0.0632788
1: -0.0295746, 0.0838206, -0.0261768, 0.0834607, -0.1130353, 0.1099974
2: -0.0105933, 0.1197037, -0.0086028, 0.1206991, -0.1312923, 0.1283065
3: -0.0481883, 0.0739372, -0.0593277, 0.0886848, -0.1368731, 0.1332649
4: -0.0338624, 0.1156039, -0.0311766, 0.1211701, -0.1550325, 0.1467805

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716324, upper bound: 0.0724500
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716324, upper bound: 0.0742207
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0058780, 0.0619953, -0.0673919, 0.0673578
1: -0.0295746, 0.0838206, -0.0304515, 0.0843779, -0.1139525, 0.1142721
2: -0.0105933, 0.1197037, -0.0112683, 0.1210694, -0.1316626, 0.1309720
3: -0.0481883, 0.0739372, -0.0502359, 0.0748101, -0.1229984, 0.1241731
4: -0.0338624, 0.1156039, -0.0342595, 0.1180273, -0.1518897, 0.1498634

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0000952, 0.0585321, -0.0644101, 0.0620905
1: -0.0304515, 0.0843779, -0.0215799, 0.0819260, -0.1123775, 0.1059579
2: -0.0112683, 0.1210694, -0.0072518, 0.1159492, -0.1272175, 0.1283212
3: -0.0502359, 0.0748101, -0.0540064, 0.0868932, -0.1371291, 0.1288164
4: -0.0342595, 0.1180273, -0.0299066, 0.1133695, -0.1476290, 0.1479339

Time for backsubstitution: 2.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711015, upper bound: 0.0745032
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716324, upper bound: 0.0724500
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716905, upper bound: 0.0747509
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467
time: 0.40 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 6.93 seconds
IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0702994, upper bound: 0.0726668
IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740311
IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0720385, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0711634, upper bound: 0.0746760
IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0716324, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0716324, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0716324, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 6.93
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467

## BFS IS instance: IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0045362, 0.0732913, -0.0039670, 0.0673659, -0.0719021, 0.0772583
1: -0.0342618, 0.0992907, -0.0276524, 0.0931255, -0.1273873, 0.1269430
2: -0.0123451, 0.1432838, -0.0123972, 0.1293659, -0.1417110, 0.1556810
3: -0.0837640, 0.1012131, -0.0660703, 0.0977019, -0.1814659, 0.1672834
4: -0.0400739, 0.1559919, -0.0406218, 0.1327123, -0.1727862, 0.1966137

Time for backsubstitution: 2.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 47
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712157, upper bound: 0.0717108
time: 0.38 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720483, upper bound: 0.0740310
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0036633, 0.0672161, 0.0119214, 0.0440228, -0.0476861, 0.0552947
1: -0.0272224, 0.0929019, 0.0005230, 0.0695521, -0.0967745, 0.0923789
2: -0.0120373, 0.1289738, 0.0096441, 0.0773689, -0.0894062, 0.1193298
3: -0.0653339, 0.0973650, -0.0194770, 0.0624470, -0.1277809, 0.1168420
4: -0.0405064, 0.1320211, -0.0188315, 0.0573395, -0.0978459, 0.1508526

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038033, 0.0607403, 0.0089256, 0.0436134, -0.0474167, 0.0518147
1: -0.0237384, 0.0831125, -0.0056828, 0.0719095, -0.0956479, 0.0887953
2: -0.0098275, 0.1182602, 0.0094991, 0.0732964, -0.0831238, 0.1087611
3: -0.0466977, 0.0731392, -0.0168171, 0.0634604, -0.1101581, 0.0899563
4: -0.0332167, 0.1134348, -0.0218024, 0.0532283, -0.0864450, 0.1352372

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038033, 0.0607403, 0.0130052, 0.0433223, -0.0471255, 0.0477351
1: -0.0237384, 0.0831125, 0.0019701, 0.0680764, -0.0918148, 0.0811424
2: -0.0098275, 0.1182602, 0.0118987, 0.0761565, -0.0859840, 0.1063615
3: -0.0466977, 0.0731392, -0.0178691, 0.0586825, -0.1053802, 0.0910084
4: -0.0332167, 0.1134348, -0.0175245, 0.0554511, -0.0886678, 0.1309593

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040268, 0.0608585, 0.0089256, 0.0436134, -0.0476403, 0.0519329
1: -0.0240528, 0.0833467, -0.0056828, 0.0719095, -0.0959624, 0.0890295
2: -0.0102135, 0.1185798, 0.0094991, 0.0732964, -0.0835099, 0.1090807
3: -0.0473803, 0.0734831, -0.0168171, 0.0634604, -0.1108407, 0.0903002
4: -0.0333277, 0.1139941, -0.0218024, 0.0532283, -0.0865560, 0.1357965

Time for backsubstitution: 2.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0036633, 0.0672161, 0.0119214, 0.0440228, -0.0476861, 0.0552947
1: -0.0272224, 0.0929019, 0.0005230, 0.0695521, -0.0967745, 0.0923789
2: -0.0120373, 0.1289738, 0.0096441, 0.0773689, -0.0894062, 0.1193298
3: -0.0653339, 0.0973650, -0.0194770, 0.0624470, -0.1277809, 0.1168420
4: -0.0405064, 0.1320211, -0.0188315, 0.0573395, -0.0978459, 0.1508526

Time for backsubstitution: 2.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040268, 0.0608585, 0.0089256, 0.0436134, -0.0476403, 0.0519329
1: -0.0240528, 0.0833467, -0.0056828, 0.0719095, -0.0959624, 0.0890295
2: -0.0102135, 0.1185798, 0.0094991, 0.0732964, -0.0835099, 0.1090807
3: -0.0473803, 0.0734831, -0.0168171, 0.0634604, -0.1108407, 0.0903002
4: -0.0333277, 0.1139941, -0.0218024, 0.0532283, -0.0865560, 0.1357965

Time for backsubstitution: 2.08 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0119214, 0.0440228, -0.0479898, 0.0554446
1: -0.0276524, 0.0931255, 0.0005230, 0.0695521, -0.0972045, 0.0926026
2: -0.0123972, 0.1293659, 0.0096441, 0.0773689, -0.0897661, 0.1197219
3: -0.0660703, 0.0977019, -0.0194770, 0.0624470, -0.1285173, 0.1171789
4: -0.0406218, 0.1327123, -0.0188315, 0.0573395, -0.0979612, 0.1515437

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0100352, 0.0447484, -0.0487154, 0.0573308
1: -0.0276524, 0.0931255, 0.0015434, 0.0713148, -0.0989672, 0.0915821
2: -0.0123972, 0.1293659, 0.0057138, 0.0774657, -0.0898629, 0.1236522
3: -0.0660703, 0.0977019, -0.0229581, 0.0693600, -0.1354303, 0.1206600
4: -0.0406218, 0.1327123, -0.0192196, 0.0607713, -0.1013930, 0.1519319

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720385, upper bound: 0.0746514
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0089256, 0.0436134, -0.0475805, 0.0584403
1: -0.0276524, 0.0931255, -0.0056828, 0.0719095, -0.0995619, 0.0988083
2: -0.0123972, 0.1293659, 0.0094991, 0.0732964, -0.0856936, 0.1198668
3: -0.0660703, 0.0977019, -0.0168171, 0.0634604, -0.1295307, 0.1145190
4: -0.0406218, 0.1327123, -0.0218024, 0.0532283, -0.0938501, 0.1545147

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720962
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0080052, 0.0442183, -0.0481854, 0.0593608
1: -0.0276524, 0.0931255, -0.0045802, 0.0743399, -0.1019923, 0.0977057
2: -0.0123972, 0.1293659, 0.0050475, 0.0742267, -0.0866239, 0.1243185
3: -0.0660703, 0.0977019, -0.0188669, 0.0684462, -0.1345165, 0.1165688
4: -0.0406218, 0.1327123, -0.0224565, 0.0562747, -0.0968964, 0.1551688

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0727767
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0727767
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0038033, 0.0607403, 0.0130052, 0.0433223, -0.0471255, 0.0477351
1: -0.0237384, 0.0831125, 0.0019701, 0.0680764, -0.0918148, 0.0811424
2: -0.0098275, 0.1182602, 0.0118987, 0.0761565, -0.0859840, 0.1063615
3: -0.0466977, 0.0731392, -0.0178691, 0.0586825, -0.1053802, 0.0910084
4: -0.0332167, 0.1134348, -0.0175245, 0.0554511, -0.0886678, 0.1309593

Time for backsubstitution: 2.10 seconds
Binary search (step 0): status=Status.UNKNOWN, low=0.0159454, high=0.0988818, mid=0.0988818, abs_max=0.09181444346904755
rel_dist={0: [-0.07549650593465267, 0.07549650593465268]}

## Binary search (step 1) starts
Candidate diff: 0.0574136


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750627, upper bound: 0.0732826
time: 0.30 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750418, upper bound: 0.0750418
time: 0.29 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.76 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0750627, upper bound: 0.0732826
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.76
Output dim: 0, lower bound: -0.0750418, upper bound: 0.0750418

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0056712, 0.0494482, -0.0122710, 0.0795435, -0.0738723, 0.0617192
1: -0.0048309, 0.0792722, -0.0482301, 0.1046001, -0.1094310, 0.1275024
2: 0.0013121, 0.0852126, -0.0215323, 0.1538582, -0.1525461, 0.1067449
3: -0.0302970, 0.0797933, -0.0898915, 0.1120336, -0.1423306, 0.1696847
4: -0.0295759, 0.0721076, -0.0514701, 0.1717802, -0.2013561, 0.1235777

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0735765, upper bound: 0.0720345
time: 0.32 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0735765, upper bound: 0.0731040
time: 0.31 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0080689, 0.0722086, -0.0122710, 0.0795435, -0.0876124, 0.0844796
1: -0.0359785, 0.0976419, -0.0482301, 0.1046001, -0.1405786, 0.1458721
2: -0.0166659, 0.1399382, -0.0215323, 0.1538582, -0.1705241, 0.1614705
3: -0.0768474, 0.1041032, -0.0898915, 0.1120336, -0.1888810, 0.1939946
4: -0.0455294, 0.1494691, -0.0514701, 0.1717802, -0.2173096, 0.2009392

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0735773, upper bound: 0.0738308
time: 0.31 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0748532, upper bound: 0.0748532
time: 0.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.66 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.0735765, upper bound: 0.0720345
IS_A1_B2, status: Status.VERIFIED, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.0735765, upper bound: 0.0731040
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.0735773, upper bound: 0.0738308
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.66
Output dim: 0, lower bound: -0.0748532, upper bound: 0.0748532

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0080689, 0.0722086, -0.0080926, 0.0739747, -0.0820436, 0.0803013
1: -0.0359785, 0.0976419, -0.0402663, 0.0988584, -0.1348368, 0.1379082
2: -0.0166659, 0.1399382, -0.0162963, 0.1430283, -0.1596942, 0.1562345
3: -0.0768474, 0.1041032, -0.0793115, 0.1036161, -0.1804635, 0.1834147
4: -0.0455294, 0.1494691, -0.0462731, 0.1545721, -0.2001015, 0.1957422

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0738308, upper bound: 0.0735773
time: 0.30 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0738308, upper bound: 0.0748532
time: 0.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.63 seconds
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.63
Output dim: 0, lower bound: -0.0738308, upper bound: 0.0735773
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -0.0738308, upper bound: 0.0748532

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0080926, 0.0739747, -0.0789240, 0.0765457
1: -0.0301443, 0.0939154, -0.0402663, 0.0988584, -0.1290027, 0.1341817
2: -0.0130560, 0.1317879, -0.0162963, 0.1430283, -0.1560843, 0.1480842
3: -0.0685425, 0.0983886, -0.0793115, 0.1036161, -0.1721586, 0.1777001
4: -0.0412280, 0.1366114, -0.0462731, 0.1545721, -0.1958002, 0.1828845

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0729613
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0729613
time: 0.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.66 seconds
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0729613
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0729613
Binary search (step 1): status=Status.VERIFIED, low=0.0574136, high=0.0988818, mid=0.0574136, abs_max=0.09181444346904755
rel_dist={0: [-0.07506273618631137, 0.07506273618631135]}

## Binary search (step 2) starts
Candidate diff: 0.0781477


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754565, upper bound: 0.0734774
time: 0.34 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753887, upper bound: 0.0753887
time: 0.30 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -0.0754565, upper bound: 0.0734774
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.82
Output dim: 0, lower bound: -0.0753887, upper bound: 0.0753887

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0056712, 0.0494482, -0.0122710, 0.0795435, -0.0738723, 0.0617192
1: -0.0048309, 0.0792722, -0.0482301, 0.1046001, -0.1094310, 0.1275024
2: 0.0013121, 0.0852126, -0.0215323, 0.1538582, -0.1525461, 0.1067449
3: -0.0302970, 0.0797933, -0.0898915, 0.1120336, -0.1423306, 0.1696847
4: -0.0295759, 0.0721076, -0.0514701, 0.1717802, -0.2013561, 0.1235777

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0740327, upper bound: 0.0720345
time: 0.32 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751727, upper bound: 0.0732195
time: 0.33 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0080689, 0.0722086, -0.0122710, 0.0795435, -0.0876124, 0.0844796
1: -0.0359785, 0.0976419, -0.0482301, 0.1046001, -0.1405786, 0.1458721
2: -0.0166659, 0.1399382, -0.0215323, 0.1538582, -0.1705241, 0.1614705
3: -0.0768474, 0.1041032, -0.0898915, 0.1120336, -0.1888810, 0.1939946
4: -0.0455294, 0.1494691, -0.0514701, 0.1717802, -0.2173096, 0.2009392

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0739507, upper bound: 0.0739688
time: 0.32 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751727, upper bound: 0.0751727
time: 0.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.68 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -0.0740327, upper bound: 0.0720345
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -0.0751727, upper bound: 0.0732195
IS_A2_A1, status: Status.VERIFIED, split count: 2, time: 2.68
Output dim: 0, lower bound: -0.0739507, upper bound: 0.0739688
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.68
Output dim: 0, lower bound: -0.0751727, upper bound: 0.0751727

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0059273, 0.0489539, -0.0096398, 0.0840788, -0.0781515, 0.0585937
1: -0.0041420, 0.0787851, -0.0483434, 0.1096082, -0.1137502, 0.1271285
2: 0.0015739, 0.0841584, -0.0175393, 0.1624446, -0.1608707, 0.1016976
3: -0.0293150, 0.0788381, -0.1008122, 0.1153759, -0.1446909, 0.1796503
4: -0.0287044, 0.0704413, -0.0483946, 0.1864471, -0.2151515, 0.1188359

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0735257, upper bound: 0.0720345
time: 0.32 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0735257, upper bound: 0.0720345
time: 0.32 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0056712, 0.0494482, -0.0080926, 0.0739747, -0.0683034, 0.0575409
1: -0.0048309, 0.0792722, -0.0402663, 0.0988584, -0.1036893, 0.1195385
2: 0.0013121, 0.0852126, -0.0162963, 0.1430283, -0.1417162, 0.1015089
3: -0.0302970, 0.0797933, -0.0793115, 0.1036161, -0.1339131, 0.1591048
4: -0.0295759, 0.0721076, -0.0462731, 0.1545721, -0.1841480, 0.1183806

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732195, upper bound: 0.0732195
time: 0.31 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732195, upper bound: 0.0732195
time: 0.32 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0122710, 0.0795435, -0.0844928, 0.0807240
1: -0.0301443, 0.0939154, -0.0482301, 0.1046001, -0.1347444, 0.1421455
2: -0.0130560, 0.1317879, -0.0215323, 0.1538582, -0.1669142, 0.1533202
3: -0.0685425, 0.0983886, -0.0898915, 0.1120336, -0.1805761, 0.1882800
4: -0.0412280, 0.1366114, -0.0514701, 0.1717802, -0.2130082, 0.1880815

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0739688, upper bound: 0.0739507
time: 0.30 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0739688, upper bound: 0.0751727
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.64 seconds
IS_A1_B1_B1, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 0, lower bound: -0.0735257, upper bound: 0.0720345
IS_A1_B1_B2, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 0, lower bound: -0.0735257, upper bound: 0.0720345
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 0, lower bound: -0.0732195, upper bound: 0.0732195
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 0, lower bound: -0.0732195, upper bound: 0.0732195
IS_A2_A2_B1, status: Status.VERIFIED, split count: 3, time: 2.64
Output dim: 0, lower bound: -0.0739688, upper bound: 0.0739507
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.64
Output dim: 0, lower bound: -0.0739688, upper bound: 0.0751727

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0080926, 0.0739747, -0.0789240, 0.0765457
1: -0.0301443, 0.0939154, -0.0402663, 0.0988584, -0.1290027, 0.1341817
2: -0.0130560, 0.1317879, -0.0162963, 0.1430283, -0.1560843, 0.1480842
3: -0.0685425, 0.0983886, -0.0793115, 0.1036161, -0.1721586, 0.1777001
4: -0.0412280, 0.1366114, -0.0462731, 0.1545721, -0.1958002, 0.1828845

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0730853, upper bound: 0.0739507
time: 0.33 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730853, upper bound: 0.0751727
time: 0.30 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.72 seconds
IS_A2_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0730853, upper bound: 0.0739507
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.72
Output dim: 0, lower bound: -0.0730853, upper bound: 0.0751727

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0049493, 0.0684531, -0.0734024, 0.0734024
1: -0.0301443, 0.0939154, -0.0301443, 0.0939154, -0.1240597, 0.1240597
2: -0.0130560, 0.1317879, -0.0130560, 0.1317879, -0.1448440, 0.1448440
3: -0.0685425, 0.0983886, -0.0685425, 0.0983886, -0.1669310, 0.1669310
4: -0.0412280, 0.1366114, -0.0412280, 0.1366114, -0.1778394, 0.1778394

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 1.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0725221
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
time: 0.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.69 seconds
IS_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.69
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0725221
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.69
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0048533, 0.0683171, -0.0741951, 0.0668486
1: -0.0304515, 0.0843779, -0.0300604, 0.0936913, -0.1241428, 0.1144384
2: -0.0112683, 0.1210694, -0.0129442, 0.1315948, -0.1428631, 0.1340136
3: -0.0502359, 0.0748101, -0.0683023, 0.0977666, -0.1480024, 0.1431123
4: -0.0342595, 0.1180273, -0.0410903, 0.1363595, -0.1706190, 0.1591176

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720762, upper bound: 0.0745388
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720762, upper bound: 0.0748467
time: 0.37 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.28 seconds
IS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0720762, upper bound: 0.0745388
IS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.28
Output dim: 0, lower bound: -0.0720762, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711015, upper bound: 0.0745032
time: 0.33 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743948
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716606, upper bound: 0.0747509
time: 0.33 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467
time: 0.42 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.91 seconds
IS_A2_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.91
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743948
IS_A2_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.91
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.91
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.91
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0024844, 0.0581586, -0.0021452, 0.0612259, -0.0637102, 0.0603038
1: -0.0264296, 0.0815245, -0.0263871, 0.0842858, -0.1107154, 0.1079116
2: -0.0051643, 0.1094249, -0.0090670, 0.1218928, -0.1270570, 0.1184920
3: -0.0423236, 0.0763173, -0.0604222, 0.0902384, -0.1325620, 0.1367395
4: -0.0359877, 0.1030843, -0.0321517, 0.1232046, -0.1591924, 0.1352361

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743949
time: 0.33 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743949
time: 0.34 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0021452, 0.0612259, -0.0666225, 0.0636251
1: -0.0295746, 0.0838206, -0.0263871, 0.0842858, -0.1138604, 0.1102077
2: -0.0105933, 0.1197037, -0.0090670, 0.1218928, -0.1324860, 0.1287707
3: -0.0481883, 0.0739372, -0.0604222, 0.0902384, -0.1384267, 0.1343594
4: -0.0338624, 0.1156039, -0.0321517, 0.1232046, -0.1570670, 0.1477557

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719711, upper bound: 0.0742207
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720762, upper bound: 0.0745388
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467
time: 0.38 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 5.55 seconds
IS_A2_A2_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743949
IS_A2_A2_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743949
IS_A2_A2_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0719711, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0720762, upper bound: 0.0745388
IS_A2_A2_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 5.55
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0024844, 0.0581586, -0.0021452, 0.0612259, -0.0637102, 0.0603038
1: -0.0264296, 0.0815245, -0.0263871, 0.0842858, -0.1107154, 0.1079116
2: -0.0051643, 0.1094249, -0.0090670, 0.1218928, -0.1270570, 0.1184920
3: -0.0423236, 0.0763173, -0.0604222, 0.0902384, -0.1325620, 0.1367395
4: -0.0359877, 0.1030843, -0.0321517, 0.1232046, -0.1591924, 0.1352361

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0708850, upper bound: 0.0721222
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0708850, upper bound: 0.0721222
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0024844, 0.0581586, -0.0058780, 0.0619953, -0.0644797, 0.0640366
1: -0.0264296, 0.0815245, -0.0304515, 0.0843779, -0.1108075, 0.1119760
2: -0.0051643, 0.1094249, -0.0112683, 0.1210694, -0.1262336, 0.1206933
3: -0.0423236, 0.0763173, -0.0502359, 0.0748101, -0.1171336, 0.1265531
4: -0.0359877, 0.1030843, -0.0342595, 0.1180273, -0.1540150, 0.1373438

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711798, upper bound: 0.0726242
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711798, upper bound: 0.0743949
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0021452, 0.0612259, -0.0666225, 0.0636251
1: -0.0295746, 0.0838206, -0.0263871, 0.0842858, -0.1138604, 0.1102077
2: -0.0105933, 0.1197037, -0.0090670, 0.1218928, -0.1324860, 0.1287707
3: -0.0481883, 0.0739372, -0.0604222, 0.0902384, -0.1384267, 0.1343594
4: -0.0338624, 0.1156039, -0.0321517, 0.1232046, -0.1570670, 0.1477557

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724658
time: 0.34 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0058780, 0.0619953, -0.0673919, 0.0673578
1: -0.0295746, 0.0838206, -0.0304515, 0.0843779, -0.1139525, 0.1142721
2: -0.0105933, 0.1197037, -0.0112683, 0.1210694, -0.1316626, 0.1309720
3: -0.0481883, 0.0739372, -0.0502359, 0.0748101, -0.1229984, 0.1241731
4: -0.0338624, 0.1156039, -0.0342595, 0.1180273, -0.1518897, 0.1498634

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0713339, upper bound: 0.0741086
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0742207
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711015, upper bound: 0.0745032
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716606, upper bound: 0.0747509
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467
time: 0.42 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 6.18 seconds
IS_A2_A2_B2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 6.18
Output dim: 0, lower bound: -0.0708850, upper bound: 0.0721222
IS_A2_A2_B2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 6.18
Output dim: 0, lower bound: -0.0708850, upper bound: 0.0721222
IS_A2_A2_B2_B2_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 9, time: 6.18
Output dim: 0, lower bound: -0.0711798, upper bound: 0.0726242
IS_A2_A2_B2_B2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 6.18
Output dim: 0, lower bound: -0.0711798, upper bound: 0.0743949
IS_A2_A2_B2_B2_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 6.18
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 9, time: 6.18
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724658
IS_A2_A2_B2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 6.18
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 6.18
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 6.18
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 6.18
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 6.18
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 6.18
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0024844, 0.0581586, -0.0053966, 0.0614798, -0.0639642, 0.0635552
1: -0.0264296, 0.0815245, -0.0295746, 0.0838206, -0.1102502, 0.1110991
2: -0.0051643, 0.1094249, -0.0105933, 0.1197037, -0.1248679, 0.1200182
3: -0.0423236, 0.0763173, -0.0481883, 0.0739372, -0.1162608, 0.1245056
4: -0.0359877, 0.1030843, -0.0338624, 0.1156039, -0.1515917, 0.1369467

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0721222
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0708031
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0058780, 0.0619953, -0.0673919, 0.0673578
1: -0.0295746, 0.0838206, -0.0304515, 0.0843779, -0.1139525, 0.1142721
2: -0.0105933, 0.1197037, -0.0112683, 0.1210694, -0.1316626, 0.1309720
3: -0.0481883, 0.0739372, -0.0502359, 0.0748101, -0.1229984, 0.1241731
4: -0.0338624, 0.1156039, -0.0342595, 0.1180273, -0.1518897, 0.1498634

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0012393, 0.0595270, -0.0654050, 0.0632346
1: -0.0304515, 0.0843779, -0.0242316, 0.0825824, -0.1130340, 0.1086095
2: -0.0112683, 0.1210694, -0.0079983, 0.1184319, -0.1297002, 0.1290677
3: -0.0502359, 0.0748101, -0.0566117, 0.0876408, -0.1378767, 0.1314218
4: -0.0342595, 0.1180273, -0.0306006, 0.1174072, -0.1516666, 0.1486279

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0709314, upper bound: 0.0722356
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718959, upper bound: 0.0741778
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720762, upper bound: 0.0745388
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467
time: 0.42 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 5.88 seconds
IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 10, time: 5.88
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0721222
IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 10, time: 5.88
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0708031
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.88
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.88
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 10, time: 5.88
Output dim: 0, lower bound: -0.0709314, upper bound: 0.0722356
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 5.88
Output dim: 0, lower bound: -0.0718959, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 5.88
Output dim: 0, lower bound: -0.0720762, upper bound: 0.0745388
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 5.88
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0017990, 0.0603538, -0.0657504, 0.0632788
1: -0.0295746, 0.0838206, -0.0261768, 0.0834607, -0.1130353, 0.1099974
2: -0.0105933, 0.1197037, -0.0086028, 0.1206991, -0.1312923, 0.1283065
3: -0.0481883, 0.0739372, -0.0593277, 0.0886848, -0.1368731, 0.1332649
4: -0.0338624, 0.1156039, -0.0311766, 0.1211701, -0.1550325, 0.1467805

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0058780, 0.0619953, -0.0673919, 0.0673578
1: -0.0295746, 0.0838206, -0.0304515, 0.0843779, -0.1139525, 0.1142721
2: -0.0105933, 0.1197037, -0.0112683, 0.1210694, -0.1316626, 0.1309720
3: -0.0481883, 0.0739372, -0.0502359, 0.0748101, -0.1229984, 0.1241731
4: -0.0338624, 0.1156039, -0.0342595, 0.1180273, -0.1518897, 0.1498634

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718942, upper bound: 0.0742207
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0000952, 0.0585321, -0.0644101, 0.0620905
1: -0.0304515, 0.0843779, -0.0215799, 0.0819260, -0.1123775, 0.1059579
2: -0.0112683, 0.1210694, -0.0072518, 0.1159492, -0.1272175, 0.1283212
3: -0.0502359, 0.0748101, -0.0540064, 0.0868932, -0.1371291, 0.1288164
4: -0.0342595, 0.1180273, -0.0299066, 0.1133695, -0.1476290, 0.1479339

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711015, upper bound: 0.0745032
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716606, upper bound: 0.0747509
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467
time: 0.39 seconds

## Summary of splitting at layer (split count: 10)
- Time for IS candidates: 6.11 seconds
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 11, time: 6.11
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 11, time: 6.11
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 6.11
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 6.11
Output dim: 0, lower bound: -0.0718942, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 11, time: 6.11
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 11, time: 6.11
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 11, time: 6.11
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 11, time: 6.11
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 11, time: 6.11
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 11, time: 6.11
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0012393, 0.0595270, -0.0649236, 0.0627191
1: -0.0295746, 0.0838206, -0.0242316, 0.0825824, -0.1121571, 0.1080522
2: -0.0105933, 0.1197037, -0.0079983, 0.1184319, -0.1290252, 0.1277020
3: -0.0481883, 0.0739372, -0.0566117, 0.0876408, -0.1358292, 0.1305490
4: -0.0338624, 0.1156039, -0.0306006, 0.1174072, -0.1512696, 0.1462045

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0709228, upper bound: 0.0723804
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0058780, 0.0619953, -0.0673919, 0.0673578
1: -0.0295746, 0.0838206, -0.0304515, 0.0843779, -0.1139525, 0.1142721
2: -0.0105933, 0.1197037, -0.0112683, 0.1210694, -0.1316626, 0.1309720
3: -0.0481883, 0.0739372, -0.0502359, 0.0748101, -0.1229984, 0.1241731
4: -0.0338624, 0.1156039, -0.0342595, 0.1180273, -0.1518897, 0.1498634

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0000952, 0.0585321, -0.0644101, 0.0620905
1: -0.0304515, 0.0843779, -0.0215799, 0.0819260, -0.1123775, 0.1059579
2: -0.0112683, 0.1210694, -0.0072518, 0.1159492, -0.1272175, 0.1283212
3: -0.0502359, 0.0748101, -0.0540064, 0.0868932, -0.1371291, 0.1288164
4: -0.0342595, 0.1180273, -0.0299066, 0.1133695, -0.1476290, 0.1479339

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0715084, upper bound: 0.0723445
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.46 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0030341, 0.0593020, -0.0651800, 0.0650294
1: -0.0304515, 0.0843779, -0.0222096, 0.0823854, -0.1128369, 0.1065876
2: -0.0112683, 0.1210694, -0.0092799, 0.1153993, -0.1266676, 0.1303493
3: -0.0502359, 0.0748101, -0.0432175, 0.0723896, -0.1226255, 0.1180276
4: -0.0342595, 0.1180273, -0.0328059, 0.1083915, -0.1426510, 0.1508332

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710245, upper bound: 0.0722935
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0710245, upper bound: 0.0741778
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0012393, 0.0595270, -0.0654050, 0.0632346
1: -0.0304515, 0.0843779, -0.0242316, 0.0825824, -0.1130340, 0.1086095
2: -0.0112683, 0.1210694, -0.0079983, 0.1184319, -0.1297002, 0.1290677
3: -0.0502359, 0.0748101, -0.0566117, 0.0876408, -0.1378767, 0.1314218
4: -0.0342595, 0.1180273, -0.0306006, 0.1174072, -0.1516666, 0.1486279

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0709314, upper bound: 0.0722356
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718959, upper bound: 0.0741778
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720762, upper bound: 0.0745388
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467
time: 0.40 seconds

## Summary of splitting at layer (split count: 11)
- Time for IS candidates: 5.94 seconds
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 12, time: 5.94
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 12, time: 5.94
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 5.94
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 5.94
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B1, status: Status.VERIFIED, split count: 12, time: 5.94
Output dim: 0, lower bound: -0.0715084, upper bound: 0.0723445
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 12, time: 5.94
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 12, time: 5.94
Output dim: 0, lower bound: -0.0710245, upper bound: 0.0722935
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 12, time: 5.94
Output dim: 0, lower bound: -0.0710245, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 12, time: 5.94
Output dim: 0, lower bound: -0.0709314, upper bound: 0.0722356
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 12, time: 5.94
Output dim: 0, lower bound: -0.0718959, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 12, time: 5.94
Output dim: 0, lower bound: -0.0720762, upper bound: 0.0745388
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 12, time: 5.94
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0012393, 0.0595270, -0.0649236, 0.0627191
1: -0.0295746, 0.0838206, -0.0242316, 0.0825824, -0.1121571, 0.1080522
2: -0.0105933, 0.1197037, -0.0079983, 0.1184319, -0.1290252, 0.1277020
3: -0.0481883, 0.0739372, -0.0566117, 0.0876408, -0.1358292, 0.1305490
4: -0.0338624, 0.1156039, -0.0306006, 0.1174072, -0.1512696, 0.1462045

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0053373, 0.0603769, -0.0657735, 0.0668171
1: -0.0295746, 0.0838206, -0.0293680, 0.0834937, -0.1130683, 0.1131886
2: -0.0105933, 0.1197037, -0.0105539, 0.1178839, -0.1284771, 0.1302576
3: -0.0481883, 0.0739372, -0.0462398, 0.0737956, -0.1219839, 0.1201770
4: -0.0338624, 0.1156039, -0.0338468, 0.1125477, -0.1464101, 0.1494508

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0713339, upper bound: 0.0741086
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
time: 0.53 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0742207
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0017990, 0.0603538, -0.0657504, 0.0632788
1: -0.0295746, 0.0838206, -0.0261768, 0.0834607, -0.1130353, 0.1099974
2: -0.0105933, 0.1197037, -0.0086028, 0.1206991, -0.1312923, 0.1283065
3: -0.0481883, 0.0739372, -0.0593277, 0.0886848, -0.1368731, 0.1332649
4: -0.0338624, 0.1156039, -0.0311766, 0.1211701, -0.1550325, 0.1467805

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
time: 0.44 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0058780, 0.0619953, -0.0673919, 0.0673578
1: -0.0295746, 0.0838206, -0.0304515, 0.0843779, -0.1139525, 0.1142721
2: -0.0105933, 0.1197037, -0.0112683, 0.1210694, -0.1316626, 0.1309720
3: -0.0481883, 0.0739372, -0.0502359, 0.0748101, -0.1229984, 0.1241731
4: -0.0338624, 0.1156039, -0.0342595, 0.1180273, -0.1518897, 0.1498634

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718942, upper bound: 0.0742207
time: 0.61 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0000952, 0.0585321, -0.0644101, 0.0620905
1: -0.0304515, 0.0843779, -0.0215799, 0.0819260, -0.1123775, 0.1059579
2: -0.0112683, 0.1210694, -0.0072518, 0.1159492, -0.1272175, 0.1283212
3: -0.0502359, 0.0748101, -0.0540064, 0.0868932, -0.1371291, 0.1288164
4: -0.0342595, 0.1180273, -0.0299066, 0.1133695, -0.1476290, 0.1479339

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0709314, upper bound: 0.0722356
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718959, upper bound: 0.0741778
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0030341, 0.0593020, -0.0651800, 0.0650294
1: -0.0304515, 0.0843779, -0.0222096, 0.0823854, -0.1128369, 0.1065876
2: -0.0112683, 0.1210694, -0.0092799, 0.1153993, -0.1266676, 0.1303493
3: -0.0502359, 0.0748101, -0.0432175, 0.0723896, -0.1226255, 0.1180276
4: -0.0342595, 0.1180273, -0.0328059, 0.1083915, -0.1426510, 0.1508332

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0000952, 0.0585321, -0.0644101, 0.0620905
1: -0.0304515, 0.0843779, -0.0215799, 0.0819260, -0.1123775, 0.1059579
2: -0.0112683, 0.1210694, -0.0072518, 0.1159492, -0.1272175, 0.1283212
3: -0.0502359, 0.0748101, -0.0540064, 0.0868932, -0.1371291, 0.1288164
4: -0.0342595, 0.1180273, -0.0299066, 0.1133695, -0.1476290, 0.1479339

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.46 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711015, upper bound: 0.0745032
time: 0.44 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716606, upper bound: 0.0747509
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467
time: 0.54 seconds

## Summary of splitting at layer (split count: 12)
- Time for IS candidates: 6.60 seconds
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0718942, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B1, status: Status.VERIFIED, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0709314, upper bound: 0.0722356
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0718959, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 13, time: 6.60
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0012393, 0.0595270, -0.0649236, 0.0627191
1: -0.0295746, 0.0838206, -0.0242316, 0.0825824, -0.1121571, 0.1080522
2: -0.0105933, 0.1197037, -0.0079983, 0.1184319, -0.1290252, 0.1277020
3: -0.0481883, 0.0739372, -0.0566117, 0.0876408, -0.1358292, 0.1305490
4: -0.0338624, 0.1156039, -0.0306006, 0.1174072, -0.1512696, 0.1462045

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0709228, upper bound: 0.0723804
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0053373, 0.0603769, -0.0657735, 0.0668171
1: -0.0295746, 0.0838206, -0.0293680, 0.0834937, -0.1130683, 0.1131886
2: -0.0105933, 0.1197037, -0.0105539, 0.1178839, -0.1284771, 0.1302576
3: -0.0481883, 0.0739372, -0.0462398, 0.0737956, -0.1219839, 0.1201770
4: -0.0338624, 0.1156039, -0.0338468, 0.1125477, -0.1464101, 0.1494508

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0012393, 0.0595270, -0.0649236, 0.0627191
1: -0.0295746, 0.0838206, -0.0242316, 0.0825824, -0.1121571, 0.1080522
2: -0.0105933, 0.1197037, -0.0079983, 0.1184319, -0.1290252, 0.1277020
3: -0.0481883, 0.0739372, -0.0566117, 0.0876408, -0.1358292, 0.1305490
4: -0.0338624, 0.1156039, -0.0306006, 0.1174072, -0.1512696, 0.1462045

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0709228, upper bound: 0.0723804
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0058780, 0.0619953, -0.0673919, 0.0673578
1: -0.0295746, 0.0838206, -0.0304515, 0.0843779, -0.1139525, 0.1142721
2: -0.0105933, 0.1197037, -0.0112683, 0.1210694, -0.1316626, 0.1309720
3: -0.0481883, 0.0739372, -0.0502359, 0.0748101, -0.1229984, 0.1241731
4: -0.0338624, 0.1156039, -0.0342595, 0.1180273, -0.1518897, 0.1498634

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0000952, 0.0585321, -0.0644101, 0.0620905
1: -0.0304515, 0.0843779, -0.0215799, 0.0819260, -0.1123775, 0.1059579
2: -0.0112683, 0.1210694, -0.0072518, 0.1159492, -0.1272175, 0.1283212
3: -0.0502359, 0.0748101, -0.0540064, 0.0868932, -0.1371291, 0.1288164
4: -0.0342595, 0.1180273, -0.0299066, 0.1133695, -0.1476290, 0.1479339

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0001039, 0.0580676, -0.0639456, 0.0618914
1: -0.0304515, 0.0843779, -0.0204136, 0.0815959, -0.1120474, 0.1047915
2: -0.0112683, 0.1210694, -0.0070285, 0.1146723, -0.1259406, 0.1280979
3: -0.0502359, 0.0748101, -0.0528774, 0.0865048, -0.1367407, 0.1276874
4: -0.0342595, 0.1180273, -0.0295022, 0.1115406, -0.1458001, 0.1475295

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0715084, upper bound: 0.0723445
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0030341, 0.0593020, -0.0651800, 0.0650294
1: -0.0304515, 0.0843779, -0.0222096, 0.0823854, -0.1128369, 0.1065876
2: -0.0112683, 0.1210694, -0.0092799, 0.1153993, -0.1266676, 0.1303493
3: -0.0502359, 0.0748101, -0.0432175, 0.0723896, -0.1226255, 0.1180276
4: -0.0342595, 0.1180273, -0.0328059, 0.1083915, -0.1426510, 0.1508332

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710245, upper bound: 0.0722935
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0000952, 0.0585321, -0.0644101, 0.0620905
1: -0.0304515, 0.0843779, -0.0215799, 0.0819260, -0.1123775, 0.1059579
2: -0.0112683, 0.1210694, -0.0072518, 0.1159492, -0.1272175, 0.1283212
3: -0.0502359, 0.0748101, -0.0540064, 0.0868932, -0.1371291, 0.1288164
4: -0.0342595, 0.1180273, -0.0299066, 0.1133695, -0.1476290, 0.1479339

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0715084, upper bound: 0.0723445
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0030341, 0.0593020, -0.0651800, 0.0650294
1: -0.0304515, 0.0843779, -0.0222096, 0.0823854, -0.1128369, 0.1065876
2: -0.0112683, 0.1210694, -0.0092799, 0.1153993, -0.1266676, 0.1303493
3: -0.0502359, 0.0748101, -0.0432175, 0.0723896, -0.1226255, 0.1180276
4: -0.0342595, 0.1180273, -0.0328059, 0.1083915, -0.1426510, 0.1508332

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710245, upper bound: 0.0722935
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0710245, upper bound: 0.0741778
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0012393, 0.0595270, -0.0654050, 0.0632346
1: -0.0304515, 0.0843779, -0.0242316, 0.0825824, -0.1130340, 0.1086095
2: -0.0112683, 0.1210694, -0.0079983, 0.1184319, -0.1297002, 0.1290677
3: -0.0502359, 0.0748101, -0.0566117, 0.0876408, -0.1378767, 0.1314218
4: -0.0342595, 0.1180273, -0.0306006, 0.1174072, -0.1516666, 0.1486279

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0709314, upper bound: 0.0722356
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718959, upper bound: 0.0741778
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720762, upper bound: 0.0745388
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467
time: 0.43 seconds

## Summary of splitting at layer (split count: 13)
- Time for IS candidates: 6.19 seconds
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0715084, upper bound: 0.0723445
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0710245, upper bound: 0.0722935
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B1, status: Status.VERIFIED, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0715084, upper bound: 0.0723445
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0710245, upper bound: 0.0722935
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0710245, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0709314, upper bound: 0.0722356
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0718959, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0720762, upper bound: 0.0745388
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 14, time: 6.19
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0012393, 0.0595270, -0.0649236, 0.0627191
1: -0.0295746, 0.0838206, -0.0242316, 0.0825824, -0.1121571, 0.1080522
2: -0.0105933, 0.1197037, -0.0079983, 0.1184319, -0.1290252, 0.1277020
3: -0.0481883, 0.0739372, -0.0566117, 0.0876408, -0.1358292, 0.1305490
4: -0.0338624, 0.1156039, -0.0306006, 0.1174072, -0.1512696, 0.1462045

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0053373, 0.0603769, -0.0657735, 0.0668171
1: -0.0295746, 0.0838206, -0.0293680, 0.0834937, -0.1130683, 0.1131886
2: -0.0105933, 0.1197037, -0.0105539, 0.1178839, -0.1284771, 0.1302576
3: -0.0481883, 0.0739372, -0.0462398, 0.0737956, -0.1219839, 0.1201770
4: -0.0338624, 0.1156039, -0.0338468, 0.1125477, -0.1464101, 0.1494508

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0713339, upper bound: 0.0741086
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
time: 0.48 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0742207
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0009499, 0.0590112, -0.0644078, 0.0624297
1: -0.0295746, 0.0838206, -0.0228628, 0.0822512, -0.1118258, 0.1066834
2: -0.0105933, 0.1197037, -0.0077484, 0.1169360, -0.1275293, 0.1274521
3: -0.0481883, 0.0739372, -0.0552436, 0.0872260, -0.1354144, 0.1291808
4: -0.0338624, 0.1156039, -0.0301728, 0.1152276, -0.1490900, 0.1457767

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0053373, 0.0603769, -0.0657735, 0.0668171
1: -0.0295746, 0.0838206, -0.0293680, 0.0834937, -0.1130683, 0.1131886
2: -0.0105933, 0.1197037, -0.0105539, 0.1178839, -0.1284771, 0.1302576
3: -0.0481883, 0.0739372, -0.0462398, 0.0737956, -0.1219839, 0.1201770
4: -0.0338624, 0.1156039, -0.0338468, 0.1125477, -0.1464101, 0.1494508

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0707835, upper bound: 0.0723431
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0012393, 0.0595270, -0.0649236, 0.0627191
1: -0.0295746, 0.0838206, -0.0242316, 0.0825824, -0.1121571, 0.1080522
2: -0.0105933, 0.1197037, -0.0079983, 0.1184319, -0.1290252, 0.1277020
3: -0.0481883, 0.0739372, -0.0566117, 0.0876408, -0.1358292, 0.1305490
4: -0.0338624, 0.1156039, -0.0306006, 0.1174072, -0.1512696, 0.1462045

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0053373, 0.0603769, -0.0657735, 0.0668171
1: -0.0295746, 0.0838206, -0.0293680, 0.0834937, -0.1130683, 0.1131886
2: -0.0105933, 0.1197037, -0.0105539, 0.1178839, -0.1284771, 0.1302576
3: -0.0481883, 0.0739372, -0.0462398, 0.0737956, -0.1219839, 0.1201770
4: -0.0338624, 0.1156039, -0.0338468, 0.1125477, -0.1464101, 0.1494508

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0713339, upper bound: 0.0741086
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
time: 0.49 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0742207
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0017990, 0.0603538, -0.0657504, 0.0632788
1: -0.0295746, 0.0838206, -0.0261768, 0.0834607, -0.1130353, 0.1099974
2: -0.0105933, 0.1197037, -0.0086028, 0.1206991, -0.1312923, 0.1283065
3: -0.0481883, 0.0739372, -0.0593277, 0.0886848, -0.1368731, 0.1332649
4: -0.0338624, 0.1156039, -0.0311766, 0.1211701, -0.1550325, 0.1467805

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0058780, 0.0619953, -0.0673919, 0.0673578
1: -0.0295746, 0.0838206, -0.0304515, 0.0843779, -0.1139525, 0.1142721
2: -0.0105933, 0.1197037, -0.0112683, 0.1210694, -0.1316626, 0.1309720
3: -0.0481883, 0.0739372, -0.0502359, 0.0748101, -0.1229984, 0.1241731
4: -0.0338624, 0.1156039, -0.0342595, 0.1180273, -0.1518897, 0.1498634

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718942, upper bound: 0.0742207
time: 0.53 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0000952, 0.0585321, -0.0644101, 0.0620905
1: -0.0304515, 0.0843779, -0.0215799, 0.0819260, -0.1123775, 0.1059579
2: -0.0112683, 0.1210694, -0.0072518, 0.1159492, -0.1272175, 0.1283212
3: -0.0502359, 0.0748101, -0.0540064, 0.0868932, -0.1371291, 0.1288164
4: -0.0342595, 0.1180273, -0.0299066, 0.1133695, -0.1476290, 0.1479339

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0715084, upper bound: 0.0723445
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0030341, 0.0593020, -0.0651800, 0.0650294
1: -0.0304515, 0.0843779, -0.0222096, 0.0823854, -0.1128369, 0.1065876
2: -0.0112683, 0.1210694, -0.0092799, 0.1153993, -0.1266676, 0.1303493
3: -0.0502359, 0.0748101, -0.0432175, 0.0723896, -0.1226255, 0.1180276
4: -0.0342595, 0.1180273, -0.0328059, 0.1083915, -0.1426510, 0.1508332

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710245, upper bound: 0.0722935
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0710245, upper bound: 0.0741778
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, 0.0001039, 0.0580676, -0.0639456, 0.0618914
1: -0.0304515, 0.0843779, -0.0204136, 0.0815959, -0.1120474, 0.1047915
2: -0.0112683, 0.1210694, -0.0070285, 0.1146723, -0.1259406, 0.1280979
3: -0.0502359, 0.0748101, -0.0528774, 0.0865048, -0.1367407, 0.1276874
4: -0.0342595, 0.1180273, -0.0295022, 0.1115406, -0.1458001, 0.1475295

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0709314, upper bound: 0.0722356
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718959, upper bound: 0.0741778
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0030341, 0.0593020, -0.0651800, 0.0650294
1: -0.0304515, 0.0843779, -0.0222096, 0.0823854, -0.1128369, 0.1065876
2: -0.0112683, 0.1210694, -0.0092799, 0.1153993, -0.1266676, 0.1303493
3: -0.0502359, 0.0748101, -0.0432175, 0.0723896, -0.1226255, 0.1180276
4: -0.0342595, 0.1180273, -0.0328059, 0.1083915, -0.1426510, 0.1508332

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0000952, 0.0585321, -0.0644101, 0.0620905
1: -0.0304515, 0.0843779, -0.0215799, 0.0819260, -0.1123775, 0.1059579
2: -0.0112683, 0.1210694, -0.0072518, 0.1159492, -0.1272175, 0.1283212
3: -0.0502359, 0.0748101, -0.0540064, 0.0868932, -0.1371291, 0.1288164
4: -0.0342595, 0.1180273, -0.0299066, 0.1133695, -0.1476290, 0.1479339

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0709314, upper bound: 0.0722356
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718959, upper bound: 0.0741778
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0030341, 0.0593020, -0.0651800, 0.0650294
1: -0.0304515, 0.0843779, -0.0222096, 0.0823854, -0.1128369, 0.1065876
2: -0.0112683, 0.1210694, -0.0092799, 0.1153993, -0.1266676, 0.1303493
3: -0.0502359, 0.0748101, -0.0432175, 0.0723896, -0.1226255, 0.1180276
4: -0.0342595, 0.1180273, -0.0328059, 0.1083915, -0.1426510, 0.1508332

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0000952, 0.0585321, -0.0644101, 0.0620905
1: -0.0304515, 0.0843779, -0.0215799, 0.0819260, -0.1123775, 0.1059579
2: -0.0112683, 0.1210694, -0.0072518, 0.1159492, -0.1272175, 0.1283212
3: -0.0502359, 0.0748101, -0.0540064, 0.0868932, -0.1371291, 0.1288164
4: -0.0342595, 0.1180273, -0.0299066, 0.1133695, -0.1476290, 0.1479339

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.45 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711015, upper bound: 0.0745032
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716606, upper bound: 0.0747509
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467
time: 0.52 seconds

## Summary of splitting at layer (split count: 14)
- Time for IS candidates: 6.95 seconds
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0707835, upper bound: 0.0723431
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_B1, status: Status.VERIFIED, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0718942, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1_B1, status: Status.VERIFIED, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0715084, upper bound: 0.0723445
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0710245, upper bound: 0.0722935
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0710245, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0709314, upper bound: 0.0722356
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0718959, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B1, status: Status.VERIFIED, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0709314, upper bound: 0.0722356
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0718959, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0718082, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 15, time: 6.95
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0012393, 0.0595270, -0.0649236, 0.0627191
1: -0.0295746, 0.0838206, -0.0242316, 0.0825824, -0.1121571, 0.1080522
2: -0.0105933, 0.1197037, -0.0079983, 0.1184319, -0.1290252, 0.1277020
3: -0.0481883, 0.0739372, -0.0566117, 0.0876408, -0.1358292, 0.1305490
4: -0.0338624, 0.1156039, -0.0306006, 0.1174072, -0.1512696, 0.1462045

Time for backsubstitution: 2.12 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0709228, upper bound: 0.0723804
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0053373, 0.0603769, -0.0657735, 0.0668171
1: -0.0295746, 0.0838206, -0.0293680, 0.0834937, -0.1130683, 0.1131886
2: -0.0105933, 0.1197037, -0.0105539, 0.1178839, -0.1284771, 0.1302576
3: -0.0481883, 0.0739372, -0.0462398, 0.0737956, -0.1219839, 0.1201770
4: -0.0338624, 0.1156039, -0.0338468, 0.1125477, -0.1464101, 0.1494508

Time for backsubstitution: 2.11 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0009499, 0.0590112, -0.0644078, 0.0624297
1: -0.0295746, 0.0838206, -0.0228628, 0.0822512, -0.1118258, 0.1066834
2: -0.0105933, 0.1197037, -0.0077484, 0.1169360, -0.1275293, 0.1274521
3: -0.0481883, 0.0739372, -0.0552436, 0.0872260, -0.1354144, 0.1291808
4: -0.0338624, 0.1156039, -0.0301728, 0.1152276, -0.1490900, 0.1457767

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0709228, upper bound: 0.0723804
time: 0.44 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716248, upper bound: 0.0742207
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0053373, 0.0603769, -0.0657735, 0.0668171
1: -0.0295746, 0.0838206, -0.0293680, 0.0834937, -0.1130683, 0.1131886
2: -0.0105933, 0.1197037, -0.0105539, 0.1178839, -0.1284771, 0.1302576
3: -0.0481883, 0.0739372, -0.0462398, 0.0737956, -0.1219839, 0.1201770
4: -0.0338624, 0.1156039, -0.0338468, 0.1125477, -0.1464101, 0.1494508

Time for backsubstitution: 2.12 seconds
Binary search (step 2): status=Status.UNKNOWN, low=0.0574136, high=0.0781477, mid=0.0781477, abs_max=0.09181444346904755
rel_dist={0: [-0.07545650429510364, 0.07545650429510364]}

## Binary search (step 3) starts
Candidate diff: 0.0677807


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752712, upper bound: 0.0733837
time: 0.39 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0752213, upper bound: 0.0752213
time: 0.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.94 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.94
Output dim: 0, lower bound: -0.0752712, upper bound: 0.0733837
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.94
Output dim: 0, lower bound: -0.0752213, upper bound: 0.0752213

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0056712, 0.0494482, -0.0122710, 0.0795435, -0.0738723, 0.0617192
1: -0.0048309, 0.0792722, -0.0482301, 0.1046001, -0.1094310, 0.1275024
2: 0.0013121, 0.0852126, -0.0215323, 0.1538582, -0.1525461, 0.1067449
3: -0.0302970, 0.0797933, -0.0898915, 0.1120336, -0.1423306, 0.1696847
4: -0.0295759, 0.0721076, -0.0514701, 0.1717802, -0.2013561, 0.1235777

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0738182, upper bound: 0.0720345
time: 0.33 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750632, upper bound: 0.0732156
time: 0.40 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0080689, 0.0722086, -0.0122710, 0.0795435, -0.0876124, 0.0844796
1: -0.0359785, 0.0976419, -0.0482301, 0.1046001, -0.1405786, 0.1458721
2: -0.0166659, 0.1399382, -0.0215323, 0.1538582, -0.1705241, 0.1614705
3: -0.0768474, 0.1041032, -0.0898915, 0.1120336, -0.1888810, 0.1939946
4: -0.0455294, 0.1494691, -0.0514701, 0.1717802, -0.2173096, 0.2009392

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0738182, upper bound: 0.0738948
time: 0.35 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750632, upper bound: 0.0750632
time: 0.33 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.76 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.76
Output dim: 0, lower bound: -0.0738182, upper bound: 0.0720345
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 0, lower bound: -0.0750632, upper bound: 0.0732156
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 2.76
Output dim: 0, lower bound: -0.0738182, upper bound: 0.0738948
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 0, lower bound: -0.0750632, upper bound: 0.0750632

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0056712, 0.0494482, -0.0080926, 0.0739747, -0.0683034, 0.0575409
1: -0.0048309, 0.0792722, -0.0402663, 0.0988584, -0.1036893, 0.1195385
2: 0.0013121, 0.0852126, -0.0162963, 0.1430283, -0.1417162, 0.1015089
3: -0.0302970, 0.0797933, -0.0793115, 0.1036161, -0.1339131, 0.1591048
4: -0.0295759, 0.0721076, -0.0462731, 0.1545721, -0.1841480, 0.1183806

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732188, upper bound: 0.0732156
time: 0.31 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732188, upper bound: 0.0732156
time: 0.31 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0080689, 0.0722086, -0.0080926, 0.0739747, -0.0820436, 0.0803013
1: -0.0359785, 0.0976419, -0.0402663, 0.0988584, -0.1348368, 0.1379082
2: -0.0166659, 0.1399382, -0.0162963, 0.1430283, -0.1596942, 0.1562345
3: -0.0768474, 0.1041032, -0.0793115, 0.1036161, -0.1804635, 0.1834147
4: -0.0455294, 0.1494691, -0.0462731, 0.1545721, -0.2001015, 0.1957422

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0738948, upper bound: 0.0738182
time: 0.31 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0738948, upper bound: 0.0750632
time: 0.29 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.63 seconds
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 2.63
Output dim: 0, lower bound: -0.0732188, upper bound: 0.0732156
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 2.63
Output dim: 0, lower bound: -0.0732188, upper bound: 0.0732156
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.63
Output dim: 0, lower bound: -0.0738948, upper bound: 0.0738182
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -0.0738948, upper bound: 0.0750632

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0080926, 0.0739747, -0.0789240, 0.0765457
1: -0.0301443, 0.0939154, -0.0402663, 0.0988584, -0.1290027, 0.1341817
2: -0.0130560, 0.1317879, -0.0162963, 0.1430283, -0.1560843, 0.1480842
3: -0.0685425, 0.0983886, -0.0793115, 0.1036161, -0.1721586, 0.1777001
4: -0.0412280, 0.1366114, -0.0462731, 0.1545721, -0.1958002, 0.1828845

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0729613
time: 0.30 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0729613
time: 0.32 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.66 seconds
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0729613
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0729613
Binary search (step 3): status=Status.VERIFIED, low=0.0677807, high=0.0781477, mid=0.0677807, abs_max=0.09181444346904755
rel_dist={0: [-0.07527119418640024, 0.07527119418640014]}

## Binary search (step 4) starts
Candidate diff: 0.0729642


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753754, upper bound: 0.0734306
time: 0.38 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753110, upper bound: 0.0753110
time: 0.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 0.94 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 0.94
Output dim: 0, lower bound: -0.0753754, upper bound: 0.0734306
IS_A2, status: Status.UNKNOWN, split count: 1, time: 0.94
Output dim: 0, lower bound: -0.0753110, upper bound: 0.0753110

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0056712, 0.0494482, -0.0122710, 0.0795435, -0.0738723, 0.0617192
1: -0.0048309, 0.0792722, -0.0482301, 0.1046001, -0.1094310, 0.1275024
2: 0.0013121, 0.0852126, -0.0215323, 0.1538582, -0.1525461, 0.1067449
3: -0.0302970, 0.0797933, -0.0898915, 0.1120336, -0.1423306, 0.1696847
4: -0.0295759, 0.0721076, -0.0514701, 0.1717802, -0.2013561, 0.1235777

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0739271, upper bound: 0.0720345
time: 0.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751354, upper bound: 0.0732194
time: 0.35 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0080689, 0.0722086, -0.0122710, 0.0795435, -0.0876124, 0.0844796
1: -0.0359785, 0.0976419, -0.0482301, 0.1046001, -0.1405786, 0.1458721
2: -0.0166659, 0.1399382, -0.0215323, 0.1538582, -0.1705241, 0.1614705
3: -0.0768474, 0.1041032, -0.0898915, 0.1120336, -0.1888810, 0.1939946
4: -0.0455294, 0.1494691, -0.0514701, 0.1717802, -0.2173096, 0.2009392

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0739131, upper bound: 0.0739228
time: 0.33 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751354, upper bound: 0.0751354
time: 0.35 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.75 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.75
Output dim: 0, lower bound: -0.0739271, upper bound: 0.0720345
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.75
Output dim: 0, lower bound: -0.0751354, upper bound: 0.0732194
IS_A2_B1, status: Status.VERIFIED, split count: 2, time: 2.75
Output dim: 0, lower bound: -0.0739131, upper bound: 0.0739228
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.75
Output dim: 0, lower bound: -0.0751354, upper bound: 0.0751354

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0056712, 0.0494482, -0.0080926, 0.0739747, -0.0683034, 0.0575409
1: -0.0048309, 0.0792722, -0.0402663, 0.0988584, -0.1036893, 0.1195385
2: 0.0013121, 0.0852126, -0.0162963, 0.1430283, -0.1417162, 0.1015089
3: -0.0302970, 0.0797933, -0.0793115, 0.1036161, -0.1339131, 0.1591048
4: -0.0295759, 0.0721076, -0.0462731, 0.1545721, -0.1841480, 0.1183806

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732194, upper bound: 0.0732194
time: 0.30 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732194, upper bound: 0.0732194
time: 0.32 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0080689, 0.0722086, -0.0080926, 0.0739747, -0.0820436, 0.0803013
1: -0.0359785, 0.0976419, -0.0402663, 0.0988584, -0.1348368, 0.1379082
2: -0.0166659, 0.1399382, -0.0162963, 0.1430283, -0.1596942, 0.1562345
3: -0.0768474, 0.1041032, -0.0793115, 0.1036161, -0.1804635, 0.1834147
4: -0.0455294, 0.1494691, -0.0462731, 0.1545721, -0.2001015, 0.1957422

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0739228, upper bound: 0.0739131
time: 0.30 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0739228, upper bound: 0.0739131
time: 0.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.61 seconds
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0732194, upper bound: 0.0732194
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0732194, upper bound: 0.0732194
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0739228, upper bound: 0.0739131
IS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.61
Output dim: 0, lower bound: -0.0739228, upper bound: 0.0739131
Binary search (step 4): status=Status.VERIFIED, low=0.0729642, high=0.0781477, mid=0.0729642, abs_max=0.09181444346904755
rel_dist={0: [-0.0753754230366303, 0.07537542303663025]}

## Binary search (step 5) starts
Candidate diff: 0.0755560


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754233, upper bound: 0.0734540
time: 0.48 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0753554, upper bound: 0.0753554
time: 0.36 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.04 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.04
Output dim: 0, lower bound: -0.0754233, upper bound: 0.0734540
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.04
Output dim: 0, lower bound: -0.0753554, upper bound: 0.0753554

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0056712, 0.0494482, -0.0122710, 0.0795435, -0.0738723, 0.0617192
1: -0.0048309, 0.0792722, -0.0482301, 0.1046001, -0.1094310, 0.1275024
2: 0.0013121, 0.0852126, -0.0215323, 0.1538582, -0.1525461, 0.1067449
3: -0.0302970, 0.0797933, -0.0898915, 0.1120336, -0.1423306, 0.1696847
4: -0.0295759, 0.0721076, -0.0514701, 0.1717802, -0.2013561, 0.1235777

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0739799, upper bound: 0.0720345
time: 0.36 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751610, upper bound: 0.0732195
time: 0.37 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0080689, 0.0722086, -0.0122710, 0.0795435, -0.0876124, 0.0844796
1: -0.0359785, 0.0976419, -0.0482301, 0.1046001, -0.1405786, 0.1458721
2: -0.0166659, 0.1399382, -0.0215323, 0.1538582, -0.1705241, 0.1614705
3: -0.0768474, 0.1041032, -0.0898915, 0.1120336, -0.1888810, 0.1939946
4: -0.0455294, 0.1494691, -0.0514701, 0.1717802, -0.2173096, 0.2009392

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0739367, upper bound: 0.0739409
time: 0.35 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751610, upper bound: 0.0751610
time: 0.38 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.79 seconds
IS_A1_B1, status: Status.VERIFIED, split count: 2, time: 2.79
Output dim: 0, lower bound: -0.0739799, upper bound: 0.0720345
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -0.0751610, upper bound: 0.0732195
IS_A2_A1, status: Status.VERIFIED, split count: 2, time: 2.79
Output dim: 0, lower bound: -0.0739367, upper bound: 0.0739409
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -0.0751610, upper bound: 0.0751610

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0056712, 0.0494482, -0.0080926, 0.0739747, -0.0683034, 0.0575409
1: -0.0048309, 0.0792722, -0.0402663, 0.0988584, -0.1036893, 0.1195385
2: 0.0013121, 0.0852126, -0.0162963, 0.1430283, -0.1417162, 0.1015089
3: -0.0302970, 0.0797933, -0.0793115, 0.1036161, -0.1339131, 0.1591048
4: -0.0295759, 0.0721076, -0.0462731, 0.1545721, -0.1841480, 0.1183806

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732195, upper bound: 0.0732195
time: 0.36 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732195, upper bound: 0.0732195
time: 0.38 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0122710, 0.0795435, -0.0844928, 0.0807240
1: -0.0301443, 0.0939154, -0.0482301, 0.1046001, -0.1347444, 0.1421455
2: -0.0130560, 0.1317879, -0.0215323, 0.1538582, -0.1669142, 0.1533202
3: -0.0685425, 0.0983886, -0.0898915, 0.1120336, -0.1805761, 0.1882800
4: -0.0412280, 0.1366114, -0.0514701, 0.1717802, -0.2130082, 0.1880815

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0739409, upper bound: 0.0739367
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0739409, upper bound: 0.0751610
time: 0.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.81 seconds
IS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 0, lower bound: -0.0732195, upper bound: 0.0732195
IS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 0, lower bound: -0.0732195, upper bound: 0.0732195
IS_A2_A2_B1, status: Status.VERIFIED, split count: 3, time: 2.81
Output dim: 0, lower bound: -0.0739409, upper bound: 0.0739367
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.81
Output dim: 0, lower bound: -0.0739409, upper bound: 0.0751610

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0080926, 0.0739747, -0.0789240, 0.0765457
1: -0.0301443, 0.0939154, -0.0402663, 0.0988584, -0.1290027, 0.1341817
2: -0.0130560, 0.1317879, -0.0162963, 0.1430283, -0.1560843, 0.1480842
3: -0.0685425, 0.0983886, -0.0793115, 0.1036161, -0.1721586, 0.1777001
4: -0.0412280, 0.1366114, -0.0462731, 0.1545721, -0.1958002, 0.1828845

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730574, upper bound: 0.0751610
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0730574, upper bound: 0.0751610
time: 0.31 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.73 seconds
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0730574, upper bound: 0.0751610
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.73
Output dim: 0, lower bound: -0.0730574, upper bound: 0.0751610

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, 0.0083338, 0.0470844, -0.0520337, 0.0601192
1: -0.0301443, 0.0939154, -0.0002709, 0.0768870, -0.1070312, 0.0941863
2: -0.0130560, 0.1317879, 0.0037108, 0.0803992, -0.0934552, 0.1280771
3: -0.0685425, 0.0983886, -0.0257773, 0.0755912, -0.1441337, 0.1241658
4: -0.0412280, 0.1366114, -0.0258763, 0.0645177, -0.1057457, 0.1624877

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 1.13 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0707840, upper bound: 0.0724643
time: 0.31 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731042, upper bound: 0.0750455
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0049493, 0.0684531, -0.0049493, 0.0684531, -0.0734024, 0.0734024
1: -0.0301443, 0.0939154, -0.0301443, 0.0939154, -0.1240597, 0.1240597
2: -0.0130560, 0.1317879, -0.0130560, 0.1317879, -0.1448440, 0.1448440
3: -0.0685425, 0.0983886, -0.0685425, 0.0983886, -0.1669310, 0.1669310
4: -0.0412280, 0.1366114, -0.0412280, 0.1366114, -0.1778394, 0.1778394

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of IS_A2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 1.11 seconds

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0725221
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467
time: 0.34 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.70 seconds
IS_A2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.70
Output dim: 0, lower bound: -0.0707840, upper bound: 0.0724643
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -0.0731042, upper bound: 0.0750455
IS_A2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.70
Output dim: 0, lower bound: -0.0711612, upper bound: 0.0725221
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.70
Output dim: 0, lower bound: -0.0716703, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0083338, 0.0470844, -0.0510514, 0.0590321
1: -0.0276524, 0.0931255, -0.0002709, 0.0768870, -0.1045393, 0.0933965
2: -0.0123972, 0.1293659, 0.0037108, 0.0803992, -0.0927964, 0.1256551
3: -0.0660703, 0.0977019, -0.0257773, 0.0755912, -0.1416615, 0.1234792
4: -0.0406218, 0.1327123, -0.0258763, 0.0645177, -0.1051394, 0.1585886

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719548, upper bound: 0.0747092
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0715436, upper bound: 0.0747313
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0047890, 0.0682258, -0.0741038, 0.0667843
1: -0.0304515, 0.0843779, -0.0300045, 0.0935407, -0.1239922, 0.1143824
2: -0.0112683, 0.1210694, -0.0128695, 0.1314652, -0.1427336, 0.1339388
3: -0.0502359, 0.0748101, -0.0681428, 0.0973495, -0.1475854, 0.1429528
4: -0.0342595, 0.1180273, -0.0409980, 0.1361903, -0.1704498, 0.1590253

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720656, upper bound: 0.0745388
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720656, upper bound: 0.0748467
time: 0.39 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.35 seconds
IS_A2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0719548, upper bound: 0.0747092
IS_A2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0715436, upper bound: 0.0747313
IS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0720656, upper bound: 0.0745388
IS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.35
Output dim: 0, lower bound: -0.0720656, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0097754, 0.0448622, -0.0488292, 0.0575906
1: -0.0276524, 0.0931255, 0.0012435, 0.0714976, -0.0991500, 0.0918821
2: -0.0123972, 0.1293659, 0.0055359, 0.0778081, -0.0902053, 0.1238300
3: -0.0660703, 0.0977019, -0.0235429, 0.0696618, -0.1357321, 0.1212448
4: -0.0406218, 0.1327123, -0.0193165, 0.0613490, -0.1019708, 0.1520288

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0715806, upper bound: 0.0717987
time: 0.33 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.33 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718772, upper bound: 0.0746514
time: 0.34 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0037871, 0.0671609, 0.0078467, 0.0443021, -0.0480891, 0.0593142
1: -0.0274996, 0.0927568, -0.0047341, 0.0745015, -0.1020010, 0.0974909
2: -0.0122096, 0.1290821, 0.0048839, 0.0745222, -0.0867317, 0.1241981
3: -0.0656836, 0.0966451, -0.0191891, 0.0687331, -0.1344168, 0.1158341
4: -0.0403940, 0.1323290, -0.0225027, 0.0567436, -0.0971376, 0.1548316

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711634, upper bound: 0.0746760
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0714749, upper bound: 0.0746703
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0021452, 0.0612259, -0.0671039, 0.0641406
1: -0.0304515, 0.0843779, -0.0263871, 0.0842858, -0.1147373, 0.1107650
2: -0.0112683, 0.1210694, -0.0090670, 0.1218928, -0.1331611, 0.1301364
3: -0.0502359, 0.0748101, -0.0604222, 0.0902384, -0.1404743, 0.1352323
4: -0.0342595, 0.1180273, -0.0321517, 0.1232046, -0.1574641, 0.1501790

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711015, upper bound: 0.0745032
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743948
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0714981, upper bound: 0.0747509
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467
time: 0.42 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 7.02 seconds
IS_A2_A2_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 7.02
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 7.02
Output dim: 0, lower bound: -0.0718772, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 7.02
Output dim: 0, lower bound: -0.0711634, upper bound: 0.0746760
IS_A2_A2_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 7.02
Output dim: 0, lower bound: -0.0714749, upper bound: 0.0746703
IS_A2_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 7.02
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743948
IS_A2_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 7.02
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 7.02
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 7.02
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0119214, 0.0440228, -0.0479898, 0.0554446
1: -0.0276524, 0.0931255, 0.0005230, 0.0695521, -0.0972045, 0.0926026
2: -0.0123972, 0.1293659, 0.0096441, 0.0773689, -0.0897661, 0.1197219
3: -0.0660703, 0.0977019, -0.0194770, 0.0624470, -0.1285173, 0.1171789
4: -0.0406218, 0.1327123, -0.0188315, 0.0573395, -0.0979612, 0.1515437

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0100352, 0.0447484, -0.0487154, 0.0573308
1: -0.0276524, 0.0931255, 0.0015434, 0.0713148, -0.0989672, 0.0915821
2: -0.0123972, 0.1293659, 0.0057138, 0.0774657, -0.0898629, 0.1236522
3: -0.0660703, 0.0977019, -0.0229581, 0.0693600, -0.1354303, 0.1206600
4: -0.0406218, 0.1327123, -0.0192196, 0.0607713, -0.1013930, 0.1519319

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718772, upper bound: 0.0746514
time: 0.37 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0089256, 0.0436134, -0.0475805, 0.0584403
1: -0.0276524, 0.0931255, -0.0056828, 0.0719095, -0.0995619, 0.0988083
2: -0.0123972, 0.1293659, 0.0094991, 0.0732964, -0.0856936, 0.1198668
3: -0.0660703, 0.0977019, -0.0168171, 0.0634604, -0.1295307, 0.1145190
4: -0.0406218, 0.1327123, -0.0218024, 0.0532283, -0.0938501, 0.1545147

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720612
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720612
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0080052, 0.0442183, -0.0481854, 0.0593608
1: -0.0276524, 0.0931255, -0.0045802, 0.0743399, -0.1019923, 0.0977057
2: -0.0123972, 0.1293659, 0.0050475, 0.0742267, -0.0866239, 0.1243185
3: -0.0660703, 0.0977019, -0.0188669, 0.0684462, -0.1345165, 0.1165688
4: -0.0406218, 0.1327123, -0.0224565, 0.0562747, -0.0968964, 0.1551688

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0726690
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0726690
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0024844, 0.0581586, -0.0021452, 0.0612259, -0.0637102, 0.0603038
1: -0.0264296, 0.0815245, -0.0263871, 0.0842858, -0.1107154, 0.1079116
2: -0.0051643, 0.1094249, -0.0090670, 0.1218928, -0.1270570, 0.1184920
3: -0.0423236, 0.0763173, -0.0604222, 0.0902384, -0.1325620, 0.1367395
4: -0.0359877, 0.1030843, -0.0321517, 0.1232046, -0.1591924, 0.1352361

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743949
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743949
time: 0.33 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0021452, 0.0612259, -0.0666225, 0.0636251
1: -0.0295746, 0.0838206, -0.0263871, 0.0842858, -0.1138604, 0.1102077
2: -0.0105933, 0.1197037, -0.0090670, 0.1218928, -0.1324860, 0.1287707
3: -0.0481883, 0.0739372, -0.0604222, 0.0902384, -0.1384267, 0.1343594
4: -0.0338624, 0.1156039, -0.0321517, 0.1232046, -0.1570670, 0.1477557

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0719482, upper bound: 0.0742207
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720656, upper bound: 0.0745388
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467
time: 0.42 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 6.82 seconds
IS_A2_A2_B2_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0718772, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720612
IS_A2_A2_B2_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720612
IS_A2_A2_B2_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0726690
IS_A2_A2_B2_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0726690
IS_A2_A2_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743949
IS_A2_A2_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0712228, upper bound: 0.0743949
IS_A2_A2_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0719482, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0720656, upper bound: 0.0745388
IS_A2_A2_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.82
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0119214, 0.0440228, -0.0479898, 0.0554446
1: -0.0276524, 0.0931255, 0.0005230, 0.0695521, -0.0972045, 0.0926026
2: -0.0123972, 0.1293659, 0.0096441, 0.0773689, -0.0897661, 0.1197219
3: -0.0660703, 0.0977019, -0.0194770, 0.0624470, -0.1285173, 0.1171789
4: -0.0406218, 0.1327123, -0.0188315, 0.0573395, -0.0979612, 0.1515437

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0037871, 0.0671609, 0.0089256, 0.0436134, -0.0474005, 0.0582353
1: -0.0274996, 0.0927568, -0.0056828, 0.0719095, -0.0994091, 0.0984396
2: -0.0122096, 0.1290821, 0.0094991, 0.0732964, -0.0855059, 0.1195830
3: -0.0656836, 0.0966451, -0.0168171, 0.0634604, -0.1291440, 0.1134621
4: -0.0403940, 0.1323290, -0.0218024, 0.0532283, -0.0936223, 0.1541314

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720612
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0741902
time: 0.35 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0100352, 0.0447484, -0.0487154, 0.0573308
1: -0.0276524, 0.0931255, 0.0015434, 0.0713148, -0.0989672, 0.0915821
2: -0.0123972, 0.1293659, 0.0057138, 0.0774657, -0.0898629, 0.1236522
3: -0.0660703, 0.0977019, -0.0229581, 0.0693600, -0.1354303, 0.1206600
4: -0.0406218, 0.1327123, -0.0192196, 0.0607713, -0.1013930, 0.1519319

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0715106, upper bound: 0.0717987
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.34 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718772, upper bound: 0.0746514
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0037871, 0.0671609, 0.0080052, 0.0442183, -0.0480054, 0.0591557
1: -0.0274996, 0.0927568, -0.0045802, 0.0743399, -0.1018395, 0.0973370
2: -0.0122096, 0.1290821, 0.0050475, 0.0742267, -0.0864362, 0.1240346
3: -0.0656836, 0.0966451, -0.0188669, 0.0684462, -0.1341299, 0.1155120
4: -0.0403940, 0.1323290, -0.0224565, 0.0562747, -0.0966687, 0.1547855

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0024844, 0.0581586, -0.0021452, 0.0612259, -0.0637102, 0.0603038
1: -0.0264296, 0.0815245, -0.0263871, 0.0842858, -0.1107154, 0.1079116
2: -0.0051643, 0.1094249, -0.0090670, 0.1218928, -0.1270570, 0.1184920
3: -0.0423236, 0.0763173, -0.0604222, 0.0902384, -0.1325620, 0.1367395
4: -0.0359877, 0.1030843, -0.0321517, 0.1232046, -0.1591924, 0.1352361

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0708850, upper bound: 0.0721222
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0708850, upper bound: 0.0721222
time: 0.42 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0024844, 0.0581586, -0.0058780, 0.0619953, -0.0644797, 0.0640366
1: -0.0264296, 0.0815245, -0.0304515, 0.0843779, -0.1108075, 0.1119760
2: -0.0051643, 0.1094249, -0.0112683, 0.1210694, -0.1262336, 0.1206933
3: -0.0423236, 0.0763173, -0.0502359, 0.0748101, -0.1171336, 0.1265531
4: -0.0359877, 0.1030843, -0.0342595, 0.1180273, -0.1540150, 0.1373438

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711798, upper bound: 0.0726242
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711798, upper bound: 0.0743949
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0021452, 0.0612259, -0.0666225, 0.0636251
1: -0.0295746, 0.0838206, -0.0263871, 0.0842858, -0.1138604, 0.1102077
2: -0.0105933, 0.1197037, -0.0090670, 0.1218928, -0.1324860, 0.1287707
3: -0.0481883, 0.0739372, -0.0604222, 0.0902384, -0.1384267, 0.1343594
4: -0.0338624, 0.1156039, -0.0321517, 0.1232046, -0.1570670, 0.1477557

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716088, upper bound: 0.0724500
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716088, upper bound: 0.0724658
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0058780, 0.0619953, -0.0673919, 0.0673578
1: -0.0295746, 0.0838206, -0.0304515, 0.0843779, -0.1139525, 0.1142721
2: -0.0105933, 0.1197037, -0.0112683, 0.1210694, -0.1316626, 0.1309720
3: -0.0481883, 0.0739372, -0.0502359, 0.0748101, -0.1229984, 0.1241731
4: -0.0338624, 0.1156039, -0.0342595, 0.1180273, -0.1518897, 0.1498634

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711713, upper bound: 0.0741086
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0742207
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0017990, 0.0603538, -0.0662317, 0.0637943
1: -0.0304515, 0.0843779, -0.0261768, 0.0834607, -0.1139122, 0.1105547
2: -0.0112683, 0.1210694, -0.0086028, 0.1206991, -0.1319674, 0.1296722
3: -0.0502359, 0.0748101, -0.0593277, 0.0886848, -0.1389207, 0.1341377
4: -0.0342595, 0.1180273, -0.0311766, 0.1211701, -0.1554296, 0.1492039

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 31

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0711015, upper bound: 0.0745032
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0716088, upper bound: 0.0724500
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0714981, upper bound: 0.0747509
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467
time: 0.43 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 7.25 seconds
IS_A2_A2_B2_B1_A2_B1_B1_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
IS_A2_A2_B2_B1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720612
IS_A2_A2_B2_B1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0718772, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
IS_A2_A2_B2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0708850, upper bound: 0.0721222
IS_A2_A2_B2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0708850, upper bound: 0.0721222
IS_A2_A2_B2_B2_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0711798, upper bound: 0.0726242
IS_A2_A2_B2_B2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0711798, upper bound: 0.0743949
IS_A2_A2_B2_B2_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0716088, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B1_A2_B1_B2, status: Status.VERIFIED, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0716088, upper bound: 0.0724658
IS_A2_A2_B2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0723431
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0711221, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0716088, upper bound: 0.0724500
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0712648, upper bound: 0.0725221
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 7.25
Output dim: 0, lower bound: -0.0731696, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0036633, 0.0672161, 0.0119214, 0.0440228, -0.0476861, 0.0552947
1: -0.0272224, 0.0929019, 0.0005230, 0.0695521, -0.0967745, 0.0923789
2: -0.0120373, 0.1289738, 0.0096441, 0.0773689, -0.0894062, 0.1193298
3: -0.0653339, 0.0973650, -0.0194770, 0.0624470, -0.1277809, 0.1168420
4: -0.0405064, 0.1320211, -0.0188315, 0.0573395, -0.0978459, 0.1508526

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040268, 0.0608585, 0.0089256, 0.0436134, -0.0476403, 0.0519329
1: -0.0240528, 0.0833467, -0.0056828, 0.0719095, -0.0959624, 0.0890295
2: -0.0102135, 0.1185798, 0.0094991, 0.0732964, -0.0835099, 0.1090807
3: -0.0473803, 0.0734831, -0.0168171, 0.0634604, -0.1108407, 0.0903002
4: -0.0333277, 0.1139941, -0.0218024, 0.0532283, -0.0865560, 0.1357965

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0119214, 0.0440228, -0.0479898, 0.0554446
1: -0.0276524, 0.0931255, 0.0005230, 0.0695521, -0.0972045, 0.0926026
2: -0.0123972, 0.1293659, 0.0096441, 0.0773689, -0.0897661, 0.1197219
3: -0.0660703, 0.0977019, -0.0194770, 0.0624470, -0.1285173, 0.1171789
4: -0.0406218, 0.1327123, -0.0188315, 0.0573395, -0.0979612, 0.1515437

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0100352, 0.0447484, -0.0487154, 0.0573308
1: -0.0276524, 0.0931255, 0.0015434, 0.0713148, -0.0989672, 0.0915821
2: -0.0123972, 0.1293659, 0.0057138, 0.0774657, -0.0898629, 0.1236522
3: -0.0660703, 0.0977019, -0.0229581, 0.0693600, -0.1354303, 0.1206600
4: -0.0406218, 0.1327123, -0.0192196, 0.0607713, -0.1013930, 0.1519319

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718772, upper bound: 0.0746514
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
time: 0.45 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0089256, 0.0436134, -0.0475805, 0.0584403
1: -0.0276524, 0.0931255, -0.0056828, 0.0719095, -0.0995619, 0.0988083
2: -0.0123972, 0.1293659, 0.0094991, 0.0732964, -0.0856936, 0.1198668
3: -0.0660703, 0.0977019, -0.0168171, 0.0634604, -0.1295307, 0.1145190
4: -0.0406218, 0.1327123, -0.0218024, 0.0532283, -0.0938501, 0.1545147

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720612
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0741902
time: 0.40 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0080052, 0.0442183, -0.0481854, 0.0593608
1: -0.0276524, 0.0931255, -0.0045802, 0.0743399, -0.1019923, 0.0977057
2: -0.0123972, 0.1293659, 0.0050475, 0.0742267, -0.0866239, 0.1243185
3: -0.0660703, 0.0977019, -0.0188669, 0.0684462, -0.1345165, 0.1165688
4: -0.0406218, 0.1327123, -0.0224565, 0.0562747, -0.0968964, 0.1551688

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0726690
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0726690
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0024844, 0.0581586, -0.0053966, 0.0614798, -0.0639642, 0.0635552
1: -0.0264296, 0.0815245, -0.0295746, 0.0838206, -0.1102502, 0.1110991
2: -0.0051643, 0.1094249, -0.0105933, 0.1197037, -0.1248679, 0.1200182
3: -0.0423236, 0.0763173, -0.0481883, 0.0739372, -0.1162608, 0.1245056
4: -0.0359877, 0.1030843, -0.0338624, 0.1156039, -0.1515917, 0.1369467

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0721222
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0721222
time: 0.44 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0053966, 0.0614798, -0.0058780, 0.0619953, -0.0673919, 0.0673578
1: -0.0295746, 0.0838206, -0.0304515, 0.0843779, -0.1139525, 0.1142721
2: -0.0105933, 0.1197037, -0.0112683, 0.1210694, -0.1316626, 0.1309720
3: -0.0481883, 0.0739372, -0.0502359, 0.0748101, -0.1229984, 0.1241731
4: -0.0338624, 0.1156039, -0.0342595, 0.1180273, -0.1518897, 0.1498634

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
time: 0.38 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718942, upper bound: 0.0742207
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0012393, 0.0595270, -0.0654050, 0.0632346
1: -0.0304515, 0.0843779, -0.0242316, 0.0825824, -0.1130340, 0.1086095
2: -0.0112683, 0.1210694, -0.0079983, 0.1184319, -0.1297002, 0.1290677
3: -0.0502359, 0.0748101, -0.0566117, 0.0876408, -0.1378767, 0.1314218
4: -0.0342595, 0.1180273, -0.0306006, 0.1174072, -0.1516666, 0.1486279

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: B, layer: 3, pos: 7
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 31
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0708868, upper bound: 0.0722356
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0741778
time: 0.39 seconds

## BFS IS instance: IS_A2_A2_B2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058780, 0.0619953, -0.0058780, 0.0619953, -0.0678733, 0.0678733
1: -0.0304515, 0.0843779, -0.0304515, 0.0843779, -0.1148294, 0.1148294
2: -0.0112683, 0.1210694, -0.0112683, 0.1210694, -0.1323377, 0.1323377
3: -0.0502359, 0.0748101, -0.0502359, 0.0748101, -0.1250459, 0.1250459
4: -0.0342595, 0.1180273, -0.0342595, 0.1180273, -0.1522867, 0.1522867

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: B, layer: 3, pos: 4
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 7
type: B, layer: 3, pos: 15
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 31
type: B, layer: 3, pos: 31
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720656, upper bound: 0.0745388
time: 0.42 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467
time: 0.42 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 7.06 seconds
IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0718772, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A1, status: Status.VERIFIED, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720612
IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0741902
IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0726690
IS_A2_A2_B2_B1_A2_B1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0710092, upper bound: 0.0726690
IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0721222
IS_A2_A2_B2_B2_A2_B1_A1_B2_B2_A2, status: Status.VERIFIED, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0707800, upper bound: 0.0721222
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0718943, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0718942, upper bound: 0.0742207
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B1, status: Status.VERIFIED, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0708868, upper bound: 0.0722356
IS_A2_A2_B2_B2_A2_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0718730, upper bound: 0.0741778
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0720656, upper bound: 0.0745388
IS_A2_A2_B2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 10, time: 7.06
Output dim: 0, lower bound: -0.0731695, upper bound: 0.0748467

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0036633, 0.0672161, 0.0119214, 0.0440228, -0.0476861, 0.0552947
1: -0.0272224, 0.0929019, 0.0005230, 0.0695521, -0.0967745, 0.0923789
2: -0.0120373, 0.1289738, 0.0096441, 0.0773689, -0.0894062, 0.1193298
3: -0.0653339, 0.0973650, -0.0194770, 0.0624470, -0.1277809, 0.1168420
4: -0.0405064, 0.1320211, -0.0188315, 0.0573395, -0.0978459, 0.1508526

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0034833, 0.0670127, 0.0089256, 0.0436134, -0.0470967, 0.0580871
1: -0.0270731, 0.0925333, -0.0056828, 0.0719095, -0.0989826, 0.0982161
2: -0.0118496, 0.1286928, 0.0094991, 0.0732964, -0.0851460, 0.1191937
3: -0.0649512, 0.0963074, -0.0168171, 0.0634604, -0.1284115, 0.1131245
4: -0.0402786, 0.1316438, -0.0218024, 0.0532283, -0.0935069, 0.1534463

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720612
time: 0.40 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0741902
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040268, 0.0608585, 0.0130052, 0.0433223, -0.0473491, 0.0478533
1: -0.0240528, 0.0833467, 0.0019701, 0.0680764, -0.0921292, 0.0813766
2: -0.0102135, 0.1185798, 0.0118987, 0.0761565, -0.0863701, 0.1066811
3: -0.0473803, 0.0734831, -0.0178691, 0.0586825, -0.1060628, 0.0913523
4: -0.0333277, 0.1139941, -0.0175245, 0.0554511, -0.0887788, 0.1315185

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: A, layer: 3, pos: 49
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
time: 0.41 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.43 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040268, 0.0608585, 0.0089256, 0.0436134, -0.0476403, 0.0519329
1: -0.0240528, 0.0833467, -0.0056828, 0.0719095, -0.0959624, 0.0890295
2: -0.0102135, 0.1185798, 0.0094991, 0.0732964, -0.0835099, 0.1090807
3: -0.0473803, 0.0734831, -0.0168171, 0.0634604, -0.1108407, 0.0903002
4: -0.0333277, 0.1139941, -0.0218024, 0.0532283, -0.0865560, 0.1357965

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720612
time: 0.43 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.41 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0119214, 0.0440228, -0.0479898, 0.0554446
1: -0.0276524, 0.0931255, 0.0005230, 0.0695521, -0.0972045, 0.0926026
2: -0.0123972, 0.1293659, 0.0096441, 0.0773689, -0.0897661, 0.1197219
3: -0.0660703, 0.0977019, -0.0194770, 0.0624470, -0.1285173, 0.1171789
4: -0.0406218, 0.1327123, -0.0188315, 0.0573395, -0.0979612, 0.1515437

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 15

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 48

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0723358
time: 0.35 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0037871, 0.0671609, 0.0089256, 0.0436134, -0.0474005, 0.0582353
1: -0.0274996, 0.0927568, -0.0056828, 0.0719095, -0.0994091, 0.0984396
2: -0.0122096, 0.1290821, 0.0094991, 0.0732964, -0.0855059, 0.1195830
3: -0.0656836, 0.0966451, -0.0168171, 0.0634604, -0.1291440, 0.1134621
4: -0.0403940, 0.1323290, -0.0218024, 0.0532283, -0.0936223, 0.1541314

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 7

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0720612
time: 0.39 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0701838, upper bound: 0.0741902
time: 0.37 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0039670, 0.0673659, 0.0100352, 0.0447484, -0.0487154, 0.0573308
1: -0.0276524, 0.0931255, 0.0015434, 0.0713148, -0.0989672, 0.0915821
2: -0.0123972, 0.1293659, 0.0057138, 0.0774657, -0.0898629, 0.1236522
3: -0.0660703, 0.0977019, -0.0229581, 0.0693600, -0.1354303, 0.1206600
4: -0.0406218, 0.1327123, -0.0192196, 0.0607713, -0.1013930, 0.1519319

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 39
type: B, layer: 3, pos: 48
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 15
type: B, layer: 3, pos: 30
type: A, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0715106, upper bound: 0.0717987
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0718772, upper bound: 0.0746514
time: 0.36 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0037871, 0.0671609, 0.0080052, 0.0442183, -0.0480054, 0.0591557
1: -0.0274996, 0.0927568, -0.0045802, 0.0743399, -0.1018395, 0.0973370
2: -0.0122096, 0.1290821, 0.0050475, 0.0742267, -0.0864362, 0.1240346
3: -0.0656836, 0.0966451, -0.0188669, 0.0684462, -0.1341299, 0.1155120
4: -0.0403940, 0.1323290, -0.0224565, 0.0562747, -0.0966687, 0.1547855

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 26
type: A, layer: 3, pos: 4
type: A, layer: 3, pos: 30
type: B, layer: 3, pos: 26
type: A, layer: 3, pos: 49
type: A, layer: 3, pos: 8
type: B, layer: 3, pos: 49
type: B, layer: 3, pos: 30
type: B, layer: 3, pos: 39
type: A, layer: 3, pos: 7
type: B, layer: 3, pos: 48
type: B, layer: 3, pos: 8
type: A, layer: 3, pos: 48
type: A, layer: 3, pos: 15
type: A, layer: 3, pos: 39
type: B, layer: 3, pos: 4
type: A, layer: 3, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 4

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 26

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 3, pos: 8

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 30

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 39

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0704325, upper bound: 0.0741902
time: 0.36 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B1_A2_B1_B2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0714595, upper bound: 0.0746514
time: 0.38 seconds

## BFS IS instance: IS_A2_A2_B2_B1_A2_B1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040268, 0.0608585, 0.0089256, 0.0436134, -0.0476403, 0.0519329
1: -0.0240528, 0.0833467, -0.0056828, 0.0719095, -0.0959624, 0.0890295
2: -0.0102135, 0.1185798, 0.0094991, 0.0732964, -0.0835099, 0.1090807
3: -0.0473803, 0.0734831, -0.0168171, 0.0634604, -0.1108407, 0.0903002
4: -0.0333277, 0.1139941, -0.0218024, 0.0532283, -0.0865560, 0.1357965

Time for backsubstitution: 2.05 seconds
Binary search (step 5): status=Status.UNKNOWN, low=0.0729642, high=0.0755560, mid=0.0755560, abs_max=0.09181444346904755
rel_dist={0: [-0.07542327007268206, 0.07542327007268206]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.07296419361126993
execution time: 1148.16 seconds
