## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 3)
Time budget: 420 seconds
Split limit: 100
Threshold: 0.07398656999999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0122710, 0.0795435, -0.0122710, 0.0795435, -0.0918144, 0.0918144)
1: (-0.0482301, 0.1046001, -0.0482301, 0.1046001, -0.1528303, 0.1528303)
2: (-0.0215323, 0.1538582, -0.0215323, 0.1538582, -0.1753905, 0.1753905)
3: (-0.0898915, 0.1120336, -0.0898915, 0.1120336, -0.2019251, 0.2019251)
4: (-0.0514701, 0.1717802, -0.0514701, 0.1717802, -0.2232503, 0.2232503)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.76 + 0.78 = 1.54 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0754965, upper bound: 0.0754965

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754965, upper bound: 0.0736748
time: 0.19 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0754287, upper bound: 0.0754287
time: 0.21 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.48 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.48
Output dim: 0, lower bound: -0.0754965, upper bound: 0.0736748
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.48
Output dim: 0, lower bound: -0.0754287, upper bound: 0.0754287

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.0056712, 0.0494482, -0.0122710, 0.0795435, -0.0738723, 0.0617192
1: -0.0048309, 0.0792722, -0.0482301, 0.1046001, -0.1094310, 0.1275024
2: 0.0013121, 0.0852126, -0.0215323, 0.1538582, -0.1525461, 0.1067449
3: -0.0302970, 0.0797933, -0.0898915, 0.1120336, -0.1423306, 0.1696847
4: -0.0295759, 0.0721076, -0.0514701, 0.1717802, -0.2013561, 0.1235777

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0720345
time: 0.19 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0732197
time: 0.19 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0080689, 0.0722086, -0.0122710, 0.0795435, -0.0876124, 0.0844796
1: -0.0359785, 0.0976419, -0.0482301, 0.1046001, -0.1405786, 0.1458721
2: -0.0166659, 0.1399382, -0.0215323, 0.1538582, -0.1705241, 0.1614705
3: -0.0768474, 0.1041032, -0.0898915, 0.1120336, -0.1888810, 0.1939946
4: -0.0455294, 0.1494691, -0.0514701, 0.1717802, -0.2173096, 0.2009392

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0739891, upper bound: 0.0741465
time: 0.20 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0751743
time: 0.22 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.21 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.21
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0720345
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.21
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0732197
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 1.21
Output dim: 0, lower bound: -0.0739891, upper bound: 0.0741465
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 1.21
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0751743

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.0056712, 0.0494482, -0.0096398, 0.0840788, -0.0784076, 0.0590880
1: -0.0048309, 0.0792722, -0.0483434, 0.1096082, -0.1144391, 0.1276156
2: 0.0013121, 0.0852126, -0.0175393, 0.1624446, -0.1611325, 0.1027519
3: -0.0302970, 0.0797933, -0.1008122, 0.1153759, -0.1456730, 0.1806055
4: -0.0295759, 0.0721076, -0.0483946, 0.1864471, -0.2160230, 0.1205022

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0736276, upper bound: 0.0720345
time: 0.21 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0736276, upper bound: 0.0720345
time: 0.21 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.0056712, 0.0494482, -0.0080926, 0.0739747, -0.0683034, 0.0575409
1: -0.0048309, 0.0792722, -0.0402663, 0.0988584, -0.1036893, 0.1195385
2: 0.0013121, 0.0852126, -0.0162963, 0.1430283, -0.1417162, 0.1015089
3: -0.0302970, 0.0797933, -0.0793115, 0.1036161, -0.1339131, 0.1591048
4: -0.0295759, 0.0721076, -0.0462731, 0.1545721, -0.1841480, 0.1183806

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0732197
time: 0.19 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0732197
time: 0.22 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.0055032, 0.0746784, -0.0122710, 0.0795435, -0.0850467, 0.0869494
1: -0.0363166, 0.1003191, -0.0482301, 0.1046001, -0.1409167, 0.1485492
2: -0.0130474, 0.1460492, -0.0215323, 0.1538582, -0.1669056, 0.1675815
3: -0.0863938, 0.1024000, -0.0898915, 0.1120336, -0.1984274, 0.1922915
4: -0.0407893, 0.1603383, -0.0514701, 0.1717802, -0.2125695, 0.2118084

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0729613
time: 0.20 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0741465
time: 0.21 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.0049493, 0.0684531, -0.0122710, 0.0795435, -0.0844928, 0.0807240
1: -0.0301443, 0.0939154, -0.0482301, 0.1046001, -0.1347444, 0.1421455
2: -0.0130560, 0.1317879, -0.0215323, 0.1538582, -0.1669142, 0.1533202
3: -0.0685425, 0.0983886, -0.0898915, 0.1120336, -0.1805761, 0.1882800
4: -0.0412280, 0.1366114, -0.0514701, 0.1717802, -0.2130082, 0.1880815

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0739891
time: 0.21 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0751743
time: 0.20 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.25 seconds
NS_A1_B1_B1, status: Status.VERIFIED, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0736276, upper bound: 0.0720345
NS_A1_B1_B2, status: Status.VERIFIED, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0736276, upper bound: 0.0720345
NS_A1_B2_B1, status: Status.VERIFIED, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0732197
NS_A1_B2_B2, status: Status.VERIFIED, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0732197
NS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0729613
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0741465
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0739891
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 1.25
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0751743

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0055032, 0.0746784, -0.0080926, 0.0739747, -0.0794779, 0.0827711
1: -0.0363166, 0.1003191, -0.0402663, 0.0988584, -0.1351749, 0.1405854
2: -0.0130474, 0.1460492, -0.0162963, 0.1430283, -0.1560757, 0.1623455
3: -0.0863938, 0.1024000, -0.0793115, 0.1036161, -0.1900099, 0.1817115
4: -0.0407893, 0.1603383, -0.0462731, 0.1545721, -0.1953614, 0.2066113

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0741465
time: 0.21 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0741465
time: 0.22 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0049493, 0.0684531, -0.0096398, 0.0840788, -0.0890281, 0.0780929
1: -0.0301443, 0.0939154, -0.0483434, 0.1096082, -0.1397525, 0.1422587
2: -0.0130560, 0.1317879, -0.0175393, 0.1624446, -0.1755006, 0.1493272
3: -0.0685425, 0.0983886, -0.1008122, 0.1153759, -0.1839184, 0.1992007
4: -0.0412280, 0.1366114, -0.0483946, 0.1864471, -0.2276751, 0.1850060

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0739891
time: 0.24 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0739891
time: 0.24 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0049493, 0.0684531, -0.0080926, 0.0739747, -0.0789240, 0.0765457
1: -0.0301443, 0.0939154, -0.0402663, 0.0988584, -0.1290027, 0.1341817
2: -0.0130560, 0.1317879, -0.0162963, 0.1430283, -0.1560843, 0.1480842
3: -0.0685425, 0.0983886, -0.0793115, 0.1036161, -0.1721586, 0.1777001
4: -0.0412280, 0.1366114, -0.0462731, 0.1545721, -0.1958002, 0.1828845

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0751743
time: 0.24 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0751743
time: 0.23 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.39 seconds
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0741465
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0720338, upper bound: 0.0741465
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0739891
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0739891
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0751743
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 1.39
Output dim: 0, lower bound: -0.0732190, upper bound: 0.0751743

## BFS NS instance: NS_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0055032, 0.0746784, 0.0083338, 0.0470844, -0.0525876, 0.0663446
1: -0.0363166, 0.1003191, -0.0002709, 0.0768870, -0.1132035, 0.1005900
2: -0.0130474, 0.1460492, 0.0037108, 0.0803992, -0.0934466, 0.1423384
3: -0.0863938, 0.1024000, -0.0257773, 0.0755912, -0.1619850, 0.1281773
4: -0.0407893, 0.1603383, -0.0258763, 0.0645177, -0.1053070, 0.1862146

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0055032, 0.0746784, -0.0049493, 0.0684531, -0.0739563, 0.0796278
1: -0.0363166, 0.1003191, -0.0301443, 0.0939154, -0.1302319, 0.1304634
2: -0.0130474, 0.1460492, -0.0130560, 0.1317879, -0.1448353, 0.1591052
3: -0.0863938, 0.1024000, -0.0685425, 0.0983886, -0.1847824, 0.1709425
4: -0.0407893, 0.1603383, -0.0412280, 0.1366114, -0.1774007, 0.2015663

Time for backsubstitution: 0.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0049493, 0.0684531, 0.0031931, 0.0596317, -0.0645811, 0.0652600
1: -0.0301443, 0.0939154, -0.0159592, 0.0935836, -0.1237278, 0.1098745
2: -0.0130560, 0.1317879, -0.0004775, 0.1070566, -0.1201126, 0.1322655
3: -0.0685425, 0.0983886, -0.0561146, 0.0980010, -0.1665435, 0.1545032
4: -0.0412280, 0.1366114, -0.0290360, 0.1065627, -0.1477907, 0.1656474

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0049493, 0.0684531, -0.0056816, 0.0768258, -0.0817752, 0.0741346
1: -0.0301443, 0.0939154, -0.0367180, 0.1016865, -0.1318308, 0.1306333
2: -0.0130560, 0.1317879, -0.0131781, 0.1492988, -0.1623548, 0.1449660
3: -0.0685425, 0.0983886, -0.0882968, 0.1038739, -0.1724163, 0.1866854
4: -0.0412280, 0.1366114, -0.0414514, 0.1655897, -0.2068177, 0.1780628

Time for backsubstitution: 0.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0049493, 0.0684531, 0.0083338, 0.0470844, -0.0520337, 0.0601192
1: -0.0301443, 0.0939154, -0.0002709, 0.0768870, -0.1070312, 0.0941863
2: -0.0130560, 0.1317879, 0.0037108, 0.0803992, -0.0934552, 0.1280771
3: -0.0685425, 0.0983886, -0.0257773, 0.0755912, -0.1441337, 0.1241658
4: -0.0412280, 0.1366114, -0.0258763, 0.0645177, -0.1057457, 0.1624877

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## BFS NS instance: NS_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0049493, 0.0684531, -0.0049493, 0.0684531, -0.0734024, 0.0734024
1: -0.0301443, 0.0939154, -0.0301443, 0.0939154, -0.1240597, 0.1240597
2: -0.0130560, 0.1317879, -0.0130560, 0.1317879, -0.1448440, 0.1448440
3: -0.0685425, 0.0983886, -0.0685425, 0.0983886, -0.1669310, 0.1669310
4: -0.0412280, 0.1366114, -0.0412280, 0.1366114, -0.1778394, 0.1778394

Time for backsubstitution: 0.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.54 + 18.44 = 19.98 seconds
