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
execution time: IAR + RelationalAnalysis = 0.65 + 0.76 = 1.42 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0754965, upper bound: 0.0754965

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0750535, upper bound: 0.0745368
time: 0.19 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0751743
time: 0.17 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 0.43 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 0.43
Output dim: 0, lower bound: -0.0750535, upper bound: 0.0745368
NS_A2, status: Status.UNKNOWN, split count: 1, time: 0.43
Output dim: 0, lower bound: -0.0751743, upper bound: 0.0751743

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0096398, 0.0840788, -0.0122710, 0.0795435, -0.0891833, 0.0963498
1: -0.0483434, 0.1096082, -0.0482301, 0.1046001, -0.1529435, 0.1578384
2: -0.0175393, 0.1624446, -0.0215323, 0.1538582, -0.1713975, 0.1839769
3: -0.1008122, 0.1153759, -0.0898915, 0.1120336, -0.2128458, 0.2052674
4: -0.0483946, 0.1864471, -0.0514701, 0.1717802, -0.2201748, 0.2379172

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0743580, upper bound: 0.0743580
time: 0.18 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0743580, upper bound: 0.0745368
time: 0.20 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0080926, 0.0739747, -0.0122710, 0.0795435, -0.0876361, 0.0862456
1: -0.0402663, 0.0988584, -0.0482301, 0.1046001, -0.1448664, 0.1470885
2: -0.0162963, 0.1430283, -0.0215323, 0.1538582, -0.1701545, 0.1645606
3: -0.0793115, 0.1036161, -0.0898915, 0.1120336, -0.1913451, 0.1935076
4: -0.0462731, 0.1545721, -0.0514701, 0.1717802, -0.2180533, 0.2060422

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0745368, upper bound: 0.0749940
time: 0.19 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0745368, upper bound: 0.0751743
time: 0.18 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 1.00 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 1.00
Output dim: 0, lower bound: -0.0743580, upper bound: 0.0743580
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 1.00
Output dim: 0, lower bound: -0.0743580, upper bound: 0.0745368
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 1.00
Output dim: 0, lower bound: -0.0745368, upper bound: 0.0749940
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 1.00
Output dim: 0, lower bound: -0.0745368, upper bound: 0.0751743

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0096398, 0.0840788, -0.0096398, 0.0840788, -0.0937186, 0.0937186
1: -0.0483434, 0.1096082, -0.0483434, 0.1096082, -0.1579516, 0.1579516
2: -0.0175393, 0.1624446, -0.0175393, 0.1624446, -0.1799838, 0.1799838
3: -0.1008122, 0.1153759, -0.1008122, 0.1153759, -0.2161881, 0.2161881
4: -0.0483946, 0.1864471, -0.0483946, 0.1864471, -0.2348417, 0.2348417

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741097, upper bound: 0.0725347
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0729613
time: 0.18 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0096398, 0.0840788, -0.0080926, 0.0739747, -0.0836145, 0.0921715
1: -0.0483434, 0.1096082, -0.0402663, 0.0988584, -0.1472017, 0.1498745
2: -0.0175393, 0.1624446, -0.0162963, 0.1430283, -0.1605676, 0.1787409
3: -0.1008122, 0.1153759, -0.0793115, 0.1036161, -0.2044283, 0.1946874
4: -0.0483946, 0.1864471, -0.0462731, 0.1545721, -0.2029667, 0.2327202

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741097, upper bound: 0.0732908
time: 0.19 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0741465
time: 0.20 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0080926, 0.0739747, -0.0096398, 0.0840788, -0.0921715, 0.0836145
1: -0.0402663, 0.0988584, -0.0483434, 0.1096082, -0.1498745, 0.1472017
2: -0.0162963, 0.1430283, -0.0175393, 0.1624446, -0.1787409, 0.1605676
3: -0.0793115, 0.1036161, -0.1008122, 0.1153759, -0.1946874, 0.2044283
4: -0.0462731, 0.1545721, -0.0483946, 0.1864471, -0.2327202, 0.2029667

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0720345
time: 0.19 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0739891
time: 0.20 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0080926, 0.0739747, -0.0080926, 0.0739747, -0.0820673, 0.0820673
1: -0.0402663, 0.0988584, -0.0402663, 0.0988584, -0.1391247, 0.1391247
2: -0.0162963, 0.1430283, -0.0162963, 0.1430283, -0.1593246, 0.1593246
3: -0.0793115, 0.1036161, -0.0793115, 0.1036161, -0.1829276, 0.1829276
4: -0.0462731, 0.1545721, -0.0462731, 0.1545721, -0.2008452, 0.2008452

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0725892
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0751743
time: 0.20 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 1.05 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.05
Output dim: 0, lower bound: -0.0741097, upper bound: 0.0725347
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 1.05
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0729613
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.05
Output dim: 0, lower bound: -0.0741097, upper bound: 0.0732908
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.05
Output dim: 0, lower bound: -0.0729613, upper bound: 0.0741465
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 1.05
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0720345
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 1.05
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0739891
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 1.05
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0725892
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 1.05
Output dim: 0, lower bound: -0.0741465, upper bound: 0.0751743

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.0031931, 0.0596317, -0.0096398, 0.0840788, -0.0808857, 0.0692715
1: -0.0159592, 0.0935836, -0.0483434, 0.1096082, -0.1255674, 0.1419269
2: -0.0004775, 0.1070566, -0.0175393, 0.1624446, -0.1629221, 0.1245959
3: -0.0561146, 0.0980010, -0.1008122, 0.1153759, -0.1714906, 0.1988132
4: -0.0290360, 0.1065627, -0.0483946, 0.1864471, -0.2154831, 0.1549573

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725347, upper bound: 0.0725347
time: 0.20 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0725347, upper bound: 0.0725347
time: 0.19 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.0031931, 0.0596317, -0.0080926, 0.0739747, -0.0707816, 0.0677244
1: -0.0159592, 0.0935836, -0.0402663, 0.0988584, -0.1148175, 0.1338498
2: -0.0004775, 0.1070566, -0.0162963, 0.1430283, -0.1435058, 0.1233529
3: -0.0561146, 0.0980010, -0.0793115, 0.1036161, -0.1597307, 0.1773125
4: -0.0290360, 0.1065627, -0.0462731, 0.1545721, -0.1836081, 0.1528358

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0732907
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0732907
time: 0.21 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0056816, 0.0768258, -0.0080926, 0.0739747, -0.0796563, 0.0849185
1: -0.0367180, 0.1016865, -0.0402663, 0.0988584, -0.1355763, 0.1419528
2: -0.0131781, 0.1492988, -0.0162963, 0.1430283, -0.1562063, 0.1655951
3: -0.0882968, 0.1038739, -0.0793115, 0.1036161, -0.1919129, 0.1831854
4: -0.0414514, 0.1655897, -0.0462731, 0.1545721, -0.1960235, 0.2118628

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0741465
time: 0.20 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0741465
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.0083338, 0.0470844, -0.0096398, 0.0840788, -0.0757450, 0.0567242
1: -0.0002709, 0.0768870, -0.0483434, 0.1096082, -0.1098791, 0.1252303
2: 0.0037108, 0.0803992, -0.0175393, 0.1624446, -0.1587338, 0.0979384
3: -0.0257773, 0.0755912, -0.1008122, 0.1153759, -0.1411532, 0.1764034
4: -0.0258763, 0.0645177, -0.0483946, 0.1864471, -0.2123234, 0.1129123

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732907, upper bound: 0.0720345
time: 0.21 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732907, upper bound: 0.0720345
time: 0.20 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0049493, 0.0684531, -0.0096398, 0.0840788, -0.0890281, 0.0780929
1: -0.0301443, 0.0939154, -0.0483434, 0.1096082, -0.1397525, 0.1422587
2: -0.0130560, 0.1317879, -0.0175393, 0.1624446, -0.1755006, 0.1493272
3: -0.0685425, 0.0983886, -0.1008122, 0.1153759, -0.1839184, 0.1992007
4: -0.0412280, 0.1366114, -0.0483946, 0.1864471, -0.2276751, 0.1850060

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732907, upper bound: 0.0739891
time: 0.20 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732907, upper bound: 0.0739891
time: 0.21 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.0083338, 0.0470844, -0.0080926, 0.0739747, -0.0656408, 0.0551770
1: -0.0002709, 0.0768870, -0.0402663, 0.0988584, -0.0991293, 0.1171533
2: 0.0037108, 0.0803992, -0.0162963, 0.1430283, -0.1393175, 0.0966955
3: -0.0257773, 0.0755912, -0.0793115, 0.1036161, -0.1293934, 0.1549027
4: -0.0258763, 0.0645177, -0.0462731, 0.1545721, -0.1804484, 0.1107908

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0725892
time: 0.20 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0725892
time: 0.20 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0049493, 0.0684531, -0.0080926, 0.0739747, -0.0789240, 0.0765457
1: -0.0301443, 0.0939154, -0.0402663, 0.0988584, -0.1290027, 0.1341817
2: -0.0130560, 0.1317879, -0.0162963, 0.1430283, -0.1560843, 0.1480842
3: -0.0685425, 0.0983886, -0.0793115, 0.1036161, -0.1721586, 0.1777001
4: -0.0412280, 0.1366114, -0.0462731, 0.1545721, -0.1958002, 0.1828845

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0751743
time: 0.21 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0751743
time: 0.21 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 1.07 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.07
Output dim: 0, lower bound: -0.0725347, upper bound: 0.0725347
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.07
Output dim: 0, lower bound: -0.0725347, upper bound: 0.0725347
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.07
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0732907
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.07
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0732907
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.07
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0741465
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.07
Output dim: 0, lower bound: -0.0720345, upper bound: 0.0741465
NS_A2_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.07
Output dim: 0, lower bound: -0.0732907, upper bound: 0.0720345
NS_A2_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.07
Output dim: 0, lower bound: -0.0732907, upper bound: 0.0720345
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.07
Output dim: 0, lower bound: -0.0732907, upper bound: 0.0739891
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.07
Output dim: 0, lower bound: -0.0732907, upper bound: 0.0739891
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 1.07
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0725892
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 1.07
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0725892
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 1.07
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0751743
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 1.07
Output dim: 0, lower bound: -0.0732197, upper bound: 0.0751743

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0056816, 0.0768258, 0.0083338, 0.0470844, -0.0527660, 0.0684920
1: -0.0367180, 0.1016865, -0.0002709, 0.0768870, -0.1136049, 0.1019575
2: -0.0131781, 0.1492988, 0.0037108, 0.0803992, -0.0935772, 0.1455880
3: -0.0882968, 0.1038739, -0.0257773, 0.0755912, -0.1638880, 0.1296511
4: -0.0414514, 0.1655897, -0.0258763, 0.0645177, -0.1059691, 0.1914660

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0056816, 0.0768258, -0.0049493, 0.0684531, -0.0741346, 0.0817752
1: -0.0367180, 0.1016865, -0.0301443, 0.0939154, -0.1306333, 0.1318308
2: -0.0131781, 0.1492988, -0.0130560, 0.1317879, -0.1449660, 0.1623548
3: -0.0882968, 0.1038739, -0.0685425, 0.0983886, -0.1866854, 0.1724163
4: -0.0414514, 0.1655897, -0.0412280, 0.1366114, -0.1780628, 0.2068177

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0049493, 0.0684531, 0.0031931, 0.0596317, -0.0645811, 0.0652600
1: -0.0301443, 0.0939154, -0.0159592, 0.0935836, -0.1237278, 0.1098745
2: -0.0130560, 0.1317879, -0.0004775, 0.1070566, -0.1201126, 0.1322655
3: -0.0685425, 0.0983886, -0.0561146, 0.0980010, -0.1665435, 0.1545032
4: -0.0412280, 0.1366114, -0.0290360, 0.1065627, -0.1477907, 0.1656474

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0049493, 0.0684531, -0.0056816, 0.0768258, -0.0817752, 0.0741346
1: -0.0301443, 0.0939154, -0.0367180, 0.1016865, -0.1318308, 0.1306333
2: -0.0130560, 0.1317879, -0.0131781, 0.1492988, -0.1623548, 0.1449660
3: -0.0685425, 0.0983886, -0.0882968, 0.1038739, -0.1724163, 0.1866854
4: -0.0412280, 0.1366114, -0.0414514, 0.1655897, -0.2068177, 0.1780628

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0049493, 0.0684531, 0.0083338, 0.0470844, -0.0520337, 0.0601192
1: -0.0301443, 0.0939154, -0.0002709, 0.0768870, -0.1070312, 0.0941863
2: -0.0130560, 0.1317879, 0.0037108, 0.0803992, -0.0934552, 0.1280771
3: -0.0685425, 0.0983886, -0.0257773, 0.0755912, -0.1441337, 0.1241658
4: -0.0412280, 0.1366114, -0.0258763, 0.0645177, -0.1057457, 0.1624877

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0049493, 0.0684531, -0.0049493, 0.0684531, -0.0734024, 0.0734024
1: -0.0301443, 0.0939154, -0.0301443, 0.0939154, -0.1240597, 0.1240597
2: -0.0130560, 0.1317879, -0.0130560, 0.1317879, -0.1448440, 0.1448440
3: -0.0685425, 0.0983886, -0.0685425, 0.0983886, -0.1669310, 0.1669310
4: -0.0412280, 0.1366114, -0.0412280, 0.1366114, -0.1778394, 0.1778394

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23

Time for candidate selection: 0.04 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 1.42 + 18.76 = 20.17 seconds
