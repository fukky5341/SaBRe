## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.38366993


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1352843, 0.1904531, -0.1352843, 0.1904531, -0.3257374, 0.3257374)
1: (-0.1201238, 0.1458962, -0.1201238, 0.1458962, -0.2660201, 0.2660201)
2: (-0.1303060, 0.1809286, -0.1303060, 0.1809286, -0.3112346, 0.3112346)
3: (-0.0989191, 0.1781699, -0.0989191, 0.1781699, -0.2770890, 0.2770890)
4: (-0.1623547, 0.1264126, -0.1623547, 0.1264126, -0.2887673, 0.2887673)
5: (-0.1397409, 0.1723848, -0.1397409, 0.1723848, -0.3121257, 0.3121257)
6: (-0.1049636, 0.2513187, -0.1049636, 0.2513187, -0.3562823, 0.3562823)
7: (-0.1651493, 0.1279102, -0.1651493, 0.1279102, -0.2930595, 0.2930595)
8: (0.5992520, 1.0262526, 0.5992520, 1.0262526, -0.4270006, 0.4270006)
9: (-0.1538356, 0.2383174, -0.1538356, 0.2383174, -0.3921530, 0.3921530)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.33 + 2.29 = 4.61 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.4081595, upper bound: 0.4081595

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 95
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 95

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4047620, upper bound: 0.4017925
time: 1.36 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4059220, upper bound: 0.4059220
time: 1.34 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.90 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 2.90
Output dim: 8, lower bound: -0.4047620, upper bound: 0.4017925
NS_B2, status: Status.UNKNOWN, split count: 1, time: 2.90
Output dim: 8, lower bound: -0.4059220, upper bound: 0.4059220

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -0.1208623, 0.1705671, -0.0539117, 0.0692396, -0.1901019, 0.2244788
1: -0.1071586, 0.1307383, -0.0419700, 0.0566659, -0.1638245, 0.1727082
2: -0.1174486, 0.1628826, -0.0512311, 0.0771920, -0.1946406, 0.2141138
3: -0.0877390, 0.1608631, -0.0303618, 0.0702400, -0.1579790, 0.1912250
4: -0.1473328, 0.1134694, -0.0781566, 0.0487180, -0.1960509, 0.1916260
5: -0.1247785, 0.1553309, -0.0503886, 0.0689669, -0.1937454, 0.2057196
6: -0.0937923, 0.2279741, -0.0418049, 0.1074286, -0.2012210, 0.2697789
7: -0.1508080, 0.1131584, -0.0816469, 0.0433215, -0.1941295, 0.1948053
8: 0.6306500, 1.0211157, 0.7946312, 1.0009336, -0.3702836, 0.2264844
9: -0.1360904, 0.2207701, -0.0525978, 0.1301623, -0.2662527, 0.2733679

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3582930, upper bound: 0.3771946
time: 1.50 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4043269, upper bound: 0.4011344
time: 1.49 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -0.1352046, 0.1903414, -0.0971362, 0.1345490, -0.2697536, 0.2874776
1: -0.1200527, 0.1458111, -0.0843641, 0.1045219, -0.2245746, 0.2301752
2: -0.1302338, 0.1808281, -0.0940011, 0.1323620, -0.2625958, 0.2748292
3: -0.0988565, 0.1780736, -0.0673682, 0.1297065, -0.2285630, 0.2454418
4: -0.1622697, 0.1263404, -0.1221205, 0.0901128, -0.2523826, 0.2484609
5: -0.1396561, 0.1722905, -0.0969386, 0.1253327, -0.2649889, 0.2692291
6: -0.1049015, 0.2511800, -0.0752351, 0.1814835, -0.2863850, 0.3264151
7: -0.1650701, 0.1278289, -0.1258591, 0.0881808, -0.2532510, 0.2536880
8: 0.5994357, 1.0262225, 0.6922069, 1.0129178, -0.4134821, 0.3340156
9: -0.1537349, 0.2382191, -0.1054325, 0.1886091, -0.3423440, 0.3436516

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4017925, upper bound: 0.4047620
time: 1.39 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4017925, upper bound: 0.4059220
time: 1.57 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.34 seconds
NS_B1_A1, status: Status.VERIFIED, split count: 2, time: 5.34
Output dim: 8, lower bound: -0.3582930, upper bound: 0.3771946
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 5.34
Output dim: 8, lower bound: -0.4043269, upper bound: 0.4011344
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 5.34
Output dim: 8, lower bound: -0.4017925, upper bound: 0.4047620
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 5.34
Output dim: 8, lower bound: -0.4017925, upper bound: 0.4059220

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -0.1129622, 0.1588319, -0.0539117, 0.0692396, -0.1822018, 0.2127436
1: -0.1000249, 0.1216619, -0.0419700, 0.0566659, -0.1566907, 0.1636318
2: -0.1095992, 0.1525539, -0.0512311, 0.0771920, -0.1867912, 0.2037850
3: -0.0814034, 0.1505057, -0.0303618, 0.0702400, -0.1516434, 0.1808675
4: -0.1379381, 0.1064756, -0.0781566, 0.0487180, -0.1866561, 0.1846323
5: -0.1162018, 0.1456227, -0.0503886, 0.0689669, -0.1851687, 0.1960113
6: -0.0878980, 0.2092112, -0.0418049, 0.1074286, -0.1953267, 0.2510161
7: -0.1428741, 0.1052969, -0.0816469, 0.0433215, -0.1861956, 0.1869438
8: 0.6547816, 1.0181485, 0.7946312, 1.0009336, -0.3461521, 0.2235173
9: -0.1248740, 0.2101882, -0.0525978, 0.1301623, -0.2550363, 0.2627859

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 95
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 95

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4009464, upper bound: 0.4009464
time: 1.23 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4009464, upper bound: 0.4011344
time: 1.08 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -0.0539117, 0.0692396, -0.0971362, 0.1345490, -0.1884607, 0.1663758
1: -0.0419700, 0.0566659, -0.0843641, 0.1045219, -0.1464919, 0.1410299
2: -0.0512311, 0.0771920, -0.0940011, 0.1323620, -0.1835932, 0.1711931
3: -0.0303618, 0.0702400, -0.0673682, 0.1297065, -0.1600683, 0.1376082
4: -0.0781566, 0.0487180, -0.1221205, 0.0901128, -0.1682695, 0.1708385
5: -0.0503886, 0.0689669, -0.0969386, 0.1253327, -0.1757214, 0.1659055
6: -0.0418049, 0.1074286, -0.0752351, 0.1814835, -0.2232884, 0.1826637
7: -0.0816469, 0.0433215, -0.1258591, 0.0881808, -0.1698277, 0.1691805
8: 0.7946312, 1.0009336, 0.6922069, 1.0129178, -0.2182865, 0.3087268
9: -0.0525978, 0.1301623, -0.1054325, 0.1886091, -0.2412069, 0.2355949

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3761647, upper bound: 0.3582929
time: 1.78 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4009464, upper bound: 0.4043268
time: 1.70 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -0.0971362, 0.1345490, -0.0971362, 0.1345490, -0.2316852, 0.2316852
1: -0.0843641, 0.1045219, -0.0843641, 0.1045219, -0.1888859, 0.1888859
2: -0.0940011, 0.1323620, -0.0940011, 0.1323620, -0.2263631, 0.2263631
3: -0.0673682, 0.1297065, -0.0673682, 0.1297065, -0.1970747, 0.1970747
4: -0.1221205, 0.0901128, -0.1221205, 0.0901128, -0.2122333, 0.2122333
5: -0.0969386, 0.1253327, -0.0969386, 0.1253327, -0.2222714, 0.2222714
6: -0.0752351, 0.1814835, -0.0752351, 0.1814835, -0.2567186, 0.2567186
7: -0.1258591, 0.0881808, -0.1258591, 0.0881808, -0.2140399, 0.2140399
8: 0.6922069, 1.0129178, 0.6922069, 1.0129178, -0.3207109, 0.3207109
9: -0.1054325, 0.1886091, -0.1054325, 0.1886091, -0.2940416, 0.2940416

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3507111, upper bound: 0.3845284
time: 1.47 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4009464, upper bound: 0.4055062
time: 1.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.31 seconds
NS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 8, lower bound: -0.4009464, upper bound: 0.4009464
NS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 8, lower bound: -0.4009464, upper bound: 0.4011344
NS_B2_A1_B1, status: Status.VERIFIED, split count: 3, time: 5.31
Output dim: 8, lower bound: -0.3761647, upper bound: 0.3582929
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 8, lower bound: -0.4009464, upper bound: 0.4043268
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 8, lower bound: -0.3507111, upper bound: 0.3845284
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 5.31
Output dim: 8, lower bound: -0.4009464, upper bound: 0.4055062

## BFS NS instance: NS_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0493161, 0.0592702, -0.0539117, 0.0692396, -0.1185558, 0.1131818
1: -0.0371491, 0.0498863, -0.0419700, 0.0566659, -0.0938150, 0.0918563
2: -0.0443956, 0.0706533, -0.0512311, 0.0771920, -0.1215875, 0.1218845
3: -0.0263210, 0.0598357, -0.0303618, 0.0702400, -0.0965610, 0.0901975
4: -0.0703086, 0.0439691, -0.0781566, 0.0487180, -0.1190267, 0.1221258
5: -0.0436823, 0.0613918, -0.0503886, 0.0689669, -0.1126492, 0.1117805
6: -0.0371571, 0.0918090, -0.0418049, 0.1074286, -0.1445858, 0.1336139
7: -0.0752574, 0.0386860, -0.0816469, 0.0433215, -0.1185789, 0.1203329
8: 0.8158978, 0.9991305, 0.7946312, 1.0009336, -0.1850358, 0.2044992
9: -0.0437454, 0.1203900, -0.0525978, 0.1301623, -0.1739077, 0.1729877

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3933460, upper bound: 0.3945046
time: 1.39 seconds

## Relational analysis of NS_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3932647, upper bound: 0.3932647
time: 1.36 seconds

## BFS NS instance: NS_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0895337, 0.1227166, -0.0539117, 0.0692396, -0.1587734, 0.1766283
1: -0.0772130, 0.0956098, -0.0419700, 0.0566659, -0.1338789, 0.1375798
2: -0.0860248, 0.1223815, -0.0512311, 0.0771920, -0.1632168, 0.1736126
3: -0.0609368, 0.1191868, -0.0303618, 0.0702400, -0.1311768, 0.1495486
4: -0.1131207, 0.0830842, -0.0781566, 0.0487180, -0.1618388, 0.1612408
5: -0.0881916, 0.1156569, -0.0503886, 0.0689669, -0.1571585, 0.1660456
6: -0.0695915, 0.1624770, -0.0418049, 0.1074286, -0.1770201, 0.2042819
7: -0.1184559, 0.0805019, -0.0816469, 0.0433215, -0.1617773, 0.1621487
8: 0.7166167, 1.0102448, 0.7946312, 1.0009336, -0.2843169, 0.2156136
9: -0.0945329, 0.1778910, -0.0525978, 0.1301623, -0.2246952, 0.2304887

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3933460, upper bound: 0.3945046
time: 1.26 seconds

## Relational analysis of NS_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3932647, upper bound: 0.3934491
time: 1.23 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0539117, 0.0692396, -0.0895337, 0.1227166, -0.1766283, 0.1587734
1: -0.0419700, 0.0566659, -0.0772130, 0.0956098, -0.1375798, 0.1338789
2: -0.0512311, 0.0771920, -0.0860248, 0.1223815, -0.1736126, 0.1632168
3: -0.0303618, 0.0702400, -0.0609368, 0.1191868, -0.1495486, 0.1311768
4: -0.0781566, 0.0487180, -0.1131207, 0.0830842, -0.1612408, 0.1618388
5: -0.0503886, 0.0689669, -0.0881916, 0.1156569, -0.1660456, 0.1571585
6: -0.0418049, 0.1074286, -0.0695915, 0.1624770, -0.2042819, 0.1770201
7: -0.0816469, 0.0433215, -0.1184559, 0.0805019, -0.1621487, 0.1617773
8: 0.7946312, 1.0009336, 0.7166167, 1.0102448, -0.2156136, 0.2843169
9: -0.0525978, 0.1301623, -0.0945329, 0.1778910, -0.2304887, 0.2246952

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3946879, upper bound: 0.3979711
time: 1.35 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3934492, upper bound: 0.3978609
time: 1.41 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0247446, 0.0238789, -0.0805600, 0.1084862, -0.1332308, 0.1044390
1: -0.0162383, 0.0122819, -0.0692241, 0.0845549, -0.1007932, 0.0815060
2: -0.0103939, 0.0349912, -0.0762060, 0.1101316, -0.1205256, 0.1111972
3: -0.0084901, 0.0165374, -0.0533526, 0.1068092, -0.1152993, 0.0698900
4: -0.0242227, 0.0155180, -0.1012748, 0.0752102, -0.0994329, 0.1167928
5: -0.0118294, 0.0332529, -0.0778004, 0.1041925, -0.1160220, 0.1110533
6: -0.0176889, 0.0211070, -0.0633715, 0.1343481, -0.1520370, 0.0844784
7: -0.0421650, 0.0091130, -0.1100898, 0.0717767, -0.1139417, 0.1192028
8: 0.9286714, 0.9932117, 0.7517720, 1.0076318, -0.0789604, 0.2414397
9: -0.0080481, 0.0536758, -0.0801916, 0.1644160, -0.1724641, 0.1338674

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 170

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3429368, upper bound: 0.3710784
time: 1.35 seconds

## Relational analysis of NS_B2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3427511, upper bound: 0.3696158
time: 1.32 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0895337, 0.1227166, -0.0971362, 0.1345490, -0.2240827, 0.2198528
1: -0.0772130, 0.0956098, -0.0843641, 0.1045219, -0.1817349, 0.1799739
2: -0.0860248, 0.1223815, -0.0940011, 0.1323620, -0.2183869, 0.2163826
3: -0.0609368, 0.1191868, -0.0673682, 0.1297065, -0.1906433, 0.1865550
4: -0.1131207, 0.0830842, -0.1221205, 0.0901128, -0.2032335, 0.2052047
5: -0.0881916, 0.1156569, -0.0969386, 0.1253327, -0.2135243, 0.2125956
6: -0.0695915, 0.1624770, -0.0752351, 0.1814835, -0.2510749, 0.2377122
7: -0.1184559, 0.0805019, -0.1258591, 0.0881808, -0.2066367, 0.2063609
8: 0.7166167, 1.0102448, 0.6922069, 1.0129178, -0.2963010, 0.3180380
9: -0.0945329, 0.1778910, -0.1054325, 0.1886091, -0.2831420, 0.2833235

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 242

## Relational analysis of NS_B2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3992466, upper bound: 0.4005849
time: 1.25 seconds

## Relational analysis of NS_B2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3991594, upper bound: 0.3991551
time: 1.16 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.55 seconds
NS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 8, lower bound: -0.3933460, upper bound: 0.3945046
NS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 8, lower bound: -0.3932647, upper bound: 0.3932647
NS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 8, lower bound: -0.3933460, upper bound: 0.3945046
NS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 8, lower bound: -0.3932647, upper bound: 0.3934491
NS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 8, lower bound: -0.3946879, upper bound: 0.3979711
NS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 8, lower bound: -0.3934492, upper bound: 0.3978609
NS_B2_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 8, lower bound: -0.3429368, upper bound: 0.3710784
NS_B2_A2_A1_B2, status: Status.VERIFIED, split count: 4, time: 4.55
Output dim: 8, lower bound: -0.3427511, upper bound: 0.3696158
NS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 8, lower bound: -0.3992466, upper bound: 0.4005849
NS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.55
Output dim: 8, lower bound: -0.3991594, upper bound: 0.3991551

## BFS NS instance: NS_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0493161, 0.0592702, -0.0300886, 0.0319900, -0.0813061, 0.0893587
1: -0.0371491, 0.0498863, -0.0208080, 0.0172889, -0.0544380, 0.0706943
2: -0.0443956, 0.0706533, -0.0185600, 0.0420460, -0.0864416, 0.0892133
3: -0.0263210, 0.0598357, -0.0135077, 0.0225847, -0.0489057, 0.0733434
4: -0.0703086, 0.0439691, -0.0350061, 0.0219175, -0.0922261, 0.0789753
5: -0.0436823, 0.0613918, -0.0203279, 0.0384139, -0.0820962, 0.0817197
6: -0.0371571, 0.0918090, -0.0228579, 0.0291719, -0.0663291, 0.1146669
7: -0.0752574, 0.0386860, -0.0483732, 0.0166063, -0.0918637, 0.0870593
8: 0.8158978, 0.9991305, 0.9107508, 0.9952801, -0.1793823, 0.0883797
9: -0.0437454, 0.1203900, -0.0087942, 0.0702257, -0.1139711, 0.1291842

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of NS_B1_A2_A1_B1_A1

### Relational analysis result of NS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3860372, upper bound: 0.3857605
time: 1.20 seconds

## Relational analysis of NS_B1_A2_A1_B1_A2

### Relational analysis result of NS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3842318, upper bound: 0.3855984
time: 1.31 seconds

## BFS NS instance: NS_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0415373, 0.0439266, -0.0302532, 0.0347485, -0.0762858, 0.0741798
1: -0.0292860, 0.0382771, -0.0216614, 0.0181517, -0.0474377, 0.0599385
2: -0.0330326, 0.0592034, -0.0198687, 0.0423186, -0.0753512, 0.0790722
3: -0.0197783, 0.0438612, -0.0137206, 0.0236965, -0.0434748, 0.0575818
4: -0.0566975, 0.0361125, -0.0347092, 0.0227078, -0.0794053, 0.0708217
5: -0.0328216, 0.0488477, -0.0207008, 0.0384815, -0.0713031, 0.0695485
6: -0.0303319, 0.0634969, -0.0254436, 0.0244029, -0.0547348, 0.0889405
7: -0.0646503, 0.0306342, -0.0499303, 0.0172547, -0.0819051, 0.0805645
8: 0.8540106, 0.9973407, 0.9157683, 0.9969412, -0.1429306, 0.0815725
9: -0.0283667, 0.1031663, -0.0089149, 0.0703666, -0.0987333, 0.1120812

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 138

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A1_B2_A1

### Relational analysis result of NS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3842645, upper bound: 0.3807488
time: 1.61 seconds

## Relational analysis of NS_B1_A2_A1_B2_A2

### Relational analysis result of NS_B1_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3805571, upper bound: 0.3805570
time: 1.41 seconds

## BFS NS instance: NS_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0895337, 0.1227166, -0.0300886, 0.0319900, -0.1215237, 0.1528052
1: -0.0772130, 0.0956098, -0.0208080, 0.0172889, -0.0945019, 0.1164178
2: -0.0860248, 0.1223815, -0.0185600, 0.0420460, -0.1280708, 0.1409415
3: -0.0609368, 0.1191868, -0.0135077, 0.0225847, -0.0835215, 0.1326945
4: -0.1131207, 0.0830842, -0.0350061, 0.0219175, -0.1350382, 0.1180903
5: -0.0881916, 0.1156569, -0.0203279, 0.0384139, -0.1266055, 0.1359848
6: -0.0695915, 0.1624770, -0.0228579, 0.0291719, -0.0987634, 0.1853349
7: -0.1184559, 0.0805019, -0.0483732, 0.0166063, -0.1350622, 0.1288751
8: 0.7166167, 1.0102448, 0.9107508, 0.9952801, -0.2786633, 0.0994940
9: -0.0945329, 0.1778910, -0.0087942, 0.0702257, -0.1647587, 0.1866852

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 84

## Relational analysis of NS_B1_A2_A2_B1_A1

### Relational analysis result of NS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3913203, upper bound: 0.3859597
time: 1.91 seconds

## Relational analysis of NS_B1_A2_A2_B1_A2

### Relational analysis result of NS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3892725, upper bound: 0.3857526
time: 1.63 seconds

## BFS NS instance: NS_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0764560, 0.1024675, -0.0302532, 0.0347485, -0.1112046, 0.1327207
1: -0.0650724, 0.0802596, -0.0216614, 0.0181517, -0.0832242, 0.1019210
2: -0.0722894, 0.1049415, -0.0198687, 0.0423186, -0.1146080, 0.1248102
3: -0.0498689, 0.1014943, -0.0137206, 0.0236965, -0.0735654, 0.1152149
4: -0.0974571, 0.0711934, -0.0347092, 0.0227078, -0.1201650, 0.1059027
5: -0.0734323, 0.0988940, -0.0207008, 0.0384815, -0.1119138, 0.1195948
6: -0.0602143, 0.1282913, -0.0254436, 0.0244029, -0.0846172, 0.1537350
7: -0.1058780, 0.0672889, -0.0499303, 0.0172547, -0.1231327, 0.1172192
8: 0.7602831, 1.0068960, 0.9157683, 0.9969412, -0.2366582, 0.0911278
9: -0.0754798, 0.1590051, -0.0089149, 0.0703666, -0.1458465, 0.1679200

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 138

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A2_B2_A1

### Relational analysis result of NS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3894189, upper bound: 0.3809352
time: 1.50 seconds

## Relational analysis of NS_B1_A2_A2_B2_A2

### Relational analysis result of NS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3867083, upper bound: 0.3807688
time: 1.46 seconds

## BFS NS instance: NS_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0300886, 0.0319900, -0.0895337, 0.1227166, -0.1528052, 0.1215237
1: -0.0208080, 0.0172889, -0.0772130, 0.0956098, -0.1164178, 0.0945019
2: -0.0185600, 0.0420460, -0.0860248, 0.1223815, -0.1409415, 0.1280708
3: -0.0135077, 0.0225847, -0.0609368, 0.1191868, -0.1326945, 0.0835215
4: -0.0350061, 0.0219175, -0.1131207, 0.0830842, -0.1180903, 0.1350382
5: -0.0203279, 0.0384139, -0.0881916, 0.1156569, -0.1359848, 0.1266055
6: -0.0228579, 0.0291719, -0.0695915, 0.1624770, -0.1853349, 0.0987634
7: -0.0483732, 0.0166063, -0.1184559, 0.0805019, -0.1288751, 0.1350622
8: 0.9107508, 0.9952801, 0.7166167, 1.0102448, -0.0994940, 0.2786633
9: -0.0087942, 0.0702257, -0.0945329, 0.1778910, -0.1866852, 0.1647587

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 132

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of NS_B2_A1_B2_A1_B1

### Relational analysis result of NS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3859598, upper bound: 0.3913202
time: 1.31 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3857527, upper bound: 0.3892726
time: 1.43 seconds

## BFS NS instance: NS_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0302532, 0.0347485, -0.0764560, 0.1024675, -0.1327207, 0.1112046
1: -0.0216614, 0.0181517, -0.0650724, 0.0802596, -0.1019210, 0.0832242
2: -0.0198687, 0.0423186, -0.0722894, 0.1049415, -0.1248102, 0.1146080
3: -0.0137206, 0.0236965, -0.0498689, 0.1014943, -0.1152149, 0.0735654
4: -0.0347092, 0.0227078, -0.0974571, 0.0711934, -0.1059027, 0.1201650
5: -0.0207008, 0.0384815, -0.0734323, 0.0988940, -0.1195948, 0.1119138
6: -0.0254436, 0.0244029, -0.0602143, 0.1282913, -0.1537350, 0.0846172
7: -0.0499303, 0.0172547, -0.1058780, 0.0672889, -0.1172192, 0.1231327
8: 0.9157683, 0.9969412, 0.7602831, 1.0068960, -0.0911278, 0.2366582
9: -0.0089149, 0.0703666, -0.0754798, 0.1590051, -0.1679200, 0.1458465

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 138

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B2_A2_B1

### Relational analysis result of NS_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3809352, upper bound: 0.3894189
time: 1.52 seconds

## Relational analysis of NS_B2_A1_B2_A2_B2

### Relational analysis result of NS_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3807688, upper bound: 0.3867083
time: 1.41 seconds

## BFS NS instance: NS_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0895337, 0.1227166, -0.0516319, 0.0645346, -0.1540683, 0.1743485
1: -0.0772130, 0.0956098, -0.0420658, 0.0514500, -0.1286630, 0.1376756
2: -0.0860248, 0.1223815, -0.0463459, 0.0713758, -0.1574006, 0.1687274
3: -0.0609368, 0.1191868, -0.0293931, 0.0665223, -0.1274591, 0.1485799
4: -0.1131207, 0.0830842, -0.0683226, 0.0494367, -0.1625574, 0.1514068
5: -0.0881916, 0.1156569, -0.0480171, 0.0662443, -0.1544359, 0.1636740
6: -0.0695915, 0.1624770, -0.0424305, 0.0698521, -0.1394435, 0.2049076
7: -0.1184559, 0.0805019, -0.0813100, 0.0428052, -0.1612611, 0.1618119
8: 0.7166167, 1.0102448, 0.8376948, 1.0023699, -0.2857531, 0.1725500
9: -0.0945329, 0.1778910, -0.0404592, 0.1225409, -0.2170738, 0.2183501

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_B2_A2_A2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3991594, upper bound: 0.3991550
time: 1.53 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3991594, upper bound: 0.3991550
time: 1.77 seconds

## BFS NS instance: NS_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0764560, 0.1024675, -0.0512692, 0.0646019, -0.1410579, 0.1537367
1: -0.0650724, 0.0802596, -0.0437252, 0.0496784, -0.1147509, 0.1239849
2: -0.0722894, 0.1049415, -0.0452062, 0.0688056, -0.1410950, 0.1501477
3: -0.0498689, 0.1014943, -0.0289838, 0.0665824, -0.1164513, 0.1304780
4: -0.0974571, 0.0711934, -0.0635918, 0.0500431, -0.1475002, 0.1347852
5: -0.0734323, 0.0988940, -0.0463921, 0.0662359, -0.1396683, 0.1452861
6: -0.0602143, 0.1282913, -0.0445507, 0.0481282, -0.1083425, 0.1728421
7: -0.1058780, 0.0672889, -0.0830862, 0.0434823, -0.1493602, 0.1503750
8: 0.7602831, 1.0068960, 0.8629319, 1.0043633, -0.2440802, 0.1439641
9: -0.0754798, 0.1590051, -0.0347201, 0.1187355, -0.1942154, 0.1937252

Time for backsubstitution: 2.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3910537, upper bound: 0.3883700
time: 1.60 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3881792, upper bound: 0.3881662
time: 1.42 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.34 seconds
NS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3860372, upper bound: 0.3857605
NS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3842318, upper bound: 0.3855984
NS_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3842645, upper bound: 0.3807488
NS_B1_A2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3805571, upper bound: 0.3805570
NS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3913203, upper bound: 0.3859597
NS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3892725, upper bound: 0.3857526
NS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3894189, upper bound: 0.3809352
NS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3867083, upper bound: 0.3807688
NS_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3859598, upper bound: 0.3913202
NS_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3857527, upper bound: 0.3892726
NS_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3809352, upper bound: 0.3894189
NS_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3807688, upper bound: 0.3867083
NS_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3991594, upper bound: 0.3991550
NS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3991594, upper bound: 0.3991550
NS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3910537, upper bound: 0.3883700
NS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 5.34
Output dim: 8, lower bound: -0.3881792, upper bound: 0.3881662

## BFS NS instance: NS_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0262009, 0.0286623, -0.0300886, 0.0319900, -0.0581909, 0.0587508
1: -0.0179295, 0.0132254, -0.0208080, 0.0172889, -0.0352184, 0.0340333
2: -0.0138241, 0.0368627, -0.0185600, 0.0420460, -0.0558701, 0.0554227
3: -0.0080795, 0.0184067, -0.0135077, 0.0225847, -0.0306642, 0.0319144
4: -0.0261774, 0.0154560, -0.0350061, 0.0219175, -0.0480948, 0.0504621
5: -0.0121403, 0.0346352, -0.0203279, 0.0384139, -0.0505542, 0.0549631
6: -0.0194737, 0.0218038, -0.0228579, 0.0291719, -0.0486456, 0.0446616
7: -0.0448946, 0.0104632, -0.0483732, 0.0166063, -0.0615009, 0.0588364
8: 0.9255182, 0.9922602, 0.9107508, 0.9952801, -0.0697619, 0.0815094
9: -0.0076317, 0.0574137, -0.0087942, 0.0702257, -0.0778574, 0.0662079

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 122

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A1_B1_A1_B1

### Relational analysis result of NS_B1_A2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3735408, upper bound: 0.3768597
time: 1.52 seconds

## Relational analysis of NS_B1_A2_A1_B1_A1_B2

### Relational analysis result of NS_B1_A2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3734474, upper bound: 0.3740630
time: 1.28 seconds

## BFS NS instance: NS_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0232592, 0.0279195, -0.0255639, 0.0300141, -0.0532733, 0.0534834
1: -0.0165659, 0.0112434, -0.0177547, 0.0135123, -0.0300781, 0.0289981
2: -0.0120883, 0.0336172, -0.0143349, 0.0368640, -0.0489523, 0.0479521
3: -0.0065249, 0.0174813, -0.0103619, 0.0186784, -0.0252033, 0.0278432
4: -0.0217224, 0.0132565, -0.0261178, 0.0173276, -0.0390500, 0.0393743
5: -0.0104126, 0.0317682, -0.0163133, 0.0341135, -0.0445261, 0.0480815
6: -0.0187103, 0.0202806, -0.0207133, 0.0215974, -0.0403077, 0.0409939
7: -0.0428876, 0.0082479, -0.0447730, 0.0105389, -0.0534265, 0.0530209
8: 0.9319441, 0.9924997, 0.9267091, 0.9936708, -0.0617266, 0.0657906
9: -0.0080327, 0.0486745, -0.0076343, 0.0564020, -0.0644347, 0.0563087

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A1_B1_A2_B1

### Relational analysis result of NS_B1_A2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3724276, upper bound: 0.3767690
time: 9.74 seconds

## Relational analysis of NS_B1_A2_A1_B1_A2_B2

### Relational analysis result of NS_B1_A2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3722624, upper bound: 0.3739998
time: 1.17 seconds

## BFS NS instance: NS_B1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0204235, 0.0280380, -0.0302532, 0.0347485, -0.0551720, 0.0582912
1: -0.0155965, 0.0098547, -0.0216614, 0.0181517, -0.0337482, 0.0315161
2: -0.0110879, 0.0312766, -0.0198687, 0.0423186, -0.0534065, 0.0511453
3: -0.0072220, 0.0169755, -0.0137206, 0.0236965, -0.0309185, 0.0306961
4: -0.0187368, 0.0136456, -0.0347092, 0.0227078, -0.0414446, 0.0483548
5: -0.0132191, 0.0289475, -0.0207008, 0.0384815, -0.0517005, 0.0496483
6: -0.0180471, 0.0192433, -0.0254436, 0.0244029, -0.0424500, 0.0446870
7: -0.0418860, 0.0065000, -0.0499303, 0.0172547, -0.0591407, 0.0564303
8: 0.9375563, 0.9921156, 0.9157683, 0.9969412, -0.0593849, 0.0763474
9: -0.0071032, 0.0424647, -0.0089149, 0.0703666, -0.0774699, 0.0513797

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A1_B2_A1_B1

### Relational analysis result of NS_B1_A2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3805359, upper bound: 0.3805358
time: 1.19 seconds

## Relational analysis of NS_B1_A2_A1_B2_A1_B2

### Relational analysis result of NS_B1_A2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3805359, upper bound: 0.3805571
time: 1.28 seconds

## BFS NS instance: NS_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0448919, 0.0500727, -0.0300886, 0.0319900, -0.0768819, 0.0801613
1: -0.0353408, 0.0395364, -0.0208080, 0.0172889, -0.0526297, 0.0603444
2: -0.0351302, 0.0618635, -0.0185600, 0.0420460, -0.0771761, 0.0804235
3: -0.0219428, 0.0481448, -0.0135077, 0.0225847, -0.0445276, 0.0616525
4: -0.0545042, 0.0405959, -0.0350061, 0.0219175, -0.0764217, 0.0756021
5: -0.0330516, 0.0558859, -0.0203279, 0.0384139, -0.0714656, 0.0762138
6: -0.0352683, 0.0384463, -0.0228579, 0.0291719, -0.0644402, 0.0613042
7: -0.0724874, 0.0365137, -0.0483732, 0.0166063, -0.0890937, 0.0848869
8: 0.8783660, 0.9961724, 0.9107508, 0.9952801, -0.1169141, 0.0854216
9: -0.0244269, 0.1075990, -0.0087942, 0.0702257, -0.0946526, 0.1163933

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 138

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A2_B1_A1_B1

### Relational analysis result of NS_B1_A2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3796415, upper bound: 0.3769985
time: 1.69 seconds

## Relational analysis of NS_B1_A2_A2_B1_A1_B2

### Relational analysis result of NS_B1_A2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3795198, upper bound: 0.3742356
time: 1.58 seconds

## BFS NS instance: NS_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0383912, 0.0368795, -0.0255639, 0.0300141, -0.0684053, 0.0624434
1: -0.0270356, 0.0285845, -0.0177547, 0.0135123, -0.0405479, 0.0463392
2: -0.0258683, 0.0534964, -0.0143349, 0.0368640, -0.0627322, 0.0678314
3: -0.0150948, 0.0321230, -0.0103619, 0.0186784, -0.0337733, 0.0424849
4: -0.0453859, 0.0313181, -0.0261178, 0.0173276, -0.0627135, 0.0574359
5: -0.0222198, 0.0463389, -0.0163133, 0.0341135, -0.0563334, 0.0626523
6: -0.0276368, 0.0306681, -0.0207133, 0.0215974, -0.0492341, 0.0513815
7: -0.0606631, 0.0280666, -0.0447730, 0.0105389, -0.0712019, 0.0728395
8: 0.8937467, 0.9941195, 0.9267091, 0.9936708, -0.0999240, 0.0674103
9: -0.0147278, 0.0941315, -0.0076343, 0.0564020, -0.0711298, 0.1017658

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 138

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A2_B1_A2_B1

### Relational analysis result of NS_B1_A2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3780251, upper bound: 0.3768975
time: 1.41 seconds

## Relational analysis of NS_B1_A2_A2_B1_A2_B2

### Relational analysis result of NS_B1_A2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3779149, upper bound: 0.3741451
time: 1.32 seconds

## BFS NS instance: NS_B1_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0355069, 0.0351230, -0.0302532, 0.0347485, -0.0702554, 0.0653762
1: -0.0247373, 0.0247402, -0.0216614, 0.0181517, -0.0428890, 0.0464016
2: -0.0232789, 0.0482869, -0.0198687, 0.0423186, -0.0655975, 0.0681556
3: -0.0153786, 0.0288542, -0.0137206, 0.0236965, -0.0390750, 0.0425748
4: -0.0411453, 0.0295964, -0.0347092, 0.0227078, -0.0638531, 0.0643057
5: -0.0233421, 0.0429618, -0.0207008, 0.0384815, -0.0618236, 0.0636626
6: -0.0258411, 0.0272678, -0.0254436, 0.0244029, -0.0502440, 0.0527114
7: -0.0564173, 0.0246444, -0.0499303, 0.0172547, -0.0736720, 0.0745747
8: 0.9033471, 0.9947039, 0.9157683, 0.9969412, -0.0935941, 0.0789356
9: -0.0112769, 0.0863070, -0.0089149, 0.0703666, -0.0816435, 0.0952219

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 138

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_A2_B2_A1_B1

### Relational analysis result of NS_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3866828, upper bound: 0.3807408
time: 1.37 seconds

## Relational analysis of NS_B1_A2_A2_B2_A1_B2

### Relational analysis result of NS_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3866828, upper bound: 0.3807687
time: 1.53 seconds

## BFS NS instance: NS_B1_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0260257, 0.0311747, -0.0247301, 0.0327541, -0.0587798, 0.0559048
1: -0.0181713, 0.0138144, -0.0179525, 0.0135406, -0.0317119, 0.0317669
2: -0.0146898, 0.0374159, -0.0153128, 0.0367113, -0.0514012, 0.0527287
3: -0.0103158, 0.0190033, -0.0104532, 0.0193011, -0.0296169, 0.0294565
4: -0.0257268, 0.0184035, -0.0251354, 0.0172772, -0.0430040, 0.0435389
5: -0.0176590, 0.0343420, -0.0174352, 0.0333082, -0.0509672, 0.0517772
6: -0.0216294, 0.0217731, -0.0229023, 0.0212042, -0.0428336, 0.0446754
7: -0.0452303, 0.0114755, -0.0449958, 0.0102249, -0.0554551, 0.0564712
8: 0.9261631, 0.9938877, 0.9285098, 0.9949025, -0.0687394, 0.0653779
9: -0.0083108, 0.0582108, -0.0083137, 0.0540328, -0.0623436, 0.0665244

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 138

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_A2_A2_B2_A2_B1

### Relational analysis result of NS_B1_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3734835, upper bound: 0.3722351
time: 1.42 seconds

## Relational analysis of NS_B1_A2_A2_B2_A2_B2

### Relational analysis result of NS_B1_A2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3732480, upper bound: 0.3675844
time: 1.55 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0300886, 0.0319900, -0.0448919, 0.0500727, -0.0801613, 0.0768819
1: -0.0208080, 0.0172889, -0.0353408, 0.0395364, -0.0603444, 0.0526297
2: -0.0185600, 0.0420460, -0.0351302, 0.0618635, -0.0804235, 0.0771761
3: -0.0135077, 0.0225847, -0.0219428, 0.0481448, -0.0616525, 0.0445276
4: -0.0350061, 0.0219175, -0.0545042, 0.0405959, -0.0756021, 0.0764217
5: -0.0203279, 0.0384139, -0.0330516, 0.0558859, -0.0762138, 0.0714656
6: -0.0228579, 0.0291719, -0.0352683, 0.0384463, -0.0613042, 0.0644402
7: -0.0483732, 0.0166063, -0.0724874, 0.0365137, -0.0848869, 0.0890937
8: 0.9107508, 0.9952801, 0.8783660, 0.9961724, -0.0854216, 0.1169141
9: -0.0087942, 0.0702257, -0.0244269, 0.1075990, -0.1163933, 0.0946526

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 138

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3769985, upper bound: 0.3796415
time: 1.67 seconds

## Relational analysis of NS_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3742356, upper bound: 0.3795198
time: 1.54 seconds

## BFS NS instance: NS_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0255639, 0.0300141, -0.0383912, 0.0368795, -0.0624434, 0.0684053
1: -0.0177547, 0.0135123, -0.0270356, 0.0285845, -0.0463392, 0.0405479
2: -0.0143349, 0.0368640, -0.0258683, 0.0534964, -0.0678314, 0.0627322
3: -0.0103619, 0.0186784, -0.0150948, 0.0321230, -0.0424849, 0.0337733
4: -0.0261178, 0.0173276, -0.0453859, 0.0313181, -0.0574359, 0.0627135
5: -0.0163133, 0.0341135, -0.0222198, 0.0463389, -0.0626523, 0.0563334
6: -0.0207133, 0.0215974, -0.0276368, 0.0306681, -0.0513815, 0.0492341
7: -0.0447730, 0.0105389, -0.0606631, 0.0280666, -0.0728395, 0.0712019
8: 0.9267091, 0.9936708, 0.8937467, 0.9941195, -0.0674103, 0.0999240
9: -0.0076343, 0.0564020, -0.0147278, 0.0941315, -0.1017658, 0.0711298

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 138

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3768975, upper bound: 0.3780251
time: 1.59 seconds

## Relational analysis of NS_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3741452, upper bound: 0.3779149
time: 2.04 seconds

## BFS NS instance: NS_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0302532, 0.0347485, -0.0355069, 0.0351230, -0.0653762, 0.0702554
1: -0.0216614, 0.0181517, -0.0247373, 0.0247402, -0.0464016, 0.0428890
2: -0.0198687, 0.0423186, -0.0232789, 0.0482869, -0.0681556, 0.0655975
3: -0.0137206, 0.0236965, -0.0153786, 0.0288542, -0.0425748, 0.0390750
4: -0.0347092, 0.0227078, -0.0411453, 0.0295964, -0.0643057, 0.0638531
5: -0.0207008, 0.0384815, -0.0233421, 0.0429618, -0.0636626, 0.0618236
6: -0.0254436, 0.0244029, -0.0258411, 0.0272678, -0.0527114, 0.0502440
7: -0.0499303, 0.0172547, -0.0564173, 0.0246444, -0.0745747, 0.0736720
8: 0.9157683, 0.9969412, 0.9033471, 0.9947039, -0.0789356, 0.0935941
9: -0.0089149, 0.0703666, -0.0112769, 0.0863070, -0.0952219, 0.0816435

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 138

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3807408, upper bound: 0.3866827
time: 2.17 seconds

## Relational analysis of NS_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3807408, upper bound: 0.3866828
time: 1.54 seconds

## BFS NS instance: NS_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0247301, 0.0327541, -0.0260257, 0.0311747, -0.0559048, 0.0587798
1: -0.0179525, 0.0135406, -0.0181713, 0.0138144, -0.0317669, 0.0317119
2: -0.0153128, 0.0367113, -0.0146898, 0.0374159, -0.0527287, 0.0514012
3: -0.0104532, 0.0193011, -0.0103158, 0.0190033, -0.0294565, 0.0296169
4: -0.0251354, 0.0172772, -0.0257268, 0.0184035, -0.0435389, 0.0430040
5: -0.0174352, 0.0333082, -0.0176590, 0.0343420, -0.0517772, 0.0509672
6: -0.0229023, 0.0212042, -0.0216294, 0.0217731, -0.0446754, 0.0428336
7: -0.0449958, 0.0102249, -0.0452303, 0.0114755, -0.0564712, 0.0554551
8: 0.9285098, 0.9949025, 0.9261631, 0.9938877, -0.0653779, 0.0687394
9: -0.0083137, 0.0540328, -0.0083108, 0.0582108, -0.0665244, 0.0623436

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 138

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3722351, upper bound: 0.3734834
time: 1.97 seconds

## Relational analysis of NS_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3675844, upper bound: 0.3732479
time: 1.44 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0469673, 0.0549290, -0.0516319, 0.0645346, -0.1115019, 0.1065608
1: -0.0371985, 0.0445935, -0.0420658, 0.0514500, -0.0886485, 0.0866593
2: -0.0394274, 0.0646318, -0.0463459, 0.0713758, -0.1108032, 0.1109777
3: -0.0253885, 0.0559221, -0.0293931, 0.0665223, -0.0919107, 0.0853152
4: -0.0603525, 0.0446249, -0.0683226, 0.0494367, -0.1097892, 0.1129475
5: -0.0412274, 0.0585657, -0.0480171, 0.0662443, -0.1074717, 0.1065828
6: -0.0378158, 0.0542037, -0.0424305, 0.0698521, -0.1076679, 0.0966343
7: -0.0748414, 0.0380237, -0.0813100, 0.0428052, -0.1176466, 0.1193336
8: 0.8591710, 1.0005426, 0.8376948, 1.0023699, -0.1431988, 0.1628478
9: -0.0313140, 0.1124656, -0.0404592, 0.1225409, -0.1538549, 0.1529248

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 84

## Relational analysis of NS_B2_A2_A2_B1_A1_B1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3907354, upper bound: 0.3941521
time: 1.24 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3905578, upper bound: 0.3920231
time: 1.29 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0472809, 0.0566355, -0.0516319, 0.0645346, -0.1118155, 0.1082674
1: -0.0389959, 0.0443354, -0.0420658, 0.0514500, -0.0904458, 0.0864012
2: -0.0397319, 0.0637763, -0.0463459, 0.0713758, -0.1111076, 0.1101222
3: -0.0255512, 0.0573239, -0.0293931, 0.0665223, -0.0920734, 0.0867170
4: -0.0582802, 0.0455337, -0.0683226, 0.0494367, -0.1077169, 0.1138563
5: -0.0408865, 0.0594829, -0.0480171, 0.0662443, -0.1071308, 0.1075000
6: -0.0402365, 0.0419593, -0.0424305, 0.0698521, -0.1100886, 0.0843898
7: -0.0769334, 0.0389803, -0.0813100, 0.0428052, -0.1197386, 0.1202903
8: 0.8731660, 1.0025433, 0.8376948, 1.0023699, -0.1292039, 0.1648485
9: -0.0287540, 0.1110357, -0.0404592, 0.1225409, -0.1512949, 0.1514949

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B1_A2_B1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3884639, upper bound: 0.3924356
time: 1.40 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3882723, upper bound: 0.3896700
time: 1.78 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0355069, 0.0351230, -0.0512692, 0.0646019, -0.1001087, 0.0863922
1: -0.0247373, 0.0247402, -0.0437252, 0.0496784, -0.0744158, 0.0684655
2: -0.0232789, 0.0482869, -0.0452062, 0.0688056, -0.0920844, 0.0934931
3: -0.0153786, 0.0288542, -0.0289838, 0.0665824, -0.0819610, 0.0578379
4: -0.0411453, 0.0295964, -0.0635918, 0.0500431, -0.0911884, 0.0931883
5: -0.0233421, 0.0429618, -0.0463921, 0.0662359, -0.0895781, 0.0893539
6: -0.0258411, 0.0272678, -0.0445507, 0.0481282, -0.0739693, 0.0718185
7: -0.0564173, 0.0246444, -0.0830862, 0.0434823, -0.0998996, 0.1077306
8: 0.9033471, 0.9947039, 0.8629319, 1.0043633, -0.1010162, 0.1317720
9: -0.0112769, 0.0863070, -0.0347201, 0.1187355, -0.1300124, 0.1210271

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 76

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B2_A1_B1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3881770, upper bound: 0.3881637
time: 1.09 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3881770, upper bound: 0.3881662
time: 1.55 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0260257, 0.0311747, -0.0413593, 0.0465696, -0.0725953, 0.0725340
1: -0.0181713, 0.0138144, -0.0319680, 0.0362612, -0.0544326, 0.0457824
2: -0.0146898, 0.0374159, -0.0321437, 0.0569092, -0.0715990, 0.0695596
3: -0.0103158, 0.0190033, -0.0207976, 0.0443577, -0.0546735, 0.0398009
4: -0.0257268, 0.0184035, -0.0504329, 0.0390623, -0.0647891, 0.0688364
5: -0.0176590, 0.0343420, -0.0335791, 0.0495242, -0.0671832, 0.0679211
6: -0.0216294, 0.0217731, -0.0346087, 0.0328570, -0.0544864, 0.0563817
7: -0.0452303, 0.0114755, -0.0680662, 0.0325247, -0.0777549, 0.0795417
8: 0.9261631, 0.9938877, 0.8881087, 1.0002666, -0.0741035, 0.1057789
9: -0.0083108, 0.0582108, -0.0200076, 0.1001926, -0.1085034, 0.0782183

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B2_A2_A2_B2_A2_B1

### Relational analysis result of NS_B2_A2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3749297, upper bound: 0.3791315
time: 1.49 seconds

## Relational analysis of NS_B2_A2_A2_B2_A2_B2

### Relational analysis result of NS_B2_A2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3747361, upper bound: 0.3747322
time: 1.22 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.97 seconds
NS_B1_A2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3735408, upper bound: 0.3768597
NS_B1_A2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3734474, upper bound: 0.3740630
NS_B1_A2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3724276, upper bound: 0.3767690
NS_B1_A2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3722624, upper bound: 0.3739998
NS_B1_A2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3805359, upper bound: 0.3805358
NS_B1_A2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3805359, upper bound: 0.3805571
NS_B1_A2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3796415, upper bound: 0.3769985
NS_B1_A2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3795198, upper bound: 0.3742356
NS_B1_A2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3780251, upper bound: 0.3768975
NS_B1_A2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3779149, upper bound: 0.3741451
NS_B1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3866828, upper bound: 0.3807408
NS_B1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3866828, upper bound: 0.3807687
NS_B1_A2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3734835, upper bound: 0.3722351
NS_B1_A2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3732480, upper bound: 0.3675844
NS_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3769985, upper bound: 0.3796415
NS_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3742356, upper bound: 0.3795198
NS_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3768975, upper bound: 0.3780251
NS_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3741452, upper bound: 0.3779149
NS_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3807408, upper bound: 0.3866827
NS_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3807408, upper bound: 0.3866828
NS_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3722351, upper bound: 0.3734834
NS_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3675844, upper bound: 0.3732479
NS_B2_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3907354, upper bound: 0.3941521
NS_B2_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3905578, upper bound: 0.3920231
NS_B2_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3884639, upper bound: 0.3924356
NS_B2_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3882723, upper bound: 0.3896700
NS_B2_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3881770, upper bound: 0.3881637
NS_B2_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3881770, upper bound: 0.3881662
NS_B2_A2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3749297, upper bound: 0.3791315
NS_B2_A2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 4.97
Output dim: 8, lower bound: -0.3747361, upper bound: 0.3747322

## BFS NS instance: NS_B1_A2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0355069, 0.0351230, -0.0132128, 0.0304581, -0.0659650, 0.0483359
1: -0.0247373, 0.0247402, -0.0142825, 0.0067406, -0.0314779, 0.0390227
2: -0.0232789, 0.0482869, -0.0098919, 0.0270763, -0.0503552, 0.0581788
3: -0.0153786, 0.0288542, -0.0067102, 0.0167565, -0.0321350, 0.0355644
4: -0.0411453, 0.0295964, -0.0140806, 0.0099074, -0.0510527, 0.0436770
5: -0.0233421, 0.0429618, -0.0124447, 0.0225253, -0.0458674, 0.0554065
6: -0.0258411, 0.0272678, -0.0189335, 0.0180612, -0.0439022, 0.0462013
7: -0.0564173, 0.0246444, -0.0406141, 0.0060155, -0.0624328, 0.0652585
8: 0.9033471, 0.9947039, 0.9423909, 0.9926499, -0.0893028, 0.0523130
9: -0.0112769, 0.0863070, -0.0071921, 0.0351064, -0.0463833, 0.0934991

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 242
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 138

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B1_A2_A2_B2_A1_B1_B1

### Relational analysis result of NS_B1_A2_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3757159, upper bound: 0.3723401
time: 1.66 seconds

## Relational analysis of NS_B1_A2_A2_B2_A1_B1_B2

### Relational analysis result of NS_B1_A2_A2_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3755420, upper bound: 0.3676629
time: 1.57 seconds

## BFS NS instance: NS_B1_A2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0355069, 0.0351230, -0.0062936, 0.0313966, -0.0669035, 0.0414166
1: -0.0247373, 0.0247402, -0.0127166, 0.0057723, -0.0305096, 0.0374568
2: -0.0232789, 0.0482869, -0.0078687, 0.0240528, -0.0473317, 0.0561556
3: -0.0153786, 0.0288542, -0.0063215, 0.0160741, -0.0314527, 0.0351757
4: -0.0411453, 0.0295964, -0.0105808, 0.0059915, -0.0471368, 0.0401773
5: -0.0233421, 0.0429618, -0.0112123, 0.0157280, -0.0390701, 0.0541742
6: -0.0258411, 0.0272678, -0.0190310, 0.0167830, -0.0426240, 0.0462988
7: -0.0564173, 0.0246444, -0.0386934, 0.0049931, -0.0614104, 0.0633378
8: 0.9033471, 0.9947039, 0.9452723, 0.9928901, -0.0895430, 0.0494316
9: -0.0112769, 0.0863070, -0.0071137, 0.0271979, -0.0384748, 0.0934207

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_B1_A2_A2_B2_A1_B2_A1

### Relational analysis result of NS_B1_A2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3713691, upper bound: 0.3546804
time: 1.48 seconds

## Relational analysis of NS_B1_A2_A2_B2_A1_B2_A2

### Relational analysis result of NS_B1_A2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3613644, upper bound: 0.3536946
time: 1.32 seconds

## BFS NS instance: NS_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0132128, 0.0304581, -0.0355069, 0.0351230, -0.0483359, 0.0659650
1: -0.0142825, 0.0067406, -0.0247373, 0.0247402, -0.0390227, 0.0314779
2: -0.0098919, 0.0270763, -0.0232789, 0.0482869, -0.0581788, 0.0503552
3: -0.0067102, 0.0167565, -0.0153786, 0.0288542, -0.0355644, 0.0321350
4: -0.0140806, 0.0099074, -0.0411453, 0.0295964, -0.0436770, 0.0510527
5: -0.0124447, 0.0225253, -0.0233421, 0.0429618, -0.0554065, 0.0458674
6: -0.0189335, 0.0180612, -0.0258411, 0.0272678, -0.0462013, 0.0439022
7: -0.0406141, 0.0060155, -0.0564173, 0.0246444, -0.0652585, 0.0624328
8: 0.9423909, 0.9926499, 0.9033471, 0.9947039, -0.0523130, 0.0893028
9: -0.0071921, 0.0351064, -0.0112769, 0.0863070, -0.0934991, 0.0463833

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 242
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 138

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A1_B2_A2_B1_A1_A1

### Relational analysis result of NS_B2_A1_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3723401, upper bound: 0.3757159
time: 1.79 seconds

## Relational analysis of NS_B2_A1_B2_A2_B1_A1_A2

### Relational analysis result of NS_B2_A1_B2_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3676629, upper bound: 0.3755420
time: 1.35 seconds

## BFS NS instance: NS_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0062936, 0.0313966, -0.0355069, 0.0351230, -0.0414166, 0.0669035
1: -0.0127166, 0.0057723, -0.0247373, 0.0247402, -0.0374568, 0.0305096
2: -0.0078687, 0.0240528, -0.0232789, 0.0482869, -0.0561556, 0.0473317
3: -0.0063215, 0.0160741, -0.0153786, 0.0288542, -0.0351757, 0.0314527
4: -0.0105808, 0.0059915, -0.0411453, 0.0295964, -0.0401773, 0.0471368
5: -0.0112123, 0.0157280, -0.0233421, 0.0429618, -0.0541742, 0.0390701
6: -0.0190310, 0.0167830, -0.0258411, 0.0272678, -0.0462988, 0.0426240
7: -0.0386934, 0.0049931, -0.0564173, 0.0246444, -0.0633378, 0.0614104
8: 0.9452723, 0.9928901, 0.9033471, 0.9947039, -0.0494316, 0.0895430
9: -0.0071137, 0.0271979, -0.0112769, 0.0863070, -0.0934207, 0.0384748

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 138

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3546804, upper bound: 0.3713691
time: 1.70 seconds

## Relational analysis of NS_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3536947, upper bound: 0.3613644
time: 1.18 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0469673, 0.0549290, -0.0294329, 0.0309816, -0.0779489, 0.0843619
1: -0.0371985, 0.0445935, -0.0203892, 0.0155976, -0.0527961, 0.0649827
2: -0.0394274, 0.0646318, -0.0169683, 0.0405569, -0.0799843, 0.0816001
3: -0.0253885, 0.0559221, -0.0112418, 0.0209310, -0.0463195, 0.0671638
4: -0.0603525, 0.0446249, -0.0318515, 0.0199327, -0.0802852, 0.0764763
5: -0.0412274, 0.0585657, -0.0165879, 0.0376258, -0.0788532, 0.0751536
6: -0.0378158, 0.0542037, -0.0219622, 0.0237869, -0.0616028, 0.0761659
7: -0.0748414, 0.0380237, -0.0475741, 0.0156498, -0.0904912, 0.0855977
8: 0.8591710, 1.0005426, 0.9184087, 0.9937777, -0.1346067, 0.0821339
9: -0.0313140, 0.1124656, -0.0081813, 0.0675240, -0.0988381, 0.1206469

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3840922, upper bound: 0.3828964
time: 1.30 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3815916, upper bound: 0.3828043
time: 1.18 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0392311, 0.0404512, -0.0249493, 0.0293165, -0.0685476, 0.0654005
1: -0.0289387, 0.0328056, -0.0174473, 0.0126282, -0.0415669, 0.0502529
2: -0.0283543, 0.0536560, -0.0134316, 0.0357954, -0.0641497, 0.0670876
3: -0.0185179, 0.0391794, -0.0079293, 0.0182387, -0.0367565, 0.0471087
4: -0.0474422, 0.0358727, -0.0241582, 0.0152422, -0.0626844, 0.0600309
5: -0.0295254, 0.0462023, -0.0129573, 0.0333812, -0.0629066, 0.0591596
6: -0.0304489, 0.0310873, -0.0202089, 0.0211406, -0.0515895, 0.0512962
7: -0.0637133, 0.0296860, -0.0440700, 0.0095876, -0.0733010, 0.0737560
8: 0.8924475, 0.9975607, 0.9283209, 0.9931873, -0.1007398, 0.0692398
9: -0.0167782, 0.0956308, -0.0083401, 0.0540990, -0.0708772, 0.1039708

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3839787, upper bound: 0.3815706
time: 2.02 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3814761, upper bound: 0.3814640
time: 1.31 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0472809, 0.0566355, -0.0265467, 0.0303759, -0.0776567, 0.0831823
1: -0.0389959, 0.0443354, -0.0182748, 0.0140857, -0.0530816, 0.0626102
2: -0.0397319, 0.0637763, -0.0148019, 0.0378118, -0.0775437, 0.0785782
3: -0.0255512, 0.0573239, -0.0109130, 0.0189780, -0.0445292, 0.0682368
4: -0.0582802, 0.0455337, -0.0267555, 0.0189954, -0.0772756, 0.0722892
5: -0.0408865, 0.0594829, -0.0179390, 0.0348805, -0.0757670, 0.0774219
6: -0.0402365, 0.0419593, -0.0211310, 0.0221200, -0.0623565, 0.0630903
7: -0.0769334, 0.0389803, -0.0453460, 0.0120630, -0.0889964, 0.0843263
8: 0.8731660, 1.0025433, 0.9249541, 0.9936405, -0.1204745, 0.0775893
9: -0.0287540, 0.1110357, -0.0079011, 0.0596590, -0.0884130, 0.1189369

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 138

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3882717, upper bound: 0.3896304
time: 1.15 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3882717, upper bound: 0.3896304
time: 1.14 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0380435, 0.0408650, -0.0174952, 0.0305281, -0.0685715, 0.0583602
1: -0.0279762, 0.0310310, -0.0152612, 0.0088604, -0.0368365, 0.0462922
2: -0.0280248, 0.0519102, -0.0112497, 0.0299549, -0.0579797, 0.0631599
3: -0.0179406, 0.0373102, -0.0080977, 0.0172371, -0.0351776, 0.0454079
4: -0.0457589, 0.0345756, -0.0167022, 0.0140185, -0.0597775, 0.0512778
5: -0.0290246, 0.0446597, -0.0158052, 0.0262554, -0.0552801, 0.0604649
6: -0.0311006, 0.0292023, -0.0196687, 0.0188001, -0.0499007, 0.0488710
7: -0.0621438, 0.0282639, -0.0415665, 0.0071869, -0.0693307, 0.0698304
8: 0.8968542, 0.9987814, 0.9401249, 0.9933930, -0.0965388, 0.0586566
9: -0.0153160, 0.0925155, -0.0075359, 0.0404489, -0.0557649, 0.1000514

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 138

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3882717, upper bound: 0.3896701
time: 1.28 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3882717, upper bound: 0.3896701
time: 1.15 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0355069, 0.0351230, -0.0267401, 0.0329629, -0.0684697, 0.0618631
1: -0.0247373, 0.0247402, -0.0189058, 0.0146470, -0.0393843, 0.0436461
2: -0.0232789, 0.0482869, -0.0161642, 0.0386078, -0.0618867, 0.0644510
3: -0.0153786, 0.0288542, -0.0111078, 0.0198881, -0.0352666, 0.0399620
4: -0.0411453, 0.0295964, -0.0275032, 0.0190706, -0.0602159, 0.0570997
5: -0.0233421, 0.0429618, -0.0185541, 0.0350990, -0.0584411, 0.0615160
6: -0.0258411, 0.0272678, -0.0234157, 0.0222461, -0.0480872, 0.0506835
7: -0.0564173, 0.0246444, -0.0461166, 0.0125291, -0.0689464, 0.0707610
8: 0.9033471, 0.9947039, 0.9244128, 0.9948584, -0.0915114, 0.0702911
9: -0.0112769, 0.0863070, -0.0086460, 0.0601269, -0.0714038, 0.0949530

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_B1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3773059, upper bound: 0.3792585
time: 1.49 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B1_B2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3771812, upper bound: 0.3746767
time: 1.68 seconds

## BFS NS instance: NS_B2_A2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0355069, 0.0351230, -0.0184726, 0.0330888, -0.0685956, 0.0535957
1: -0.0247373, 0.0247402, -0.0160349, 0.0097972, -0.0345345, 0.0407751
2: -0.0232789, 0.0482869, -0.0129859, 0.0310726, -0.0543515, 0.0612728
3: -0.0153786, 0.0288542, -0.0084654, 0.0184206, -0.0337992, 0.0373196
4: -0.0411453, 0.0295964, -0.0178071, 0.0142795, -0.0554248, 0.0474036
5: -0.0233421, 0.0429618, -0.0163691, 0.0271318, -0.0504739, 0.0593309
6: -0.0258411, 0.0272678, -0.0220950, 0.0189288, -0.0447698, 0.0493628
7: -0.0564173, 0.0246444, -0.0424605, 0.0087974, -0.0652147, 0.0671049
8: 0.9033471, 0.9947039, 0.9376197, 0.9943979, -0.0910508, 0.0570841
9: -0.0112769, 0.0863070, -0.0083429, 0.0419880, -0.0532649, 0.0946499

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A1

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3747538, upper bound: 0.3598437
time: 1.36 seconds

## Relational analysis of NS_B2_A2_A2_B2_A1_B2_A2

### Relational analysis result of NS_B2_A2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3632123, upper bound: 0.3588835
time: 1.38 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 4.94 seconds
NS_B1_A2_A2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3757159, upper bound: 0.3723401
NS_B1_A2_A2_B2_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3755420, upper bound: 0.3676629
NS_B1_A2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3713691, upper bound: 0.3546804
NS_B1_A2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3613644, upper bound: 0.3536946
NS_B2_A1_B2_A2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3723401, upper bound: 0.3757159
NS_B2_A1_B2_A2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3676629, upper bound: 0.3755420
NS_B2_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3546804, upper bound: 0.3713691
NS_B2_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3536947, upper bound: 0.3613644
NS_B2_A2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3840922, upper bound: 0.3828964
NS_B2_A2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3815916, upper bound: 0.3828043
NS_B2_A2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3839787, upper bound: 0.3815706
NS_B2_A2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3814761, upper bound: 0.3814640
NS_B2_A2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3882717, upper bound: 0.3896304
NS_B2_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3882717, upper bound: 0.3896304
NS_B2_A2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3882717, upper bound: 0.3896701
NS_B2_A2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3882717, upper bound: 0.3896701
NS_B2_A2_A2_B2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3773059, upper bound: 0.3792585
NS_B2_A2_A2_B2_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3771812, upper bound: 0.3746767
NS_B2_A2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3747538, upper bound: 0.3598437
NS_B2_A2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.94
Output dim: 8, lower bound: -0.3632123, upper bound: 0.3588835

## BFS NS instance: NS_B2_A2_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0241974, 0.0294440, -0.0294329, 0.0309816, -0.0551790, 0.0588769
1: -0.0170986, 0.0124903, -0.0203892, 0.0155976, -0.0326962, 0.0328795
2: -0.0132049, 0.0352023, -0.0169683, 0.0405569, -0.0537618, 0.0521706
3: -0.0094863, 0.0181807, -0.0112418, 0.0209310, -0.0304173, 0.0294225
4: -0.0231653, 0.0172174, -0.0318515, 0.0199327, -0.0430980, 0.0490689
5: -0.0166111, 0.0326092, -0.0165879, 0.0376258, -0.0542369, 0.0491971
6: -0.0199791, 0.0208699, -0.0219622, 0.0237869, -0.0437661, 0.0428320
7: -0.0437747, 0.0090982, -0.0475741, 0.0156498, -0.0594245, 0.0566723
8: 0.9300646, 0.9931545, 0.9184087, 0.9937777, -0.0637131, 0.0747458
9: -0.0076679, 0.0528125, -0.0081813, 0.0675240, -0.0751919, 0.0609938

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 138

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3815492, upper bound: 0.3828043
time: 1.55 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3815492, upper bound: 0.3828043
time: 1.43 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0202881, 0.0285406, -0.0249493, 0.0293165, -0.0496046, 0.0534900
1: -0.0155913, 0.0099435, -0.0174473, 0.0126282, -0.0282195, 0.0273908
2: -0.0112108, 0.0313527, -0.0134316, 0.0357954, -0.0470062, 0.0447843
3: -0.0076853, 0.0170259, -0.0079293, 0.0182387, -0.0259239, 0.0249552
4: -0.0186654, 0.0143004, -0.0241582, 0.0152422, -0.0339075, 0.0384587
5: -0.0142451, 0.0287940, -0.0129573, 0.0333812, -0.0476264, 0.0417513
6: -0.0186351, 0.0192376, -0.0202089, 0.0211406, -0.0397757, 0.0394465
7: -0.0417672, 0.0069948, -0.0440700, 0.0095876, -0.0513548, 0.0510648
8: 0.9378074, 0.9925894, 0.9283209, 0.9931873, -0.0553799, 0.0642685
9: -0.0073070, 0.0426413, -0.0083401, 0.0540990, -0.0614060, 0.0509813

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3814318, upper bound: 0.3814160
time: 1.25 seconds

## Relational analysis of NS_B2_A2_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_B2_A2_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3814318, upper bound: 0.3814639
time: 1.45 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0246968, 0.0320132, -0.0265467, 0.0303759, -0.0550727, 0.0585600
1: -0.0177735, 0.0132473, -0.0182748, 0.0140857, -0.0318592, 0.0315221
2: -0.0146782, 0.0363331, -0.0148019, 0.0378118, -0.0524900, 0.0511350
3: -0.0099098, 0.0190076, -0.0109130, 0.0189780, -0.0288878, 0.0299206
4: -0.0242815, 0.0173603, -0.0267555, 0.0189954, -0.0432768, 0.0441158
5: -0.0172261, 0.0331388, -0.0179390, 0.0348805, -0.0521066, 0.0510778
6: -0.0223350, 0.0211225, -0.0211310, 0.0221200, -0.0444550, 0.0422536
7: -0.0446470, 0.0103095, -0.0453460, 0.0120630, -0.0567100, 0.0556555
8: 0.9288768, 0.9943386, 0.9249541, 0.9936405, -0.0647637, 0.0693845
9: -0.0084282, 0.0541373, -0.0079011, 0.0596590, -0.0680872, 0.0620384

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 138

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3793461, upper bound: 0.3784165
time: 1.28 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3747490, upper bound: 0.3783063
time: 1.23 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0164834, 0.0323575, -0.0265467, 0.0303759, -0.0468592, 0.0589043
1: -0.0153720, 0.0086272, -0.0182748, 0.0140857, -0.0294577, 0.0269020
2: -0.0119131, 0.0295386, -0.0148019, 0.0378118, -0.0497250, 0.0443405
3: -0.0079803, 0.0178389, -0.0109130, 0.0189780, -0.0269583, 0.0287519
4: -0.0163284, 0.0129827, -0.0267555, 0.0189954, -0.0353238, 0.0397382
5: -0.0153126, 0.0253789, -0.0179390, 0.0348805, -0.0501931, 0.0433179
6: -0.0212402, 0.0186177, -0.0211310, 0.0221200, -0.0433602, 0.0397488
7: -0.0416874, 0.0082153, -0.0453460, 0.0120630, -0.0537504, 0.0535613
8: 0.9391308, 0.9941189, 0.9249541, 0.9936405, -0.0545097, 0.0691648
9: -0.0081398, 0.0396405, -0.0079011, 0.0596590, -0.0677988, 0.0475416

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 198
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 138

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A2_A1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3764212, upper bound: 0.3707281
time: 1.66 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B1_A2_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3591681, upper bound: 0.3664690
time: 1.14 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0246968, 0.0320132, -0.0174952, 0.0305281, -0.0552248, 0.0495085
1: -0.0177735, 0.0132473, -0.0152612, 0.0088604, -0.0266338, 0.0285085
2: -0.0146782, 0.0363331, -0.0112497, 0.0299549, -0.0446331, 0.0475828
3: -0.0099098, 0.0190076, -0.0080977, 0.0172371, -0.0271468, 0.0271053
4: -0.0242815, 0.0173603, -0.0167022, 0.0140185, -0.0383000, 0.0340626
5: -0.0172261, 0.0331388, -0.0158052, 0.0262554, -0.0434815, 0.0489439
6: -0.0223350, 0.0211225, -0.0196687, 0.0188001, -0.0411350, 0.0407912
7: -0.0446470, 0.0103095, -0.0415665, 0.0071869, -0.0518339, 0.0518760
8: 0.9288768, 0.9943386, 0.9401249, 0.9933930, -0.0645162, 0.0542137
9: -0.0084282, 0.0541373, -0.0075359, 0.0404489, -0.0488770, 0.0616732

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 138

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1_A1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3792096, upper bound: 0.3759513
time: 1.68 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A1_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3745849, upper bound: 0.3757799
time: 1.39 seconds

## BFS NS instance: NS_B2_A2_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0164834, 0.0323575, -0.0174952, 0.0305281, -0.0470114, 0.0498528
1: -0.0153720, 0.0086272, -0.0152612, 0.0088604, -0.0242323, 0.0238884
2: -0.0119131, 0.0295386, -0.0112497, 0.0299549, -0.0418680, 0.0407883
3: -0.0079803, 0.0178389, -0.0080977, 0.0172371, -0.0252173, 0.0259366
4: -0.0163284, 0.0129827, -0.0167022, 0.0140185, -0.0303469, 0.0296849
5: -0.0153126, 0.0253789, -0.0158052, 0.0262554, -0.0415680, 0.0411840
6: -0.0212402, 0.0186177, -0.0196687, 0.0188001, -0.0400403, 0.0382864
7: -0.0416874, 0.0082153, -0.0415665, 0.0071869, -0.0488743, 0.0497818
8: 0.9391308, 0.9941189, 0.9401249, 0.9933930, -0.0542622, 0.0539940
9: -0.0081398, 0.0396405, -0.0075359, 0.0404489, -0.0485887, 0.0471764

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 84
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 196
type: B, layer: 1, pos: 196
type: A, layer: 1, pos: 84
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 198
type: A, layer: 1, pos: 198
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 249
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 170
type: A, layer: 1, pos: 170
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A2_A1

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3792096, upper bound: 0.3758861
time: 1.53 seconds

## Relational analysis of NS_B2_A2_A2_B1_A2_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2_B1_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.3745849, upper bound: 0.3757365
time: 1.43 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 5.20 seconds
NS_B2_A2_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.20
Output dim: 8, lower bound: -0.3815492, upper bound: 0.3828043
NS_B2_A2_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.20
Output dim: 8, lower bound: -0.3815492, upper bound: 0.3828043
NS_B2_A2_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 5.20
Output dim: 8, lower bound: -0.3814318, upper bound: 0.3814160
NS_B2_A2_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 5.20
Output dim: 8, lower bound: -0.3814318, upper bound: 0.3814639
NS_B2_A2_A2_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 8, time: 5.20
Output dim: 8, lower bound: -0.3793461, upper bound: 0.3784165
NS_B2_A2_A2_B1_A2_B1_A1_A2, status: Status.VERIFIED, split count: 8, time: 5.20
Output dim: 8, lower bound: -0.3747490, upper bound: 0.3783063
NS_B2_A2_A2_B1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 8, time: 5.20
Output dim: 8, lower bound: -0.3764212, upper bound: 0.3707281
NS_B2_A2_A2_B1_A2_B1_A2_A2, status: Status.VERIFIED, split count: 8, time: 5.20
Output dim: 8, lower bound: -0.3591681, upper bound: 0.3664690
NS_B2_A2_A2_B1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 8, time: 5.20
Output dim: 8, lower bound: -0.3792096, upper bound: 0.3759513
NS_B2_A2_A2_B1_A2_B2_A1_A2, status: Status.VERIFIED, split count: 8, time: 5.20
Output dim: 8, lower bound: -0.3745849, upper bound: 0.3757799
NS_B2_A2_A2_B1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 8, time: 5.20
Output dim: 8, lower bound: -0.3792096, upper bound: 0.3758861
NS_B2_A2_A2_B1_A2_B2_A2_A2, status: Status.VERIFIED, split count: 8, time: 5.20
Output dim: 8, lower bound: -0.3745849, upper bound: 0.3757365

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.61 + 270.04 = 274.65 seconds
