## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 3.5844923581200003


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.0416027, 0.9973887, -1.0416027, 0.9973887, -2.0389907, 2.0389910)
1: (-6.0310273, 2.4925969, -6.0310273, 2.4925969, -8.5236244, 8.5236244)
2: (-3.6638064, 2.4245920, -3.6638064, 2.4245920, -6.0883980, 6.0883980)
3: (-4.0412731, 1.7521288, -4.0412731, 1.7521288, -5.7934012, 5.7934017)
4: (-2.6173491, 2.1098194, -2.6173491, 2.1098194, -4.7271686, 4.7271686)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.24 + 1.50 = 2.74 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -3.5852094, upper bound: 3.5852094

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5852094, upper bound: 3.5851896
time: 0.48 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851962, upper bound: 3.5851962
time: 0.61 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.19 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.19
Output dim: 4, lower bound: -3.5852094, upper bound: 3.5851896
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.19
Output dim: 4, lower bound: -3.5851962, upper bound: 3.5851962

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.9502153, 0.9010830, -1.0402133, 0.9959229, -1.9461381, 1.9412963
1: -5.4717979, 2.2891376, -6.0222788, 2.4895148, -7.9613123, 8.3114166
2: -3.3616776, 2.2088692, -3.6591039, 2.4212422, -5.7829199, 5.8679733
3: -3.6821830, 1.6030900, -4.0356674, 1.7498310, -5.4320126, 5.6387572
4: -2.3926487, 1.9109943, -2.6138763, 2.1067400, -4.4993882, 4.5248709

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851896, upper bound: 3.5851896
time: 0.54 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851896, upper bound: 3.5851896
time: 0.68 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.9161215, 0.8688707, -1.0350590, 0.9903964, -1.9065179, 1.9039295
1: -5.3982286, 2.2014146, -5.9963536, 2.4776187, -7.8758473, 8.1977673
2: -3.2229776, 2.1504076, -3.6421437, 2.4105136, -5.6334901, 5.7925510
3: -3.5824347, 1.5547254, -4.0169644, 1.7416220, -5.3240566, 5.5716896
4: -2.2819757, 1.8428931, -2.6008584, 2.0966792, -4.3786540, 4.4437513

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851896, upper bound: 3.5851962
time: 0.47 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5851896, upper bound: 3.5851962
time: 0.45 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.09 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.09
Output dim: 4, lower bound: -3.5851896, upper bound: 3.5851896
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.09
Output dim: 4, lower bound: -3.5851896, upper bound: 3.5851896
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.09
Output dim: 4, lower bound: -3.5851896, upper bound: 3.5851962
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.09
Output dim: 4, lower bound: -3.5851896, upper bound: 3.5851962

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.9502153, 0.9010830, -0.9502153, 0.9010830, -1.8512980, 1.8512983
1: -5.4717979, 2.2891376, -5.4717979, 2.2891376, -7.7609358, 7.7609348
2: -3.3616776, 2.2088692, -3.3616776, 2.2088692, -5.5705471, 5.5705471
3: -3.6821830, 1.6030900, -3.6821830, 1.6030900, -5.2852726, 5.2852726
4: -2.3926487, 1.9109943, -2.3926487, 1.9109943, -4.3036427, 4.3036432

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5772952, upper bound: 3.5839554
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850708, upper bound: 3.5850202
time: 0.58 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.9502153, 0.9010830, -0.9161215, 0.8688707, -1.8190857, 1.8172042
1: -5.4717979, 2.2891376, -5.3982286, 2.2014146, -7.6732116, 7.6873646
2: -3.3616776, 2.2088692, -3.2229776, 2.1504076, -5.5120850, 5.4318461
3: -3.6821830, 1.6030900, -3.5824347, 1.5547254, -5.2369075, 5.1855249
4: -2.3926487, 1.9109943, -2.2819757, 1.8428931, -4.2355418, 4.1929703

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5772952, upper bound: 3.5839554
time: 0.46 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850708, upper bound: 3.5850202
time: 0.58 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.9161215, 0.8688707, -0.9502153, 0.9010830, -1.8172045, 1.8190857
1: -5.3982286, 2.2014146, -5.4717979, 2.2891376, -7.6873655, 7.6732121
2: -3.2229776, 2.1504076, -3.3616776, 2.2088692, -5.4318461, 5.5120850
3: -3.5824347, 1.5547254, -3.6821830, 1.6030900, -5.1855245, 5.2369080
4: -2.2819757, 1.8428931, -2.3926487, 1.9109943, -4.1929703, 4.2355418

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5789360, upper bound: 3.5841476
time: 0.53 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850202, upper bound: 3.5850209
time: 0.54 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.9161215, 0.8688707, -0.9161215, 0.8688707, -1.7849922, 1.7849922
1: -5.3982286, 2.2014146, -5.3982286, 2.2014146, -7.5996413, 7.5996428
2: -3.2229776, 2.1504076, -3.2229776, 2.1504076, -5.3733850, 5.3733850
3: -3.5824347, 1.5547254, -3.5824347, 1.5547254, -5.1371603, 5.1371598
4: -2.2819757, 1.8428931, -2.2819757, 1.8428931, -4.1248689, 4.1248689

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5789360, upper bound: 3.5841787
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5850202, upper bound: 3.5850209
time: 0.55 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.24 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 4, lower bound: -3.5772952, upper bound: 3.5839554
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 4, lower bound: -3.5850708, upper bound: 3.5850202
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 4, lower bound: -3.5772952, upper bound: 3.5839554
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 4, lower bound: -3.5850708, upper bound: 3.5850202
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 4, lower bound: -3.5789360, upper bound: 3.5841476
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 4, lower bound: -3.5850202, upper bound: 3.5850209
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.24
Output dim: 4, lower bound: -3.5789360, upper bound: 3.5841787
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.24
Output dim: 4, lower bound: -3.5850202, upper bound: 3.5850209

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.9104127, 0.8594086, -0.9502153, 0.9010830, -1.8114957, 1.8096236
1: -5.2789812, 2.1993809, -5.4717979, 2.2891376, -7.5681190, 7.6711788
2: -3.2426996, 2.1281240, -3.3616776, 2.2088692, -5.4515686, 5.4898014
3: -3.5491338, 1.5419269, -3.6821830, 1.6030900, -5.1522236, 5.2241087
4: -2.2957745, 1.8329167, -2.3926487, 1.9109943, -4.2067685, 4.2255645

Time for backsubstitution: 1.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5840272, upper bound: 3.5772952
time: 0.48 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5840272, upper bound: 3.5772952
time: 0.50 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.9104127, 0.8594086, -0.9161215, 0.8688707, -1.7792834, 1.7755299
1: -5.2789812, 2.1993809, -5.3982286, 2.2014146, -7.4803948, 7.5976086
2: -3.2426996, 2.1281240, -3.2229776, 2.1504076, -5.3931074, 5.3511019
3: -3.5491338, 1.5419269, -3.5824347, 1.5547254, -5.1038589, 5.1243615
4: -2.2957745, 1.8329167, -2.2819757, 1.8428931, -4.1386676, 4.1148925

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5841476, upper bound: 3.5789360
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5841476, upper bound: 3.5850202
time: 0.47 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.8892985, 0.8421925, -0.9502153, 0.9010830, -1.7903814, 1.7924073
1: -5.2806196, 2.1398804, -5.4717979, 2.2891376, -7.5697575, 7.6116781
2: -3.1342716, 2.0976434, -3.3616776, 2.2088692, -5.3431401, 5.4593210
3: -3.4976764, 1.5155388, -3.6821830, 1.6030900, -5.1007662, 5.1977215
4: -2.2108879, 1.7941842, -2.3926487, 1.9109943, -4.1218824, 4.1868324

Time for backsubstitution: 1.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5839554, upper bound: 3.5772966
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5839554, upper bound: 3.5850716
time: 0.61 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.8892985, 0.8421925, -0.9161215, 0.8688707, -1.7581692, 1.7583138
1: -5.2806196, 2.1398804, -5.3982286, 2.2014146, -7.4820333, 7.5381088
2: -3.1342716, 2.0976434, -3.2229776, 2.1504076, -5.2846789, 5.3206210
3: -3.4976764, 1.5155388, -3.5824347, 1.5547254, -5.0524015, 5.0979733
4: -2.2108879, 1.7941842, -2.2819757, 1.8428931, -4.0537810, 4.0761600

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5839554, upper bound: 3.5788650
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5839554, upper bound: 3.5850209
time: 0.51 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.40 seconds
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 4, lower bound: -3.5840272, upper bound: 3.5772952
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 4, lower bound: -3.5840272, upper bound: 3.5772952
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 4, lower bound: -3.5841476, upper bound: 3.5789360
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.40
Output dim: 4, lower bound: -3.5841476, upper bound: 3.5850202
NS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 4, lower bound: -3.5839554, upper bound: 3.5772966
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.40
Output dim: 4, lower bound: -3.5839554, upper bound: 3.5850716
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 4, lower bound: -3.5839554, upper bound: 3.5788650
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.40
Output dim: 4, lower bound: -3.5839554, upper bound: 3.5850209

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.9104127, 0.8594086, -0.8892985, 0.8421925, -1.7526051, 1.7487071
1: -5.2789812, 2.1993809, -5.2806196, 2.1398804, -7.4188614, 7.4800005
2: -3.2426996, 2.1281240, -3.1342716, 2.0976434, -5.3403430, 5.2623959
3: -3.5491338, 1.5419269, -3.4976764, 1.5155388, -5.0646725, 5.0396032
4: -2.2957745, 1.8329167, -2.2108879, 1.7941842, -4.0899577, 4.0438046

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5817315, upper bound: 3.5848806
time: 0.50 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5828736, upper bound: 3.5849826
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.8892985, 0.8421925, -0.9104127, 0.8594086, -1.7487069, 1.7526050
1: -5.2806196, 2.1398804, -5.2789812, 2.1993809, -7.4800005, 7.4188614
2: -3.1342716, 2.0976434, -3.2426996, 2.1281240, -5.2623959, 5.3403430
3: -3.4976764, 1.5155388, -3.5491338, 1.5419269, -5.0396032, 5.0646725
4: -2.2108879, 1.7941842, -2.2957745, 1.8329167, -4.0438046, 4.0899577

Time for backsubstitution: 1.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5829347, upper bound: 3.5831195
time: 0.52 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5833642, upper bound: 3.5734205
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.8892985, 0.8421925, -0.8892985, 0.8421925, -1.7314907, 1.7314906
1: -5.2806196, 2.1398804, -5.2806196, 2.1398804, -7.4204998, 7.4204998
2: -3.1342716, 2.0976434, -3.1342716, 2.0976434, -5.2319145, 5.2319145
3: -3.4976764, 1.5155388, -3.4976764, 1.5155388, -5.0132151, 5.0132151
4: -2.2108879, 1.7941842, -2.2108879, 1.7941842, -4.0050721, 4.0050721

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5765947, upper bound: 3.5824330
time: 0.52 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5838200, upper bound: 3.5780436
time: 0.66 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.40 seconds
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 4, lower bound: -3.5817315, upper bound: 3.5848806
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.40
Output dim: 4, lower bound: -3.5828736, upper bound: 3.5849826
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 4, lower bound: -3.5829347, upper bound: 3.5831195
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 4, lower bound: -3.5833642, upper bound: 3.5734205
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 4, lower bound: -3.5765947, upper bound: 3.5824330
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.40
Output dim: 4, lower bound: -3.5838200, upper bound: 3.5780436

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.6879883, 0.6490112, -0.8892985, 0.8421925, -1.5301808, 1.5383092
1: -4.3774362, 1.6690217, -5.2806196, 2.1398804, -6.5173163, 6.9496412
2: -2.5056734, 1.6916413, -3.1342716, 2.0976434, -4.6033168, 4.8259125
3: -2.8534515, 1.2148154, -3.4976764, 1.5155388, -4.3689904, 4.7124920
4: -1.7268329, 1.4306159, -2.2108879, 1.7941842, -3.5210171, 3.6415038

Time for backsubstitution: 1.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5828555, upper bound: 3.5814780
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5828555, upper bound: 3.5814780
time: 0.49 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.0403577, 1.0339395, -0.8736009, 0.8267932, -1.8671509, 1.9075403
1: -6.9301624, 2.5237029, -5.2044764, 2.1036403, -9.0338020, 7.7281790
2: -3.7957993, 2.4799988, -3.0830657, 2.0659461, -5.8617454, 5.5630636
3: -4.4950910, 1.8370211, -3.4438510, 1.4920125, -5.9871035, 5.2808723
4: -2.6782477, 2.1385512, -2.1699572, 1.7659255, -4.4441733, 4.3085079

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849523, upper bound: 3.5815283
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5849523, upper bound: 3.5849826
time: 0.57 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.69 seconds
NS_A1_B2_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 4, lower bound: -3.5828555, upper bound: 3.5814780
NS_A1_B2_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.69
Output dim: 4, lower bound: -3.5828555, upper bound: 3.5814780
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 4, lower bound: -3.5849523, upper bound: 3.5815283
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.69
Output dim: 4, lower bound: -3.5849523, upper bound: 3.5849826

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.0403577, 1.0339395, -0.6825389, 0.6473945, -1.6877518, 1.7164783
1: -6.9301624, 2.5237029, -4.4092274, 1.6560408, -8.5862036, 6.9329290
2: -3.7957993, 2.4799988, -2.4415860, 1.6869321, -5.4827313, 4.9215837
3: -4.4950910, 1.8370211, -2.8428011, 1.2133894, -5.7084804, 4.6798220
4: -2.6782477, 2.1385512, -1.6733770, 1.4239478, -4.1021957, 3.8119278

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844417, upper bound: 3.5806809
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848872, upper bound: 3.5807714
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.0403577, 1.0339395, -1.0106112, 1.0217710, -2.0621285, 2.0445504
1: -6.9301624, 2.5237029, -6.9925671, 2.4471467, -9.3773088, 9.5162687
2: -3.7957993, 2.4799988, -3.6819937, 2.4494812, -6.2452803, 6.1619911
3: -4.4950910, 1.8370211, -4.4831657, 1.8149254, -6.3100166, 6.3201866
4: -2.6782477, 2.1385512, -2.5683951, 2.1084263, -4.7866740, 4.7069454

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5844417, upper bound: 3.5841316
time: 0.54 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848872, upper bound: 3.5807714
time: 0.54 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 2.86 seconds
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 2.86
Output dim: 4, lower bound: -3.5844417, upper bound: 3.5806809
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 2.86
Output dim: 4, lower bound: -3.5848872, upper bound: 3.5807714
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 2.86
Output dim: 4, lower bound: -3.5844417, upper bound: 3.5841316
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 2.86
Output dim: 4, lower bound: -3.5848872, upper bound: 3.5807714

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.0391917, 1.0363985, -0.6825389, 0.6473945, -1.6865857, 1.7189374
1: -7.0102324, 2.5267577, -4.4092274, 1.6560408, -8.6662731, 6.9359851
2: -3.8112779, 2.4836757, -2.4415860, 1.6869321, -5.4982100, 4.9252615
3: -4.5338335, 1.8484162, -2.8428011, 1.2133894, -5.7472229, 4.6912165
4: -2.6776671, 2.1424503, -1.6733770, 1.4239478, -4.1016150, 3.8158264

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5846756, upper bound: 3.5805160
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848510, upper bound: 3.5807714
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848510, upper bound: 3.5807714
time: 0.61 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.0391917, 1.0363985, -1.0106112, 1.0217710, -2.0609622, 2.0470097
1: -7.0102324, 2.5267577, -6.9925671, 2.4471467, -9.4573793, 9.5193253
2: -3.8112779, 2.4836757, -3.6819937, 2.4494812, -6.2607584, 6.1656694
3: -4.5338335, 1.8484162, -4.4831657, 1.8149254, -6.3487587, 6.3315811
4: -2.6776671, 2.1424503, -2.5683951, 2.1084263, -4.7860928, 4.7108440

Time for backsubstitution: 1.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5847499, upper bound: 3.5839070
time: 0.56 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848029, upper bound: 3.5847423
time: 0.52 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 3.43 seconds
NS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.43
Output dim: 4, lower bound: -3.5848510, upper bound: 3.5807714
NS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.43
Output dim: 4, lower bound: -3.5848510, upper bound: 3.5807714
NS_A1_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.43
Output dim: 4, lower bound: -3.5847499, upper bound: 3.5839070
NS_A1_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.43
Output dim: 4, lower bound: -3.5848029, upper bound: 3.5847423

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.0391917, 1.0363985, -0.5872400, 0.5523240, -1.5915155, 1.6236386
1: -7.0102324, 2.5267577, -3.8896010, 1.4470863, -8.4573193, 6.4163589
2: -3.8112779, 2.4836757, -2.1345744, 1.4741397, -5.2854161, 4.6182494
3: -4.5338335, 1.8484162, -2.4871609, 1.0689480, -5.6027813, 4.3355770
4: -2.6776671, 2.1424503, -1.4308943, 1.2404464, -3.9181135, 3.5733438

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5837804, upper bound: 3.5803641
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848445, upper bound: 3.5804062
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.0391917, 1.0363985, -0.6973647, 0.6626393, -1.7018309, 1.7337632
1: -7.0102324, 2.5267577, -4.3814802, 1.6795366, -8.6897688, 6.9082379
2: -3.8112779, 2.4836757, -2.4760828, 1.6927285, -5.5040054, 4.9597583
3: -4.5338335, 1.8484162, -2.8336501, 1.2229857, -5.7568188, 4.6820655
4: -2.6776671, 2.1424503, -1.7049348, 1.4363471, -4.1140141, 3.8473849

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -3.5837804, upper bound: 3.5803641
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -3.5848445, upper bound: 3.5804062
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.0185183, 1.0092046, -0.9043481, 0.8824180, -1.9009359, 1.9135528
1: -6.8136473, 2.4807062, -5.9176016, 2.2096024, -9.0232487, 8.3983078
2: -3.7371256, 2.4308162, -3.2874150, 2.1624341, -5.8995600, 5.7182312
3: -4.4110804, 1.8068498, -3.8301549, 1.5944717, -6.0055513, 5.6370049
4: -2.6218114, 2.0940149, -2.2800217, 1.8452930, -4.4671044, 4.3740368

Time for backsubstitution: 1.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.0391917, 1.0363985, -0.9903715, 1.0033014, -2.0424929, 2.0267701
1: -7.0102324, 2.5267577, -6.8827868, 2.3969722, -9.4072046, 9.4095440
2: -3.8112779, 2.4836757, -3.6071048, 2.4077976, -6.2190742, 6.0907803
3: -4.5338335, 1.8484162, -4.4066744, 1.7815361, -6.3153696, 6.2550898
4: -2.6776671, 2.1424503, -2.5166969, 2.0703952, -4.7480612, 4.6591473

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.74 + 58.29 = 61.03 seconds
