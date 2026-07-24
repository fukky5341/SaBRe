## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 51.042030738


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-22.1557789, 40.7848358, -22.1557789, 40.7848358, -62.9406128, 62.9406128)
1: (-24.9570370, 38.1305771, -24.9570370, 38.1305771, -63.0876160, 63.0876160)
2: (-25.5280037, 37.2813034, -25.5280037, 37.2813034, -62.8093071, 62.8093071)
3: (-30.7027779, 44.2187729, -30.7027779, 44.2187729, -74.9215546, 74.9215546)
4: (-28.9112663, 41.7009773, -28.9112663, 41.7009773, -70.6122437, 70.6122437)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.67 + 1.54 = 2.21 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -54.3000327, upper bound: 54.3000327

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2822885, upper bound: 54.1864055
time: 0.57 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1789857, upper bound: 54.1789857
time: 0.58 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.22 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 0, lower bound: -54.2822885, upper bound: 54.1864055
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.22
Output dim: 0, lower bound: -54.1789857, upper bound: 54.1789857

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -21.4620304, 39.5157166, -22.1557789, 40.7848358, -62.2468643, 61.6714935
1: -24.1841469, 36.9097633, -24.9570370, 38.1305771, -62.3147202, 61.8667984
2: -24.7331676, 36.1095581, -25.5280037, 37.2813034, -62.0144730, 61.6375618
3: -29.7504082, 42.7644386, -30.7027779, 44.2187729, -73.9691696, 73.4672165
4: -27.9875393, 40.3854408, -28.9112663, 41.7009773, -69.6885147, 69.2967072

Time for backsubstitution: 0.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1992664, upper bound: 54.1691807
time: 0.60 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2541776, upper bound: 54.1817565
time: 1.43 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -26.7091293, 48.1378326, -21.4018230, 39.3348732, -66.0439911, 69.5396423
1: -30.1031876, 45.2596588, -24.1025562, 36.8575134, -66.9607010, 69.3622055
2: -30.7034187, 44.1854095, -24.6576900, 36.0461578, -66.7495728, 68.8430939
3: -36.9264183, 52.4159775, -29.6387196, 42.7294922, -79.6559143, 82.0546875
4: -34.6874695, 49.6881943, -27.9459286, 40.2688103, -74.9562836, 77.6341248

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9842377, upper bound: 54.0653922
time: 1.00 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1747573, upper bound: 54.1747573
time: 0.97 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.79 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -54.1992664, upper bound: 54.1691807
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -54.2541776, upper bound: 54.1817565
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -53.9842377, upper bound: 54.0653922
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 2.79
Output dim: 0, lower bound: -54.1747573, upper bound: 54.1747573

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -17.7770462, 33.3612900, -21.1103840, 38.9932709, -56.7703171, 54.4716721
1: -20.0228424, 30.7962456, -23.7775726, 36.3467484, -56.3695908, 54.5738182
2: -20.5344810, 30.2653313, -24.3371181, 35.5812073, -56.1156883, 54.6024475
3: -24.5992012, 35.4297714, -29.2467651, 42.0971375, -66.6963348, 64.6765366
4: -23.1935501, 33.5928764, -27.5300293, 39.7430649, -62.9366150, 61.1229057

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0611393, upper bound: 53.9856069
time: 0.45 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0611393, upper bound: 54.1691807
time: 0.65 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -22.4064369, 41.7706070, -21.1892605, 39.1450081, -61.5514412, 62.9598465
1: -25.2263889, 38.4714890, -23.8366375, 36.5415115, -61.7678986, 62.3081169
2: -25.8467789, 37.6974525, -24.4353638, 35.7691040, -61.6158829, 62.1328011
3: -31.0495148, 44.5623512, -29.2917061, 42.3570557, -73.4065704, 73.8540573
4: -29.1854439, 42.1343079, -27.6725006, 39.8568916, -69.0423355, 69.8068008

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0723505, upper bound: 53.9856069
time: 0.55 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0723505, upper bound: 54.1817565
time: 0.82 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -23.2516384, 42.1274719, -20.3454723, 37.5356064, -60.7872391, 62.4729347
1: -26.1851673, 39.4338531, -22.9124336, 35.0550346, -61.2402000, 62.3462791
2: -26.7555904, 38.6069260, -23.4558887, 34.3186684, -61.0742455, 62.0628128
3: -32.0657310, 45.5586777, -28.1708336, 40.5960426, -72.6617737, 73.7295074
4: -30.2180138, 43.1763458, -26.5523930, 38.2984886, -68.5165024, 69.7287369

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9764153, upper bound: 53.9764153
time: 0.47 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9764153, upper bound: 54.0653922
time: 0.56 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -27.9166622, 50.7506638, -20.4541264, 37.7340622, -65.6507187, 71.2047806
1: -31.3626385, 47.0429802, -22.9955387, 35.3013458, -66.6639786, 70.0385208
2: -32.1079407, 45.9946899, -23.5878296, 34.5639496, -66.6718826, 69.5825195
3: -38.4340134, 54.6292343, -28.2404575, 40.8998489, -79.3338547, 82.8696899
4: -36.1575050, 51.6862259, -26.7245464, 38.4524918, -74.6099854, 78.4107590

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0653922, upper bound: 53.9842377
time: 0.61 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0653922, upper bound: 54.1747573
time: 0.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.00 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -54.0611393, upper bound: 53.9856069
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -54.0611393, upper bound: 54.1691807
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -54.0723505, upper bound: 53.9856069
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -54.0723505, upper bound: 54.1817565
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -53.9764153, upper bound: 53.9764153
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -53.9764153, upper bound: 54.0653922
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -54.0653922, upper bound: 53.9842377
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.00
Output dim: 0, lower bound: -54.0653922, upper bound: 54.1747573

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -17.7770462, 33.3612900, -18.4868755, 34.6143112, -52.3913498, 51.8481674
1: -20.0228424, 30.7962456, -20.8144035, 32.0120316, -52.0348740, 51.6106491
2: -20.5344810, 30.2653313, -21.3472996, 31.4293518, -51.9638329, 51.6126213
3: -24.5992012, 35.4297714, -25.5772781, 36.8826523, -61.4818497, 61.0070496
4: -23.1935501, 33.5928764, -24.1322937, 34.9039764, -58.0975037, 57.7251587

Time for backsubstitution: 0.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0611393, upper bound: 53.9863819
time: 1.01 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0611393, upper bound: 53.9863819
time: 0.81 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -17.7770462, 33.3612900, -23.0712452, 42.9994736, -60.7765198, 56.4325333
1: -20.0228424, 30.7962456, -25.9661083, 39.6324463, -59.6552734, 56.7623520
2: -20.5344810, 30.2653313, -26.6078491, 38.8119164, -59.3463974, 56.8731804
3: -24.5992012, 35.4297714, -31.9601364, 45.9458580, -70.5450592, 67.3899078
4: -23.1935501, 33.5928764, -30.0656071, 43.3875542, -66.5811005, 63.6584778

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0611393, upper bound: 54.1691806
time: 0.58 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0611393, upper bound: 54.1691807
time: 0.47 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -22.4064369, 41.7706070, -18.4868755, 34.6143112, -57.0207443, 60.2574806
1: -25.2263889, 38.4714890, -20.8144035, 32.0120316, -57.2384186, 59.2858887
2: -25.8467789, 37.6974525, -21.3472996, 31.4293518, -57.2761307, 59.0447540
3: -31.0495148, 44.5623512, -25.5772781, 36.8826523, -67.9321671, 70.1396332
4: -29.1854439, 42.1343079, -24.1322937, 34.9039764, -64.0894165, 66.2665863

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0716059, upper bound: 53.9850036
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0654712, upper bound: 53.9851354
time: 0.52 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -22.4064369, 41.7706070, -23.0712452, 42.9994736, -65.4058838, 64.8418503
1: -25.2263889, 38.4714890, -25.9661083, 39.6324463, -64.8588333, 64.4375916
2: -25.8467789, 37.6974525, -26.6078491, 38.8119164, -64.6586914, 64.3052979
3: -31.0495148, 44.5623512, -31.9601364, 45.9458580, -76.9953690, 76.5224915
4: -29.1854439, 42.1343079, -30.0656071, 43.3875542, -72.5729980, 72.1999054

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0723505, upper bound: 54.1791501
time: 0.46 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0723505, upper bound: 54.1791501
time: 0.57 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -23.2516384, 42.1274719, -17.7131004, 33.1498032, -56.4014397, 59.8405647
1: -26.1851673, 39.4338531, -19.9430618, 30.7202320, -56.9053917, 59.3769112
2: -26.7555904, 38.6069260, -20.4552021, 30.1702194, -56.9258003, 59.0621185
3: -32.0657310, 45.5586777, -24.4933662, 35.3640099, -67.4297333, 70.0520477
4: -30.2180138, 43.1763458, -23.1366615, 33.4625473, -63.6805611, 66.3130035

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9764153, upper bound: 53.9764153
time: 0.60 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9764153, upper bound: 53.9764153
time: 0.70 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -23.2516384, 42.1274719, -22.1684475, 41.1879005, -64.4395218, 64.2959137
1: -26.1851673, 39.4338531, -24.9253387, 38.0477867, -64.2329559, 64.3591843
2: -26.7555904, 38.6069260, -25.5649681, 37.2771454, -64.0327301, 64.1718903
3: -32.0657310, 45.5586777, -30.6560726, 44.0980186, -76.1637497, 76.2147522
4: -30.2180138, 43.1763458, -28.8915005, 41.5944061, -71.8124237, 72.0678482

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9764153, upper bound: 54.0653922
time: 0.52 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9764153, upper bound: 54.0653922
time: 0.63 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -27.9166622, 50.7506638, -17.7131004, 33.1498032, -61.0664482, 68.4637604
1: -31.3626385, 47.0429802, -19.9430618, 30.7202320, -62.0828705, 66.9860382
2: -32.1079407, 45.9946899, -20.4552021, 30.1702194, -62.2781487, 66.4498901
3: -38.4340134, 54.6292343, -24.4933662, 35.3640099, -73.7979965, 79.1226044
4: -36.1575050, 51.6862259, -23.1366615, 33.4625473, -69.6200562, 74.8228912

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0653921, upper bound: 53.9842377
time: 0.56 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0653921, upper bound: 53.9842377
time: 0.57 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -27.9166622, 50.7506638, -22.1684475, 41.1879005, -69.1045456, 72.9191132
1: -31.3626385, 47.0429802, -24.9253387, 38.0477867, -69.4104233, 71.9683151
2: -32.1079407, 45.9946899, -25.5649681, 37.2771454, -69.3850708, 71.5596619
3: -38.4340134, 54.6292343, -30.6560726, 44.0980186, -82.5320206, 85.2853088
4: -36.1575050, 51.6862259, -28.8915005, 41.5944061, -77.7519073, 80.5777283

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0653922, upper bound: 54.1747573
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0653922, upper bound: 54.1747573
time: 0.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.83 seconds
NS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -54.0611393, upper bound: 53.9863819
NS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -54.0611393, upper bound: 53.9863819
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -54.0611393, upper bound: 54.1691806
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -54.0611393, upper bound: 54.1691807
NS_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -54.0716059, upper bound: 53.9850036
NS_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -54.0654712, upper bound: 53.9851354
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -54.0723505, upper bound: 54.1791501
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -54.0723505, upper bound: 54.1791501
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -53.9764153, upper bound: 53.9764153
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -53.9764153, upper bound: 53.9764153
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -53.9764153, upper bound: 54.0653922
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -53.9764153, upper bound: 54.0653922
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -54.0653921, upper bound: 53.9842377
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -54.0653921, upper bound: 53.9842377
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -54.0653922, upper bound: 54.1747573
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.83
Output dim: 0, lower bound: -54.0653922, upper bound: 54.1747573

## BFS NS instance: NS_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -17.7770462, 33.3612900, -17.7770462, 33.3612900, -51.1383324, 51.1383362
1: -20.0228424, 30.7962456, -20.0228424, 30.7962456, -50.8190880, 50.8190880
2: -20.5344810, 30.2653313, -20.5344810, 30.2653313, -50.7998123, 50.7998123
3: -24.5992012, 35.4297714, -24.5992012, 35.4297714, -60.0289726, 60.0289726
4: -23.1935501, 33.5928764, -23.1935501, 33.5928764, -56.7864151, 56.7864151

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B1_B1_A1

### Relational analysis result of NS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0490718, upper bound: 53.9714771
time: 0.48 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2

### Relational analysis result of NS_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0504792, upper bound: 53.9710799
time: 0.72 seconds

## BFS NS instance: NS_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -17.7770462, 33.3612900, -23.1562748, 41.9909935, -59.7680397, 56.5175629
1: -20.0228424, 30.7962456, -26.0854168, 39.3220367, -59.3448792, 56.8816605
2: -20.5344810, 30.2653313, -26.6512947, 38.4988022, -59.0332832, 56.9166222
3: -24.5992012, 35.4297714, -31.9547825, 45.4292755, -70.0284729, 67.3845520
4: -23.1935501, 33.5928764, -30.1156654, 43.0629501, -66.2565002, 63.7085342

Time for backsubstitution: 0.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0511266, upper bound: 53.9726544
time: 0.53 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7182918, upper bound: 53.9615342
time: 0.65 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0537105, upper bound: 53.9771132
time: 0.58 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2

### Relational analysis result of NS_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0504792, upper bound: 53.9710799
time: 0.76 seconds

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -17.7770462, 33.3612900, -22.4064369, 41.7706070, -59.5476456, 55.7677231
1: -20.0228424, 30.7962456, -25.2263889, 38.4714890, -58.4943314, 56.0226364
2: -20.5344810, 30.2653313, -25.8467789, 37.6974525, -58.2319298, 56.1121025
3: -24.5992012, 35.4297714, -31.0495148, 44.5623512, -69.1615524, 66.4792862
4: -23.1935501, 33.5928764, -29.1854439, 42.1343079, -65.3278503, 62.7783203

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0172619, upper bound: 53.9396992
time: 0.48 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0166936, upper bound: 53.9397755
time: 0.85 seconds

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -17.7770462, 33.3612900, -27.9166622, 50.7506638, -68.5277100, 61.2779312
1: -20.0228424, 30.7962456, -31.3626385, 47.0429802, -67.0658112, 62.1588821
2: -20.5344810, 30.2653313, -32.1079407, 45.9946899, -66.5291748, 62.3732605
3: -24.5992012, 35.4297714, -38.4340134, 54.6292343, -79.2284393, 73.8637695
4: -23.1935501, 33.5928764, -36.1575050, 51.6862259, -74.8797607, 69.7503815

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0748835, upper bound: 54.0081810
time: 1.00 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0166936, upper bound: 53.9397755
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -20.3654213, 38.0456085, -17.9829254, 33.7408295, -54.1062508, 56.0285263
1: -22.8911438, 35.1187973, -20.2406998, 31.1912918, -54.0824318, 55.3594971
2: -23.5145607, 34.4564972, -20.7732353, 30.6368790, -54.1514397, 55.2297325
3: -28.1487122, 40.6303978, -24.8656673, 35.9176064, -64.0662994, 65.4960556
4: -26.5712662, 38.3529472, -23.4857483, 33.9854469, -60.5567093, 61.8386841

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_A1_A1

### Relational analysis result of NS_A1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9909497, upper bound: 53.9848359
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B1_A1_A2

### Relational analysis result of NS_A1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0513981, upper bound: 53.9850036
time: 0.51 seconds

## BFS NS instance: NS_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -23.5930691, 43.1950836, -17.5553894, 32.9799500, -56.5730171, 60.7504654
1: -26.4277229, 39.5947609, -19.7590027, 30.4039822, -56.8317032, 59.3537636
2: -27.1425877, 38.8779221, -20.2822113, 29.8874607, -57.0300484, 59.1601334
3: -32.4202347, 45.8534431, -24.2648048, 34.9887543, -67.4089890, 70.1182480
4: -30.2695675, 43.5349312, -22.9095497, 33.1426964, -63.4122505, 66.4444656

Time for backsubstitution: 0.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_A2_A1

### Relational analysis result of NS_A1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9899964, upper bound: 53.9817775
time: 0.59 seconds

## Relational analysis of NS_A1_A2_B1_A2_A2

### Relational analysis result of NS_A1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0654712, upper bound: 53.9851354
time: 0.62 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -22.4064369, 41.7706070, -22.4064369, 41.7706070, -64.1770248, 64.1770248
1: -25.2263889, 38.4714890, -25.2263889, 38.4714890, -63.6978760, 63.6978760
2: -25.8467789, 37.6974525, -25.8467789, 37.6974525, -63.5442314, 63.5442314
3: -31.0495148, 44.5623512, -31.0495148, 44.5623512, -75.6118622, 75.6118622
4: -29.1854439, 42.1343079, -29.1854439, 42.1343079, -71.3197479, 71.3197479

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2057258, upper bound: 54.1504220
time: 0.63 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2143093, upper bound: 54.1621446
time: 0.82 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -22.4064369, 41.7706070, -27.9166622, 50.7506638, -73.1570740, 69.6872559
1: -25.2263889, 38.4714890, -31.3626385, 47.0429802, -72.2693710, 69.8341293
2: -25.8467789, 37.6974525, -32.1079407, 45.9946899, -71.8414688, 69.8053894
3: -31.0495148, 44.5623512, -38.4340134, 54.6292343, -85.6787491, 82.9963608
4: -29.1854439, 42.1343079, -36.1575050, 51.6862259, -80.8716660, 78.2918091

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2057258, upper bound: 54.1504220
time: 0.61 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2143093, upper bound: 54.1621446
time: 0.96 seconds

## BFS NS instance: NS_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -23.2516384, 42.1274719, -17.7770462, 33.3612900, -56.6129227, 59.9045181
1: -26.1851673, 39.4338531, -20.0228424, 30.7962456, -56.9814148, 59.4566879
2: -26.7555904, 38.6069260, -20.5344810, 30.2653313, -57.0209122, 59.1414070
3: -32.0657310, 45.5586777, -24.5992012, 35.4297714, -67.4954987, 70.1578827
4: -30.2180138, 43.1763458, -23.1935501, 33.5928764, -63.8108902, 66.3698959

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9669256, upper bound: 53.9616079
time: 0.62 seconds

## Relational analysis of NS_A2_A1_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9611735, upper bound: 53.9611735
time: 0.71 seconds

## BFS NS instance: NS_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -23.2516384, 42.1274719, -23.1562748, 41.9909935, -65.2426071, 65.2837372
1: -26.1851673, 39.4338531, -26.0854168, 39.3220367, -65.5072021, 65.5192719
2: -26.7555904, 38.6069260, -26.6512947, 38.4988022, -65.2543793, 65.2582245
3: -32.0657310, 45.5586777, -31.9547825, 45.4292755, -77.4950104, 77.5134583
4: -30.2180138, 43.1763458, -30.1156654, 43.0629501, -73.2809601, 73.2920074

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B1_B2_B1

### Relational analysis result of NS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9616079, upper bound: 53.9669256
time: 0.58 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2

### Relational analysis result of NS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9611735, upper bound: 53.9611735
time: 0.68 seconds

## BFS NS instance: NS_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -23.2516384, 42.1274719, -22.3785515, 41.6637230, -64.9153519, 64.5060196
1: -26.1851673, 39.4338531, -25.1951447, 38.3878021, -64.5729599, 64.6289978
2: -26.7555904, 38.6069260, -25.8150673, 37.6171761, -64.3727493, 64.4219971
3: -32.0657310, 45.5586777, -31.0132332, 44.4735565, -76.5392685, 76.5718994
4: -30.2180138, 43.1763458, -29.1528759, 42.0477409, -72.2657547, 72.3292236

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9824551, upper bound: 54.0501770
time: 0.74 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9669256, upper bound: 54.0594578
time: 0.62 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9693877, upper bound: 54.0541446
time: 0.71 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -23.2516384, 42.1274719, -27.7531433, 50.3157806, -73.5673981, 69.8806076
1: -26.1851673, 39.4338531, -31.1860085, 46.7014809, -72.8866348, 70.6198578
2: -26.7555904, 38.6069260, -31.9245186, 45.6688995, -72.4244843, 70.5314407
3: -32.0657310, 45.5586777, -38.2334633, 54.2549095, -86.3206406, 83.7921371
4: -30.2180138, 43.1763458, -35.9707260, 51.3419876, -81.5599976, 79.1470718

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9824551, upper bound: 54.0501770
time: 0.52 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9754729, upper bound: 54.0594579
time: 0.65 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9611735, upper bound: 54.0541446
time: 0.94 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -27.9166622, 50.7506638, -17.7770462, 33.3612900, -61.2779388, 68.5277100
1: -31.3626385, 47.0429802, -20.0228424, 30.7962456, -62.1588821, 67.0658112
2: -32.1079407, 45.9946899, -20.5344810, 30.2653313, -62.3732567, 66.5291748
3: -38.4340134, 54.6292343, -24.5992012, 35.4297714, -73.8637695, 79.2284393
4: -36.1575050, 51.6862259, -23.1935501, 33.5928764, -69.7503815, 74.8797607

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9707565, upper bound: 53.9619053
time: 0.53 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0541446, upper bound: 53.9693877
time: 0.68 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -27.9166622, 50.7506638, -23.1562748, 41.9909935, -69.9076309, 73.9069366
1: -31.3626385, 47.0429802, -26.0854168, 39.3220367, -70.6846771, 73.1283951
2: -32.1079407, 45.9946899, -26.6512947, 38.4988022, -70.6067276, 72.6459808
3: -38.4340134, 54.6292343, -31.9547825, 45.4292755, -83.8632889, 86.5840149
4: -36.1575050, 51.6862259, -30.1156654, 43.0629501, -79.2204590, 81.8018799

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B1_B2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0532152, upper bound: 53.9825748
time: 1.54 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B1_B2_B1

### Relational analysis result of NS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0594578, upper bound: 53.9754729
time: 1.24 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2

### Relational analysis result of NS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9611735, upper bound: 53.9693877
time: 0.85 seconds

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -27.9166622, 50.7506638, -22.3785515, 41.6637230, -69.5803757, 73.1292114
1: -31.3626385, 47.0429802, -25.1951447, 38.3878021, -69.7504349, 72.2381287
2: -32.1079407, 45.9946899, -25.8150673, 37.6171761, -69.7250977, 71.8097534
3: -38.4340134, 54.6292343, -31.0132332, 44.4735565, -82.9075394, 85.6424561
4: -36.1575050, 51.6862259, -29.1528759, 42.0477409, -78.2052460, 80.8390808

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1204489, upper bound: 54.1364008
time: 0.57 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0600671, upper bound: 54.1204489
time: 0.72 seconds

## BFS NS instance: NS_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -27.9166622, 50.7506638, -27.7531433, 50.3157806, -78.2324295, 78.5038071
1: -31.3626385, 47.0429802, -31.1860085, 46.7014809, -78.0641174, 78.2289886
2: -32.1079407, 45.9946899, -31.9245186, 45.6688995, -77.7768173, 77.9191895
3: -38.4340134, 54.6292343, -38.2334633, 54.2549095, -92.6889191, 92.8627014
4: -36.1575050, 51.6862259, -35.9707260, 51.3419876, -87.4994888, 87.6569366

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1610341, upper bound: 54.1246349
time: 1.01 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1204489, upper bound: 54.1204489
time: 0.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.33 seconds
NS_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.0490718, upper bound: 53.9714771
NS_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.0504792, upper bound: 53.9710799
NS_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.0537105, upper bound: 53.9771132
NS_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.0504792, upper bound: 53.9710799
NS_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.0172619, upper bound: 53.9396992
NS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.0166936, upper bound: 53.9397755
NS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.0748835, upper bound: 54.0081810
NS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.0166936, upper bound: 53.9397755
NS_A1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -53.9909497, upper bound: 53.9848359
NS_A1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.0513981, upper bound: 53.9850036
NS_A1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -53.9899964, upper bound: 53.9817775
NS_A1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.0654712, upper bound: 53.9851354
NS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.2057258, upper bound: 54.1504220
NS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.2143093, upper bound: 54.1621446
NS_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.2057258, upper bound: 54.1504220
NS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.2143093, upper bound: 54.1621446
NS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -53.9669256, upper bound: 53.9616079
NS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -53.9611735, upper bound: 53.9611735
NS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -53.9616079, upper bound: 53.9669256
NS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -53.9611735, upper bound: 53.9611735
NS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -53.9669256, upper bound: 54.0594578
NS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -53.9693877, upper bound: 54.0541446
NS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -53.9754729, upper bound: 54.0594579
NS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -53.9611735, upper bound: 54.0541446
NS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -53.9707565, upper bound: 53.9619053
NS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.0541446, upper bound: 53.9693877
NS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.0594578, upper bound: 53.9754729
NS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -53.9611735, upper bound: 53.9693877
NS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.1204489, upper bound: 54.1364008
NS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.0600671, upper bound: 54.1204489
NS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.1610341, upper bound: 54.1246349
NS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.33
Output dim: 0, lower bound: -54.1204489, upper bound: 54.1204489

## BFS NS instance: NS_A1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -17.2079239, 32.4127541, -17.5539570, 32.9662476, -50.1741714, 49.9667091
1: -19.3610115, 29.9016571, -19.7678318, 30.4206161, -49.7816238, 49.6694870
2: -19.8891678, 29.3916550, -20.2797394, 29.9027481, -49.7919159, 49.6713943
3: -23.7634315, 34.3966904, -24.2806511, 34.9897270, -58.7531586, 58.6773415
4: -22.4781494, 32.5509262, -22.9054146, 33.1723137, -55.6504593, 55.4563408

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2416957, upper bound: 54.2416957
time: 0.56 seconds

## Relational analysis of NS_A1_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2416957, upper bound: 54.2491913
time: 0.65 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -17.3456173, 32.5988235, -17.6812744, 33.1915703, -50.5371857, 50.2800980
1: -19.5256310, 30.0672913, -19.9125137, 30.6336842, -50.1592979, 49.9798012
2: -20.0441341, 29.5605125, -20.4255390, 30.1084251, -50.1525574, 49.9860535
3: -23.9777222, 34.5819740, -24.4612732, 35.2400818, -59.2178040, 59.0432472
4: -22.6333179, 32.7782593, -23.0685997, 33.4110680, -56.0443878, 55.8468590

Time for backsubstitution: 0.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2491913, upper bound: 54.2418570
time: 1.07 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2491913, upper bound: 54.2556993
time: 0.78 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -17.5539570, 32.9662476, -22.6010666, 41.0435104, -58.5974503, 55.5673141
1: -19.7678318, 30.4206161, -25.4470558, 38.4601746, -58.2280045, 55.8676720
2: -20.2797394, 29.9027481, -26.0237408, 37.6582336, -57.9379730, 55.9264832
3: -24.2806511, 34.9897270, -31.1604176, 44.4320755, -68.7127151, 66.1501465
4: -22.9054146, 33.1723137, -29.4234562, 42.0567856, -64.9622040, 62.5957718

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0490584, upper bound: 53.9710799
time: 0.82 seconds

## Relational analysis of NS_A1_A1_B1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0490584, upper bound: 53.9710799
time: 0.70 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -17.6812744, 33.1915703, -22.7344017, 41.2614822, -58.9427567, 55.9259720
1: -19.9125137, 30.6336842, -25.5986557, 38.6146011, -58.5271149, 56.2323303
2: -20.4255390, 30.1084251, -26.1717892, 37.8171082, -58.2426453, 56.2802010
3: -24.4612732, 35.2400818, -31.3480244, 44.6017151, -69.0629883, 66.5881042
4: -23.0685997, 33.4110680, -29.5632267, 42.2719345, -65.3405304, 62.9742851

Time for backsubstitution: 0.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B1_B2_B2_A1

### Relational analysis result of NS_A1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0490584, upper bound: 53.9710799
time: 0.68 seconds

## Relational analysis of NS_A1_A1_B1_B2_B2_A2

### Relational analysis result of NS_A1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0490584, upper bound: 53.9710799
time: 0.51 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -17.4691315, 32.8352776, -22.4064369, 41.7706070, -59.2397156, 55.2417145
1: -19.6763039, 30.2733688, -25.2263889, 38.4714890, -58.1477928, 55.4997559
2: -20.1833611, 29.7573318, -25.8467789, 37.6974525, -57.8808022, 55.6041107
3: -24.1675491, 34.8072128, -31.0495148, 44.5623512, -68.7299042, 65.8567276
4: -22.7919178, 33.0261192, -29.1854439, 42.1343079, -64.9262085, 62.2115631

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9625132, upper bound: 53.9893189
time: 0.71 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0782195, upper bound: 53.9940916
time: 0.70 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0782195, upper bound: 53.9940916
time: 0.76 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -16.5338707, 31.4167767, -21.6470852, 40.4636116, -56.9974823, 53.0638542
1: -18.5587463, 28.7881393, -24.3750191, 37.1996765, -55.7584229, 53.1631584
2: -19.1609478, 28.3280373, -24.9827881, 36.4679070, -55.6288452, 53.3108253
3: -22.7396259, 33.1354904, -30.0026474, 43.0691376, -65.8087616, 63.1381187
4: -21.6517124, 31.2588787, -28.2152500, 40.7338638, -62.3855705, 59.4741287

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0782195, upper bound: 53.9941310
time: 0.54 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0782195, upper bound: 53.9941310
time: 0.62 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -17.7770462, 33.3612900, -27.6787663, 50.3301163, -68.1071625, 61.0400429
1: -20.0228424, 30.7962456, -31.0957489, 46.6380844, -66.6609268, 61.8919945
2: -20.5344810, 30.2653313, -31.8336945, 45.6035843, -66.1380615, 62.0990105
3: -24.5992012, 35.4297714, -38.1030960, 54.1500931, -78.7492905, 73.5328674
4: -23.1935501, 33.5928764, -35.8439064, 51.2403145, -74.4338684, 69.4367752

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B2_B1_B1

### Relational analysis result of NS_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0226416, upper bound: 53.9528134
time: 0.50 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_B2

### Relational analysis result of NS_A1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0226416, upper bound: 54.0024345
time: 0.64 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -17.0348969, 32.0707397, -25.9742393, 47.4225922, -64.4574890, 58.0449791
1: -19.1831322, 29.5653267, -29.1398602, 43.8285370, -63.0116577, 58.7051849
2: -19.6907272, 29.0764751, -29.9018555, 42.9042702, -62.5949898, 58.9783287
3: -23.5589733, 33.9891434, -35.6756897, 50.7980423, -74.3570099, 69.6648331
4: -22.2482166, 32.2228394, -33.6830750, 48.0195236, -70.2677383, 65.9059143

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0165393, upper bound: 53.9396992
time: 0.57 seconds

## Relational analysis of NS_A1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0165393, upper bound: 53.9397755
time: 0.66 seconds

## BFS NS instance: NS_A1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -19.0665932, 35.8253899, -17.9829254, 33.7408295, -52.8074226, 53.8083115
1: -21.4488754, 33.0621719, -20.2406998, 31.1912918, -52.6401558, 53.3028679
2: -22.0280685, 32.4570274, -20.7732353, 30.6368790, -52.6649399, 53.2302551
3: -26.3753548, 38.1990280, -24.8656673, 35.9176064, -62.2929573, 63.0646935
4: -24.9311504, 36.0444450, -23.4857483, 33.9854469, -58.9165726, 59.5301933

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A2_B1_A1_A1_B1

### Relational analysis result of NS_A1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0716059, upper bound: 53.9848359
time: 1.69 seconds

## Relational analysis of NS_A1_A2_B1_A1_A1_B2

### Relational analysis result of NS_A1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0716059, upper bound: 53.9848359
time: 0.80 seconds

## BFS NS instance: NS_A1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -20.8273449, 38.6348724, -16.8669224, 31.8130741, -52.6404190, 55.5017929
1: -23.4697704, 35.9373055, -19.0084000, 29.4190254, -52.8887901, 54.9457054
2: -24.0478859, 35.2222443, -19.4994717, 28.9113693, -52.9592552, 54.7217140
3: -28.8204193, 41.5316734, -23.3645458, 33.8318634, -62.6522827, 64.8962173
4: -27.2456913, 39.2179260, -22.0933170, 32.0202446, -59.2659378, 61.3112259

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A2_B1_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0697008, upper bound: 53.9850036
time: 0.64 seconds

## Relational analysis of NS_A1_A2_B1_A1_A2_B2

### Relational analysis result of NS_A1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0697008, upper bound: 53.9850036
time: 0.62 seconds

## BFS NS instance: NS_A1_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -22.2498035, 40.8227806, -17.5553894, 32.9799500, -55.2297478, 58.3781586
1: -24.9381390, 37.4796982, -19.7590027, 30.4039822, -55.3421211, 57.2387009
2: -25.6048298, 36.8177643, -20.2822113, 29.8874607, -55.4922905, 57.0999680
3: -30.5896168, 43.3540154, -24.2648048, 34.9887543, -65.5783691, 67.6188202
4: -28.5820236, 41.1579742, -22.9095497, 33.1426964, -61.7246971, 64.0675125

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A2_B1_A2_A1_B1

### Relational analysis result of NS_A1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9899964, upper bound: 53.9817775
time: 0.83 seconds

## Relational analysis of NS_A1_A2_B1_A2_A1_B2

### Relational analysis result of NS_A1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9899964, upper bound: 53.9817775
time: 0.59 seconds

## BFS NS instance: NS_A1_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -22.2341480, 40.8952026, -16.4673214, 31.0995560, -53.3337021, 57.3625259
1: -24.9747925, 37.8177567, -18.5571671, 28.6935635, -53.6683502, 56.3749237
2: -25.6240425, 37.0815125, -19.0427608, 28.2165089, -53.8405495, 56.1242752
3: -30.6127739, 43.7908516, -22.7995014, 32.9826927, -63.5954666, 66.5903473
4: -28.8029804, 41.3763466, -21.5587769, 31.2382011, -60.0411835, 62.9351120

Time for backsubstitution: 0.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_A2_B1_A2_A2_B1

### Relational analysis result of NS_A1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0654712, upper bound: 53.9851354
time: 0.83 seconds

## Relational analysis of NS_A1_A2_B1_A2_A2_B2

### Relational analysis result of NS_A1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0654712, upper bound: 53.9851354
time: 1.58 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -20.9316540, 39.2729454, -22.4064369, 41.7706070, -62.7022400, 61.6793671
1: -23.5863914, 36.0346832, -25.2263889, 38.4714890, -62.0578804, 61.2610703
2: -24.1684208, 35.3427315, -25.8467789, 37.6974525, -61.8658562, 61.1895065
3: -29.0400372, 41.6875725, -31.0495148, 44.5623512, -73.6023865, 72.7370911
4: -27.2903099, 39.4600220, -29.1854439, 42.1343079, -69.4246140, 68.6454620

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2379379, upper bound: 54.2376635
time: 1.01 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2490936, upper bound: 54.2376635
time: 0.95 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -21.5867138, 40.3077583, -21.6662540, 40.4784241, -62.0651398, 61.9740105
1: -24.3135166, 37.0018425, -24.3943653, 37.2462654, -61.5597839, 61.3961983
2: -24.9132423, 36.2872734, -24.9990768, 36.5145149, -61.4277458, 61.2863503
3: -29.8741703, 42.7935410, -30.0197468, 43.1129341, -72.9871063, 72.8132782
4: -28.1290760, 40.5437164, -28.2336006, 40.7702713, -68.8993454, 68.7773132

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2297739, upper bound: 54.2465763
time: 0.80 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2297739, upper bound: 54.2465763
time: 0.63 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -20.9316540, 39.2729454, -27.9166622, 50.7506638, -71.6823120, 67.1895981
1: -23.5863914, 36.0346832, -31.3626385, 47.0429802, -70.6293640, 67.3973236
2: -24.1684208, 35.3427315, -32.1079407, 45.9946899, -70.1631088, 67.4506454
3: -29.0400372, 41.6875725, -38.4340134, 54.6292343, -83.6692734, 80.1215820
4: -27.2903099, 39.4600220, -36.1575050, 51.6862259, -78.9765244, 75.6175232

Time for backsubstitution: 0.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1253946, upper bound: 54.1077559
time: 0.53 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1989413, upper bound: 54.1440125
time: 0.71 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -21.5867138, 40.3077583, -27.2007713, 49.4930801, -71.0797958, 67.5085297
1: -24.3135166, 37.0018425, -30.5600548, 45.8428345, -70.1563492, 67.5618973
2: -24.9132423, 36.2872734, -31.2853298, 44.8353920, -69.7486343, 67.5726013
3: -29.8741703, 42.7935410, -37.4441566, 53.2050819, -83.0792542, 80.2376709
4: -28.1290760, 40.5437164, -35.2281075, 50.3662033, -78.4952621, 75.7718201

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B2_A2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1737183, upper bound: 54.1200057
time: 1.02 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1664991, upper bound: 54.1198930
time: 0.55 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -22.6879253, 41.1678200, -17.5539570, 32.9662476, -55.6541634, 58.7217712
1: -25.5380325, 38.5620499, -19.7678318, 30.4206161, -55.9586487, 58.3298798
2: -26.1186962, 37.7568893, -20.2797394, 29.9027481, -56.0214462, 58.0366287
3: -31.2614918, 44.5497284, -24.2806511, 34.9897270, -66.2512054, 68.8303757
4: -29.5164909, 42.1602592, -22.9054146, 33.1723137, -62.6888046, 65.0656738

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B1_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9710799, upper bound: 54.0490584
time: 0.71 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_B2

### Relational analysis result of NS_A2_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9710799, upper bound: 54.0504792
time: 0.57 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -22.8221550, 41.3871040, -17.6812744, 33.1915703, -56.0137177, 59.0683784
1: -25.6904716, 38.7174873, -19.9125137, 30.6336842, -56.3241463, 58.6300011
2: -26.2676792, 37.9167290, -20.4255390, 30.1084251, -56.3760986, 58.3422699
3: -31.4501839, 44.7205887, -24.4612732, 35.2400818, -66.6902618, 69.1818619
4: -29.6572666, 42.3763924, -23.0685997, 33.4110680, -63.0683250, 65.4449921

Time for backsubstitution: 0.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9710799, upper bound: 54.0490584
time: 0.77 seconds

## Relational analysis of NS_A2_A1_B1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9710799, upper bound: 54.0504792
time: 0.71 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -23.0224438, 41.7192841, -22.6010666, 41.0435104, -64.0659256, 64.3203506
1: -25.9240208, 39.0534134, -25.4470558, 38.4601746, -64.3841934, 64.5004730
2: -26.4950676, 38.2395287, -26.0237408, 37.6582336, -64.1532898, 64.2632675
3: -31.7411079, 45.1136665, -31.1604176, 44.4320755, -76.1731567, 76.2740860
4: -29.9233017, 42.7477303, -29.4234562, 42.0567856, -71.9800797, 72.1711884

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9611735, upper bound: 53.9611735
time: 0.59 seconds

## Relational analysis of NS_A2_A1_B1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9611735, upper bound: 53.9611735
time: 0.64 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -23.1559963, 41.9627838, -22.7344017, 41.2614822, -64.4174805, 64.6971893
1: -26.0750217, 39.2744675, -25.5986557, 38.6146011, -64.6896210, 64.8731232
2: -26.6469250, 38.4534302, -26.1717892, 37.8171082, -64.4640350, 64.6252136
3: -31.9288006, 45.3721428, -31.3480244, 44.6017151, -76.5305176, 76.7201691
4: -30.0933094, 42.9984131, -29.5632267, 42.2719345, -72.3652420, 72.5616302

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9611735, upper bound: 53.9611735
time: 0.61 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9611735, upper bound: 53.9611735
time: 0.89 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -22.6879253, 41.1678200, -22.1343155, 41.2476273, -63.9355431, 63.3021355
1: -25.5380325, 38.5620499, -24.9223690, 37.9789124, -63.5169449, 63.4844208
2: -26.1186962, 37.7568893, -25.5367928, 37.2218246, -63.3405228, 63.2936783
3: -31.2614918, 44.5497284, -30.6771145, 43.9959526, -75.2574310, 75.2268372
4: -29.5164909, 42.1602592, -28.8379803, 41.5987549, -71.1152496, 70.9982376

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B2_B1_A1_B1

### Relational analysis result of NS_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9622758, upper bound: 53.9735998
time: 0.60 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_B2

### Relational analysis result of NS_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9622758, upper bound: 54.0621525
time: 1.04 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -22.8221550, 41.3871040, -22.2933254, 41.5092888, -64.3314438, 63.6804276
1: -25.6904716, 38.7174873, -25.0954189, 38.2460632, -63.9365349, 63.8128967
2: -26.2676792, 37.9167290, -25.7180748, 37.4805183, -63.7481880, 63.6348038
3: -31.4501839, 44.7205887, -30.8885937, 44.3077393, -75.7579193, 75.6091843
4: -29.6572666, 42.3763924, -29.0426826, 41.8850784, -71.5423431, 71.4190674

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9622758, upper bound: 53.9735998
time: 0.59 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_B2

### Relational analysis result of NS_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9622758, upper bound: 54.0621525
time: 0.89 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -22.6879253, 41.1678200, -27.5281963, 49.9292374, -72.6171417, 68.6960144
1: -25.5380325, 38.5620499, -30.9350700, 46.3248138, -71.8628464, 69.4971161
2: -26.1186962, 37.7568893, -31.6674652, 45.3084259, -71.4271240, 69.4243469
3: -31.2614918, 44.5497284, -37.9242401, 53.8151321, -85.0766144, 82.4739685
4: -29.5164909, 42.1602592, -35.6801224, 50.9302139, -80.4467010, 77.8403778

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B2_B2_A1_B1

### Relational analysis result of NS_A2_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9619053, upper bound: 53.9707565
time: 0.69 seconds

## Relational analysis of NS_A2_A1_B2_B2_A1_B2

### Relational analysis result of NS_A2_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9619053, upper bound: 54.0541446
time: 0.67 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -22.8221550, 41.3871040, -27.6650105, 50.1538010, -72.9759521, 69.0521164
1: -25.6904716, 38.7174873, -31.0822487, 46.5537338, -72.2441940, 69.7997360
2: -26.2676792, 37.9167290, -31.8237247, 45.5265999, -71.7942810, 69.7404556
3: -31.4501839, 44.7205887, -38.1035233, 54.0821571, -85.5323334, 82.8241043
4: -29.6572666, 42.3763924, -35.8553314, 51.1721802, -80.8294449, 78.2317123

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B2_B2_A2_B1

### Relational analysis result of NS_A2_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9619053, upper bound: 53.9707565
time: 0.83 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_B2

### Relational analysis result of NS_A2_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9619053, upper bound: 54.0541446
time: 0.56 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -27.1459351, 49.4174004, -17.5539570, 32.9662476, -60.1121826, 66.9713516
1: -30.4976730, 45.8090210, -19.7678318, 30.4206161, -60.9182816, 65.5768433
2: -31.2257614, 44.8032074, -20.2797394, 29.9027481, -61.1285095, 65.0829391
3: -37.3664398, 53.2030945, -24.2806511, 34.9897270, -72.3561707, 77.4837418
4: -35.1837616, 50.2943001, -22.9054146, 33.1723137, -68.3560791, 73.1996994

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B1_B1_A1_B1

### Relational analysis result of NS_A2_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9812451, upper bound: 54.0548413
time: 0.71 seconds

## Relational analysis of NS_A2_A2_B1_B1_A1_B2

### Relational analysis result of NS_A2_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9812451, upper bound: 54.0563767
time: 0.50 seconds

## BFS NS instance: NS_A2_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -27.5079098, 49.9935341, -17.6812744, 33.1915703, -60.6994781, 67.6747971
1: -30.8884201, 46.3548164, -19.9125137, 30.6336842, -61.5221024, 66.2673264
2: -31.6402645, 45.3322067, -20.4255390, 30.1084251, -61.7486877, 65.7577438
3: -37.8341980, 53.8254700, -24.4612732, 35.2400818, -73.0742798, 78.2867432
4: -35.6236992, 50.8980179, -23.0685997, 33.4110680, -69.0347519, 73.9666138

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B1_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0769934, upper bound: 54.1007960
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0769934, upper bound: 54.1947430
time: 0.69 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -27.6892643, 50.3555794, -22.6010666, 41.0435104, -68.7327576, 72.9566422
1: -31.1089096, 46.6596069, -25.4470558, 38.4601746, -69.5690842, 72.1066589
2: -31.8478374, 45.6288528, -26.0237408, 37.6582336, -69.5060577, 71.6525803
3: -38.1217766, 54.1821251, -31.1604176, 44.4320755, -82.5538406, 85.3425446
4: -35.8637962, 51.2678185, -29.4234562, 42.0567856, -77.9205780, 80.6912766

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9707565, upper bound: 53.9619053
time: 0.54 seconds

## Relational analysis of NS_A2_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9707565, upper bound: 53.9693877
time: 0.85 seconds

## BFS NS instance: NS_A2_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -27.8266888, 50.5840836, -22.7344017, 41.2614822, -69.0881729, 73.3184738
1: -31.2568054, 46.8916016, -25.5986557, 38.6146011, -69.8714066, 72.4902573
2: -32.0050011, 45.8489799, -26.1717892, 37.8171082, -69.8220978, 72.0207596
3: -38.3018494, 54.4524307, -31.3480244, 44.6017151, -82.9035568, 85.8004532
4: -36.0400238, 51.5127754, -29.5632267, 42.2719345, -78.3119583, 81.0759888

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B1_B2_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9707565, upper bound: 53.9619053
time: 0.85 seconds

## Relational analysis of NS_A2_A2_B1_B2_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9707565, upper bound: 53.9693877
time: 0.51 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -26.8967381, 48.8967857, -19.4801235, 36.9443207, -63.8410568, 68.3768997
1: -30.2070484, 45.3304214, -21.8982525, 33.7243996, -63.9314461, 67.2286682
2: -30.9327736, 44.3404579, -22.4941502, 33.1094513, -64.0422134, 66.8346024
3: -36.9905014, 52.6191597, -26.9263992, 38.9688416, -75.9593353, 79.5455627
4: -34.8229904, 49.7424965, -25.3940353, 36.7423363, -71.5653229, 75.1365280

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1273751, upper bound: 54.1832143
time: 1.46 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1273751, upper bound: 54.1832143
time: 0.54 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -27.7513847, 50.4532700, -21.6157990, 40.3355522, -68.0869370, 72.0690689
1: -31.1813965, 46.7669220, -24.3604546, 37.1097221, -68.2911148, 71.1273727
2: -31.9183559, 45.7291641, -24.9428616, 36.3737679, -68.2921219, 70.6720200
3: -38.2149086, 54.3083344, -29.9993553, 42.9932442, -81.2081528, 84.3076782
4: -35.9463768, 51.3876648, -28.1840897, 40.6615143, -76.6078796, 79.5717468

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1278072, upper bound: 54.1832143
time: 0.63 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1278071, upper bound: 54.1832143
time: 0.76 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -24.9989815, 45.6509018, -26.7541809, 48.5552025, -73.5541840, 72.4050827
1: -28.0598755, 42.1997566, -30.0538425, 45.0601349, -73.1200027, 72.2535858
2: -28.7609997, 41.3211327, -30.7740154, 44.0823708, -72.8433685, 72.0951233
3: -34.3129692, 48.9402542, -36.8177872, 52.3206749, -86.6336441, 85.7580414
4: -32.3584633, 46.2080040, -34.6623344, 49.4701958, -81.8286438, 80.8703384

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1204489, upper bound: 54.1204489
time: 0.83 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1204489, upper bound: 54.1204489
time: 0.78 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -27.1733932, 49.4113731, -27.5904274, 50.0251770, -77.1985703, 77.0018005
1: -30.5476780, 45.8119125, -31.0076008, 46.4309273, -76.9786072, 76.8195114
2: -31.2554703, 44.8115463, -31.7377834, 45.4075813, -76.6630554, 76.5493317
3: -37.4488449, 53.1881409, -38.0175056, 53.9400902, -91.3889313, 91.2056351
4: -35.2109261, 50.3438759, -35.7625198, 51.0488625, -86.2597809, 86.1063995

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1204488, upper bound: 54.1204489
time: 1.08 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1204489, upper bound: 54.1204489
time: 0.53 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.38 seconds
NS_A1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.2416957, upper bound: 54.2416957
NS_A1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.2416957, upper bound: 54.2491913
NS_A1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.2491913, upper bound: 54.2418570
NS_A1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.2491913, upper bound: 54.2556993
NS_A1_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0490584, upper bound: 53.9710799
NS_A1_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0490584, upper bound: 53.9710799
NS_A1_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0490584, upper bound: 53.9710799
NS_A1_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0490584, upper bound: 53.9710799
NS_A1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0782195, upper bound: 53.9940916
NS_A1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0782195, upper bound: 53.9940916
NS_A1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0782195, upper bound: 53.9941310
NS_A1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0782195, upper bound: 53.9941310
NS_A1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0226416, upper bound: 53.9528134
NS_A1_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0226416, upper bound: 54.0024345
NS_A1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0165393, upper bound: 53.9396992
NS_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0165393, upper bound: 53.9397755
NS_A1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0716059, upper bound: 53.9848359
NS_A1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0716059, upper bound: 53.9848359
NS_A1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0697008, upper bound: 53.9850036
NS_A1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0697008, upper bound: 53.9850036
NS_A1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9899964, upper bound: 53.9817775
NS_A1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9899964, upper bound: 53.9817775
NS_A1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0654712, upper bound: 53.9851354
NS_A1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0654712, upper bound: 53.9851354
NS_A1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.2379379, upper bound: 54.2376635
NS_A1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.2490936, upper bound: 54.2376635
NS_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.2297739, upper bound: 54.2465763
NS_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.2297739, upper bound: 54.2465763
NS_A1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.1253946, upper bound: 54.1077559
NS_A1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.1989413, upper bound: 54.1440125
NS_A1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.1737183, upper bound: 54.1200057
NS_A1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.1664991, upper bound: 54.1198930
NS_A2_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9710799, upper bound: 54.0490584
NS_A2_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9710799, upper bound: 54.0504792
NS_A2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9710799, upper bound: 54.0490584
NS_A2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9710799, upper bound: 54.0504792
NS_A2_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9611735, upper bound: 53.9611735
NS_A2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9611735, upper bound: 53.9611735
NS_A2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9611735, upper bound: 53.9611735
NS_A2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9611735, upper bound: 53.9611735
NS_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9622758, upper bound: 53.9735998
NS_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9622758, upper bound: 54.0621525
NS_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9622758, upper bound: 53.9735998
NS_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9622758, upper bound: 54.0621525
NS_A2_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9619053, upper bound: 53.9707565
NS_A2_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9619053, upper bound: 54.0541446
NS_A2_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9619053, upper bound: 53.9707565
NS_A2_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9619053, upper bound: 54.0541446
NS_A2_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9812451, upper bound: 54.0548413
NS_A2_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9812451, upper bound: 54.0563767
NS_A2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0769934, upper bound: 54.1007960
NS_A2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.0769934, upper bound: 54.1947430
NS_A2_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9707565, upper bound: 53.9619053
NS_A2_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9707565, upper bound: 53.9693877
NS_A2_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9707565, upper bound: 53.9619053
NS_A2_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -53.9707565, upper bound: 53.9693877
NS_A2_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.1273751, upper bound: 54.1832143
NS_A2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.1273751, upper bound: 54.1832143
NS_A2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.1278072, upper bound: 54.1832143
NS_A2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.1278071, upper bound: 54.1832143
NS_A2_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.1204489, upper bound: 54.1204489
NS_A2_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.1204489, upper bound: 54.1204489
NS_A2_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.1204488, upper bound: 54.1204489
NS_A2_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.38
Output dim: 0, lower bound: -54.1204489, upper bound: 54.1204489

## BFS NS instance: NS_A1_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -17.2079239, 32.4127541, -17.2079239, 32.4127541, -49.6206779, 49.6206779
1: -19.3610115, 29.9016571, -19.3610115, 29.9016571, -49.2626648, 49.2626648
2: -19.8891678, 29.3916550, -19.8891678, 29.3916550, -49.2808228, 49.2808228
3: -23.7634315, 34.3966904, -23.7634315, 34.3966904, -58.1601219, 58.1601219
4: -22.4781494, 32.5509262, -22.4781494, 32.5509262, -55.0290756, 55.0290756

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3779234, upper bound: 53.7968863
time: 0.63 seconds

## Relational analysis of NS_A1_A1_B1_B1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3174578, upper bound: 53.3174578
time: 0.57 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -17.2079239, 32.4127541, -17.3456173, 32.5988235, -49.8067474, 49.7583694
1: -19.3610115, 29.9016571, -19.5256310, 30.0672913, -49.4282990, 49.4272766
2: -19.8891678, 29.3916550, -20.0441341, 29.5605125, -49.4496803, 49.4357910
3: -23.7634315, 34.3966904, -23.9777222, 34.5819740, -58.3454056, 58.3744125
4: -22.4781494, 32.5509262, -22.6333179, 32.7782593, -55.2564087, 55.1842422

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B1_A1_B2_A1

### Relational analysis result of NS_A1_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7968863, upper bound: 53.4830774
time: 0.71 seconds

## Relational analysis of NS_A1_A1_B1_B1_A1_B2_A2

### Relational analysis result of NS_A1_A1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3174578, upper bound: 53.4324896
time: 0.70 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -17.3456173, 32.5988235, -17.2079239, 32.4127541, -49.7583694, 49.8067474
1: -19.5256310, 30.0672913, -19.3610115, 29.9016571, -49.4272766, 49.4282990
2: -20.0441341, 29.5605125, -19.8891678, 29.3916550, -49.4357910, 49.4496803
3: -23.9777222, 34.5819740, -23.7634315, 34.3966904, -58.3744125, 58.3454056
4: -22.6333179, 32.7782593, -22.4781494, 32.5509262, -55.1842422, 55.2564087

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B1_A2_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4830774, upper bound: 53.8185195
time: 0.62 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2_B1_B2

### Relational analysis result of NS_A1_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4324896, upper bound: 53.3386869
time: 0.62 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -17.3456173, 32.5988235, -17.3456173, 32.5988235, -49.9444427, 49.9444427
1: -19.5256310, 30.0672913, -19.5256310, 30.0672913, -49.5929222, 49.5929222
2: -20.0441341, 29.5605125, -20.0441341, 29.5605125, -49.6046448, 49.6046448
3: -23.9777222, 34.5819740, -23.9777222, 34.5819740, -58.5596962, 58.5596962
4: -22.6333179, 32.7782593, -22.6333179, 32.7782593, -55.4115753, 55.4115753

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B1_A2_B2_B1

### Relational analysis result of NS_A1_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4830774, upper bound: 53.9147337
time: 0.63 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2_B2_B2

### Relational analysis result of NS_A1_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4324896, upper bound: 53.4537588
time: 0.78 seconds

## BFS NS instance: NS_A1_A1_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -17.2079239, 32.4127541, -22.6010666, 41.0435104, -58.2514343, 55.0138206
1: -19.3610115, 29.9016571, -25.4470558, 38.4601746, -57.8211784, 55.3487091
2: -19.8891678, 29.3916550, -26.0237408, 37.6582336, -57.5474014, 55.4153938
3: -23.7634315, 34.3966904, -31.1604176, 44.4320755, -68.1955032, 65.5571060
4: -22.4781494, 32.5509262, -29.4234562, 42.0567856, -64.5349350, 61.9743805

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_A1_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -17.3456173, 32.5988235, -22.6010666, 41.0435104, -58.3891068, 55.1998901
1: -19.5256310, 30.0672913, -25.4470558, 38.4601746, -57.9857941, 55.5143471
2: -20.0441341, 29.5605125, -26.0237408, 37.6582336, -57.7023697, 55.5842514
3: -23.9777222, 34.5819740, -31.1604176, 44.4320755, -68.4097977, 65.7423935
4: -22.6333179, 32.7782593, -29.4234562, 42.0567856, -64.6901016, 62.2017136

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_A1_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -17.2079239, 32.4127541, -22.7344017, 41.2614822, -58.4694061, 55.1471558
1: -19.3610115, 29.9016571, -25.5986557, 38.6146011, -57.9756050, 55.5003128
2: -19.8891678, 29.3916550, -26.1717892, 37.8171082, -57.7062759, 55.5634422
3: -23.7634315, 34.3966904, -31.3480244, 44.6017151, -68.3651428, 65.7447128
4: -22.4781494, 32.5509262, -29.5632267, 42.2719345, -64.7500839, 62.1141472

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_A1_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -17.3456173, 32.5988235, -22.7344017, 41.2614822, -58.6071014, 55.3332253
1: -19.5256310, 30.0672913, -25.5986557, 38.6146011, -58.1402206, 55.6659470
2: -20.0441341, 29.5605125, -26.1717892, 37.8171082, -57.8612442, 55.7322960
3: -23.9777222, 34.5819740, -31.3480244, 44.6017151, -68.5794373, 65.9300003
4: -22.6333179, 32.7782593, -29.5632267, 42.2719345, -64.9052505, 62.3414803

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -17.4691315, 32.8352776, -22.1134243, 41.2710953, -58.7402115, 54.9487000
1: -19.6763039, 30.2733688, -24.9001865, 37.9851608, -57.6614647, 55.1735535
2: -20.1833611, 29.7573318, -25.5130939, 37.2273521, -57.4107094, 55.2704239
3: -24.1675491, 34.8072128, -30.6496944, 43.9874153, -68.1549683, 65.4569092
4: -22.7919178, 33.0261192, -28.8103142, 41.6005440, -64.3924561, 61.8364334

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0145101, upper bound: 53.9762009
time: 0.93 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0729939, upper bound: 53.9872118
time: 0.71 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -17.4691315, 32.8352776, -20.6804638, 38.7241859, -56.1933136, 53.5157394
1: -19.6763039, 30.2733688, -23.2125874, 35.4786377, -55.1549377, 53.4859543
2: -20.1833611, 29.7573318, -23.8803062, 34.7942734, -54.9776344, 53.6376381
3: -24.1675491, 34.8072128, -28.5257511, 41.0759277, -65.2434769, 63.3329620
4: -22.7919178, 33.0261192, -26.9572926, 38.7657585, -61.5576706, 59.9834137

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0145101, upper bound: 53.9762009
time: 0.75 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0729939, upper bound: 53.9872118
time: 0.53 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -16.5338707, 31.4167767, -22.1134243, 41.2710953, -57.8049660, 53.5301971
1: -18.5587463, 28.7881393, -24.9001865, 37.9851608, -56.5439072, 53.6883240
2: -19.1609478, 28.3280373, -25.5130939, 37.2273521, -56.3882866, 53.8411331
3: -22.7396259, 33.1354904, -30.6496944, 43.9874153, -66.7270432, 63.7851791
4: -21.6517124, 31.2588787, -28.8103142, 41.6005440, -63.2522583, 60.0691872

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -16.5338707, 31.4167767, -20.6804638, 38.7241859, -55.2580566, 52.0972366
1: -18.5587463, 28.7881393, -23.2125874, 35.4786377, -54.0373764, 52.0007248
2: -19.1609478, 28.3280373, -23.8803062, 34.7942734, -53.9552155, 52.2083435
3: -22.7396259, 33.1354904, -28.5257511, 41.0759277, -63.8155518, 61.6612320
4: -21.6517124, 31.2588787, -26.9572926, 38.7657585, -60.4174728, 58.2161713

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0471852, upper bound: 53.9790208
time: 0.88 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0709575, upper bound: 53.9872291
time: 0.68 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -17.5539570, 32.9662476, -26.9132576, 49.0046349, -66.5585861, 59.8795052
1: -19.7678318, 30.4206161, -30.2366276, 45.4139519, -65.1817703, 60.6572380
2: -20.2797394, 29.9027481, -30.9580040, 44.4133453, -64.6930847, 60.8607483
3: -24.2806511, 34.9897270, -37.0427322, 52.7356339, -77.0162735, 72.0324554
4: -22.9054146, 33.1723137, -34.8777657, 49.8586960, -72.7641144, 68.0500793

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B2_B1_B1_A1

### Relational analysis result of NS_A1_A1_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0226416, upper bound: 53.9528134
time: 0.75 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_B1_A2

### Relational analysis result of NS_A1_A1_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0226416, upper bound: 53.9528134
time: 0.87 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -17.6812744, 33.1915703, -27.2705288, 49.5736427, -67.2549133, 60.4620972
1: -19.9125137, 30.6336842, -30.6212692, 45.9507065, -65.8632050, 61.2549477
2: -20.4255390, 30.1084251, -31.3666096, 44.9417458, -65.3672867, 61.4750328
3: -24.4612732, 35.2400818, -37.5039520, 53.3472824, -77.8085403, 72.7440338
4: -23.0685997, 33.4110680, -35.3105354, 50.4530678, -73.5216599, 68.7216034

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B2_B1_B2_A1

### Relational analysis result of NS_A1_A1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0632378, upper bound: 53.9990815
time: 0.73 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_B2_A2

### Relational analysis result of NS_A1_A1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0632378, upper bound: 54.0024346
time: 0.57 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -17.4691315, 32.8352776, -25.9742393, 47.4225922, -64.8917007, 58.8095169
1: -19.6763039, 30.2733688, -29.1398602, 43.8285370, -63.5048370, 59.4132309
2: -20.1833611, 29.7573318, -29.9018555, 42.9042702, -63.0876312, 59.6591873
3: -24.1675491, 34.8072128, -35.6756897, 50.7980423, -74.9655914, 70.4828949
4: -22.7919178, 33.0261192, -33.6830750, 48.0195236, -70.8114395, 66.7091827

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_A1_B2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -16.5227280, 31.3870964, -25.9742393, 47.4225922, -63.9453201, 57.3613358
1: -18.5455608, 28.7562618, -29.1398602, 43.8285370, -62.3740997, 57.8961182
2: -19.1467876, 28.2999191, -29.9018555, 42.9042702, -62.0510559, 58.2017708
3: -22.7221947, 33.0942078, -35.6756897, 50.7980423, -73.5202255, 68.7698898
4: -21.6301804, 31.2255955, -33.6830750, 48.0195236, -69.6497040, 64.9086456

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_A2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -19.0665932, 35.8253899, -17.2855721, 32.5094986, -51.5760918, 53.1109619
1: -21.4488754, 33.0621719, -19.4626617, 29.9990730, -51.4479408, 52.5248337
2: -22.0280685, 32.4570274, -19.9745197, 29.4919262, -51.5199814, 52.4315453
3: -26.3753548, 38.1990280, -23.9034653, 34.4958725, -60.8712273, 62.1024933
4: -24.9311504, 36.0444450, -22.5660534, 32.7013626, -57.6325111, 58.6104965

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_B1

### Relational analysis result of NS_A1_A2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.6874087, upper bound: 53.9219610
time: 0.95 seconds

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_B1

### Relational analysis result of NS_A1_A2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0440302, upper bound: 53.9815108
time: 0.62 seconds

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_A1

### Relational analysis result of NS_A1_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7820183, upper bound: 53.9471489
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_A2

### Relational analysis result of NS_A1_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0615574, upper bound: 53.9698709
time: 0.65 seconds

## BFS NS instance: NS_A1_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -19.0665932, 35.8253899, -22.6286945, 41.0853386, -60.1519318, 58.4540787
1: -21.4488754, 33.0621719, -25.4894161, 38.4517670, -59.9006424, 58.5515823
2: -22.0280685, 32.4570274, -26.0497303, 37.6586952, -59.6867561, 58.5067520
3: -26.3753548, 38.1990280, -31.2145634, 44.4052925, -70.7806396, 69.4135818
4: -24.9311504, 36.0444450, -29.4317265, 42.0964317, -67.0275726, 65.4761658

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A1

### Relational analysis result of NS_A1_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8999370, upper bound: 53.9759908
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A2

### Relational analysis result of NS_A1_A2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9842739, upper bound: 53.9829808
time: 0.82 seconds

## BFS NS instance: NS_A1_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -20.8273449, 38.6348724, -16.1751289, 30.5996284, -51.4269714, 54.8100014
1: -23.4697704, 35.9373055, -18.2364311, 28.2436733, -51.7134323, 54.1737366
2: -24.0478859, 35.2222443, -18.7095146, 27.7789421, -51.8268280, 53.9317589
3: -28.8204193, 41.5316734, -22.4085484, 32.4325104, -61.2529297, 63.9402161
4: -27.2456913, 39.2179260, -21.1841125, 30.7543716, -58.0000610, 60.4020309

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7896110, upper bound: 53.9474410
time: 0.60 seconds

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0601265, upper bound: 53.9700306
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -20.8273449, 38.6348724, -20.9691257, 37.9433517, -58.7706985, 59.6039925
1: -23.4697704, 35.9373055, -23.6437359, 35.7565727, -59.2263336, 59.5810394
2: -24.0478859, 35.2222443, -24.1442451, 35.0390778, -59.0869637, 59.3664818
3: -28.8204193, 41.5316734, -28.9734192, 41.2305984, -70.0510178, 70.5050735
4: -27.2456913, 39.2179260, -27.3266125, 39.0961952, -66.3418884, 66.5445251

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9671000, upper bound: 53.9744721
time: 0.88 seconds

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9434723, upper bound: 53.9371860
time: 0.68 seconds

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0630038, upper bound: 53.9758991
time: 0.80 seconds

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0601266, upper bound: 53.9700306
time: 0.65 seconds

## BFS NS instance: NS_A1_A2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -22.2498035, 40.8227806, -16.8703041, 31.7674103, -54.0172081, 57.6930847
1: -24.9381390, 37.4796982, -18.9943848, 29.2350883, -54.1732254, 56.4740829
2: -25.6048298, 36.8177643, -19.4989071, 28.7616863, -54.3665123, 56.3166695
3: -30.5896168, 43.3540154, -23.3179893, 33.5987587, -64.1883774, 66.6720047
4: -28.5820236, 41.1579742, -22.0071392, 31.8869877, -60.4689827, 63.1651115

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_A2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7956376, upper bound: 53.9503084
time: 0.57 seconds

## Relational analysis of NS_A1_A2_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9770613, upper bound: 53.9666365
time: 0.65 seconds

## BFS NS instance: NS_A1_A2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -22.2498035, 40.8227806, -22.1467133, 40.1593323, -62.4091339, 62.9694748
1: -24.9381390, 37.4796982, -24.9439659, 37.5061073, -62.4442444, 62.4236641
2: -25.6048298, 36.8177643, -25.4906979, 36.7675781, -62.3724060, 62.3084641
3: -30.5896168, 43.3540154, -30.5265617, 43.2697487, -73.8593674, 73.8805771
4: -28.5820236, 41.1579742, -28.7412949, 41.0896301, -69.6716461, 69.8992691

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A2_A1_B2_B1

### Relational analysis result of NS_A1_A2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9819010, upper bound: 53.9726152
time: 0.92 seconds

## Relational analysis of NS_A1_A2_B1_A2_A1_B2_B2

### Relational analysis result of NS_A1_A2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9770613, upper bound: 53.9666365
time: 0.56 seconds

## BFS NS instance: NS_A1_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -22.2341480, 40.8952026, -15.7946043, 29.9126968, -52.1468353, 56.6897964
1: -24.9747925, 37.8177567, -17.8069363, 27.5488262, -52.5236130, 55.6246948
2: -25.6240425, 37.0815125, -18.2754936, 27.1117859, -52.7358284, 55.3570023
3: -30.6127739, 43.7908516, -21.8686008, 31.6213989, -62.2341728, 65.6594543
4: -28.8029804, 41.3763466, -20.6732159, 30.0056496, -58.8086319, 62.0495605

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A2_A2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9729870, upper bound: 53.9615425
time: 0.67 seconds

## Relational analysis of NS_A1_A2_B1_A2_A2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0554984, upper bound: 53.9701549
time: 0.57 seconds

## BFS NS instance: NS_A1_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -22.2341480, 40.8952026, -19.9995861, 36.1785736, -58.4127197, 60.8947906
1: -24.9747925, 37.8177567, -22.5448933, 34.0265007, -59.0012856, 60.3626442
2: -25.6240425, 37.0815125, -23.0371437, 33.3849792, -59.0090103, 60.1186562
3: -30.6127739, 43.7908516, -27.6092186, 39.2249680, -69.8377380, 71.4000626
4: -28.8029804, 41.3763466, -26.0493698, 37.1904182, -65.9933853, 67.4257202

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_A2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0583985, upper bound: 53.9760185
time: 0.60 seconds

## Relational analysis of NS_A1_A2_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_A2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0554984, upper bound: 53.9701549
time: 0.53 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -19.9709053, 37.6195297, -19.4801235, 36.9443207, -56.9152260, 57.0996437
1: -22.4919643, 34.4881020, -21.8982525, 33.7243996, -56.2163620, 56.3863525
2: -23.0672455, 33.8458862, -22.4941502, 33.1094513, -56.1766968, 56.3400345
3: -27.6811600, 39.8655052, -26.9263992, 38.9688416, -66.6499939, 66.7919006
4: -26.0491295, 37.6885681, -25.3940353, 36.7423363, -62.7914658, 63.0826035

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_A1_B1_B1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2379381, upper bound: 54.2376635
time: 0.70 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B1_B2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2379381, upper bound: 54.2376635
time: 0.70 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -20.7535648, 38.9688454, -21.6370335, 40.4165039, -61.1700630, 60.6058807
1: -23.3915901, 35.7409630, -24.3842545, 37.1729889, -60.5645790, 60.1252022
2: -23.9654617, 35.0568199, -24.9670448, 36.4345131, -60.3999748, 60.0238647
3: -28.8036175, 41.3473244, -30.0269737, 43.0604019, -71.8640213, 71.3742981
4: -27.0659561, 39.1405182, -28.2089272, 40.7269516, -67.7929077, 67.3494415

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_A1_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2490936, upper bound: 54.2376635
time: 0.65 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B2_B2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2490936, upper bound: 54.2376635
time: 0.96 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -21.5867138, 40.3077583, -20.9316540, 39.2729454, -60.8596573, 61.2394066
1: -24.3135166, 37.0018425, -23.5863914, 36.0346832, -60.3481979, 60.5882263
2: -24.9132423, 36.2872734, -24.1684208, 35.3427315, -60.2559547, 60.4556770
3: -29.8741703, 42.7935410, -29.0400372, 41.6875725, -71.5617447, 71.8335800
4: -28.1290760, 40.5437164, -27.2903099, 39.4600220, -67.5890961, 67.8340302

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2452873, upper bound: 54.2456145
time: 0.84 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2456463, upper bound: 54.2455755
time: 0.78 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -21.5867138, 40.3077583, -21.5867138, 40.3077583, -61.8944664, 61.8944702
1: -24.3135166, 37.0018425, -24.3135166, 37.0018425, -61.3153610, 61.3153534
2: -24.9132423, 36.2872734, -24.9132423, 36.2872734, -61.2005119, 61.2005119
3: -29.8741703, 42.7935410, -29.8741703, 42.7935410, -72.6677094, 72.6677094
4: -28.1290760, 40.5437164, -28.1290760, 40.5437164, -68.6727905, 68.6727905

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2452873, upper bound: 54.2456145
time: 0.68 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2456463, upper bound: 54.2455755
time: 0.52 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -19.7382240, 37.2840958, -24.4106922, 44.5641518, -64.3023758, 61.6947784
1: -22.2689476, 34.0633774, -27.4668503, 41.1522369, -63.4211845, 61.5302277
2: -22.8086624, 33.4443474, -28.0897503, 40.3418503, -63.1505127, 61.5340958
3: -27.4345913, 39.3556671, -33.6982193, 47.6462898, -75.0808716, 73.0538864
4: -25.7743301, 37.3207817, -31.6211891, 45.2717323, -71.0460510, 68.9419708

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B2_B2_A1_B1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8763503, upper bound: 53.9673597
time: 0.51 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1215487, upper bound: 54.1055263
time: 0.83 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -20.2739372, 38.1013718, -31.3419724, 56.5433960, -76.8173294, 69.4433441
1: -22.8562202, 34.9313622, -35.2866096, 52.5778503, -75.4340668, 70.2179718
2: -23.4192066, 34.2780228, -36.0381165, 51.3734932, -74.7927017, 70.3161392
3: -28.1498909, 40.3865738, -43.3227921, 60.9897804, -89.1396637, 83.7093658
4: -26.4399033, 38.2647095, -40.5259285, 57.9199066, -84.3597946, 78.7906342

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0362276, upper bound: 54.0855195
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0362276, upper bound: 54.1440125
time: 0.91 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -18.5929279, 35.4050598, -26.1797638, 47.6393814, -66.2322998, 61.5848236
1: -20.9081707, 32.2002640, -29.4044971, 44.1266785, -65.0348511, 61.6047592
2: -21.4890232, 31.6307678, -30.1098785, 43.1783981, -64.6674194, 61.7406311
3: -25.6358967, 37.1361847, -35.9992752, 51.1909866, -76.8268814, 73.1354599
4: -24.2581673, 35.0568504, -33.8928566, 48.4198952, -72.6780472, 68.9497070

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1664991, upper bound: 54.1198930
time: 0.65 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1664993, upper bound: 54.1198930
time: 0.98 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -20.9512825, 39.1883583, -27.0350571, 49.1955185, -70.1468048, 66.2234116
1: -23.6223164, 35.9247818, -30.3783207, 45.5665550, -69.1888657, 66.3031006
2: -24.1859741, 35.2365913, -31.0952663, 44.5664978, -68.7524643, 66.3318558
3: -29.0305367, 41.5495491, -37.2243614, 52.8839340, -81.9144745, 78.7739029
4: -27.3181362, 39.3811989, -35.0166054, 50.0671692, -77.3853073, 74.3977966

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9605547, upper bound: 53.9565637
time: 0.59 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9453017, upper bound: 53.9228449
time: 0.67 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -22.6879253, 41.1678200, -17.2079239, 32.4127541, -55.1006737, 58.3757439
1: -25.5380325, 38.5620499, -19.3610115, 29.9016571, -55.4396896, 57.9230537
2: -26.1186962, 37.7568893, -19.8891678, 29.3916550, -55.5103531, 57.6460571
3: -31.2614918, 44.5497284, -23.7634315, 34.3966904, -65.6581726, 68.3131485
4: -29.5164909, 42.1602592, -22.4781494, 32.5509262, -62.0674171, 64.6383972

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -22.6879253, 41.1678200, -17.3456173, 32.5988235, -55.2867470, 58.5134354
1: -25.5380325, 38.5620499, -19.5256310, 30.0672913, -55.6053238, 58.0876694
2: -26.1186962, 37.7568893, -20.0441341, 29.5605125, -55.6792068, 57.8010254
3: -31.2614918, 44.5497284, -23.9777222, 34.5819740, -65.8434448, 68.5274506
4: -29.5164909, 42.1602592, -22.6333179, 32.7782593, -62.2947502, 64.7935715

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -22.8221550, 41.3871040, -17.2079239, 32.4127541, -55.2349091, 58.5950279
1: -25.6904716, 38.7174873, -19.3610115, 29.9016571, -55.5921249, 58.0784988
2: -26.2676792, 37.9167290, -19.8891678, 29.3916550, -55.6593323, 57.8058968
3: -31.4501839, 44.7205887, -23.7634315, 34.3966904, -65.8468781, 68.4840240
4: -29.6572666, 42.3763924, -22.4781494, 32.5509262, -62.2081871, 64.8545303

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -22.8221550, 41.3871040, -17.3456173, 32.5988235, -55.4209785, 58.7327194
1: -25.6904716, 38.7174873, -19.5256310, 30.0672913, -55.7577629, 58.2431183
2: -26.2676792, 37.9167290, -20.0441341, 29.5605125, -55.8281937, 57.9608612
3: -31.4501839, 44.7205887, -23.9777222, 34.5819740, -66.0321579, 68.6983109
4: -29.6572666, 42.3763924, -22.6333179, 32.7782593, -62.4355240, 65.0097046

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -22.6879253, 41.1678200, -22.6010666, 41.0435104, -63.7314072, 63.7688866
1: -25.5380325, 38.5620499, -25.4470558, 38.4601746, -63.9982071, 64.0091095
2: -26.1186962, 37.7568893, -26.0237408, 37.6582336, -63.7769318, 63.7806282
3: -31.2614918, 44.5497284, -31.1604176, 44.4320755, -75.6935425, 75.7101440
4: -29.5164909, 42.1602592, -29.4234562, 42.0567856, -71.5732727, 71.5837173

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -22.8221550, 41.3871040, -22.6010666, 41.0435104, -63.8656464, 63.9881706
1: -25.6904716, 38.7174873, -25.4470558, 38.4601746, -64.1506500, 64.1645432
2: -26.2676792, 37.9167290, -26.0237408, 37.6582336, -63.9259109, 63.9404678
3: -31.4501839, 44.7205887, -31.1604176, 44.4320755, -75.8822556, 75.8810043
4: -29.6572666, 42.3763924, -29.4234562, 42.0567856, -71.7140503, 71.7998505

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -22.6879253, 41.1678200, -22.7344017, 41.2614822, -63.9494095, 63.9022217
1: -25.5380325, 38.5620499, -25.5986557, 38.6146011, -64.1526337, 64.1607056
2: -26.1186962, 37.7568893, -26.1717892, 37.8171082, -63.9358063, 63.9286728
3: -31.2614918, 44.5497284, -31.3480244, 44.6017151, -75.8631821, 75.8977509
4: -29.5164909, 42.1602592, -29.5632267, 42.2719345, -71.7884216, 71.7234802

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -22.8221550, 41.3871040, -22.7344017, 41.2614822, -64.0836334, 64.1215057
1: -25.6904716, 38.7174873, -25.5986557, 38.6146011, -64.3050690, 64.3161392
2: -26.2676792, 37.9167290, -26.1717892, 37.8171082, -64.0847855, 64.0885162
3: -31.4501839, 44.7205887, -31.3480244, 44.6017151, -76.0518951, 76.0686111
4: -29.6572666, 42.3763924, -29.5632267, 42.2719345, -71.9291992, 71.9396210

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -22.6879253, 41.1678200, -21.6132240, 40.3477631, -63.0356712, 62.7810402
1: -25.5380325, 38.5620499, -24.3291454, 37.1506233, -62.6886559, 62.8911972
2: -26.1186962, 37.7568893, -24.9422073, 36.4105797, -62.5292740, 62.6990852
3: -31.2614918, 44.5497284, -29.9352512, 43.0429916, -74.3044739, 74.4849777
4: -29.5164909, 42.1602592, -28.1801186, 40.6411896, -70.1576843, 70.3403778

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -22.6879253, 41.1678200, -21.9912109, 40.9604340, -63.6483421, 63.1590309
1: -25.5380325, 38.5620499, -24.7420311, 37.7420349, -63.2800674, 63.3040810
2: -26.1186962, 37.7568893, -25.3736916, 36.9945221, -63.1132202, 63.1305809
3: -31.2614918, 44.5497284, -30.4472237, 43.7183762, -74.9798660, 74.9969482
4: -29.5164909, 42.1602592, -28.6507778, 41.3070107, -70.8235016, 70.8110352

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -22.8221550, 41.3871040, -21.6132240, 40.3477631, -63.1699104, 63.0003242
1: -25.6904716, 38.7174873, -24.3291454, 37.1506233, -62.8410835, 63.0466309
2: -26.2676792, 37.9167290, -24.9422073, 36.4105797, -62.6782494, 62.8589363
3: -31.4501839, 44.7205887, -29.9352512, 43.0429916, -74.4931793, 74.6558380
4: -29.6572666, 42.3763924, -28.1801186, 40.6411896, -70.2984543, 70.5565033

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -22.8221550, 41.3871040, -21.9912109, 40.9604340, -63.7825813, 63.3783150
1: -25.6904716, 38.7174873, -24.7420311, 37.7420349, -63.4325027, 63.4595184
2: -26.2676792, 37.9167290, -25.3736916, 36.9945221, -63.2621994, 63.2904205
3: -31.4501839, 44.7205887, -30.4472237, 43.7183762, -75.1685638, 75.1678162
4: -29.6572666, 42.3763924, -28.6507778, 41.3070107, -70.9642792, 71.0271683

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -22.6879253, 41.1678200, -26.9941216, 49.0133781, -71.7012787, 68.1619263
1: -25.5380325, 38.5620499, -30.3334293, 45.4923515, -71.0303802, 68.8954773
2: -26.1186962, 37.7568893, -31.0560799, 44.4996567, -70.6183548, 68.8129654
3: -31.2614918, 44.5497284, -37.1796837, 52.8559036, -84.1173935, 81.7294083
4: -29.5164909, 42.1602592, -35.0107651, 49.9743805, -79.4908752, 77.1710205

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -22.6879253, 41.1678200, -27.3528118, 49.5796242, -72.2675400, 68.5206299
1: -25.5380325, 38.5620499, -30.7202301, 46.0297432, -71.5677795, 69.2822800
2: -26.1186962, 37.7568893, -31.4665699, 45.0220184, -71.1407166, 69.2234573
3: -31.2614918, 44.5497284, -37.6436577, 53.4694099, -84.7308807, 82.1933823
4: -29.5164909, 42.1602592, -35.4463387, 50.5703735, -80.0868683, 77.6065979

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -22.8221550, 41.3871040, -26.9941216, 49.0133781, -71.8355103, 68.3812180
1: -25.6904716, 38.7174873, -30.3334293, 45.4923515, -71.1828156, 69.0509033
2: -26.2676792, 37.9167290, -31.0560799, 44.4996567, -70.7673340, 68.9728088
3: -31.4501839, 44.7205887, -37.1796837, 52.8559036, -84.3060913, 81.9002686
4: -29.6572666, 42.3763924, -35.0107651, 49.9743805, -79.6316376, 77.3871613

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -22.8221550, 41.3871040, -27.3528118, 49.5796242, -72.4017715, 68.7399139
1: -25.6904716, 38.7174873, -30.7202301, 46.0297432, -71.7202148, 69.4377136
2: -26.2676792, 37.9167290, -31.4665699, 45.0220184, -71.2896957, 69.3833008
3: -31.4501839, 44.7205887, -37.6436577, 53.4694099, -84.9195938, 82.3642426
4: -29.6572666, 42.3763924, -35.4463387, 50.5703735, -80.2276306, 77.8227310

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A1_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -27.1459351, 49.4174004, -17.2079239, 32.4127541, -59.5586891, 66.6253204
1: -30.4976730, 45.8090210, -19.3610115, 29.9016571, -60.3993301, 65.1700134
2: -31.2257614, 44.8032074, -19.8891678, 29.3916550, -60.6174164, 64.6923752
3: -37.3664398, 53.2030945, -23.7634315, 34.3966904, -71.7631302, 76.9665222
4: -35.1837616, 50.2943001, -22.4781494, 32.5509262, -67.7346878, 72.7724457

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -27.1459351, 49.4174004, -17.3456173, 32.5988235, -59.7447586, 66.7630081
1: -30.4976730, 45.8090210, -19.5256310, 30.0672913, -60.5649643, 65.3346405
2: -31.2257614, 44.8032074, -20.0441341, 29.5605125, -60.7862740, 64.8473434
3: -37.3664398, 53.2030945, -23.9777222, 34.5819740, -71.9484100, 77.1808167
4: -35.1837616, 50.2943001, -22.6333179, 32.7782593, -67.9620209, 72.9276047

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A2_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -27.5079098, 49.9935341, -17.2079239, 32.4127541, -59.9206619, 67.2014618
1: -30.8884201, 46.3548164, -19.3610115, 29.9016571, -60.7900772, 65.7158203
2: -31.6402645, 45.3322067, -19.8891678, 29.3916550, -61.0319214, 65.2213745
3: -37.8341980, 53.8254700, -23.7634315, 34.3966904, -72.2308884, 77.5888977
4: -35.6236992, 50.8980179, -22.4781494, 32.5509262, -68.1746216, 73.3761597

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A2_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -27.5079098, 49.9935341, -17.3456173, 32.5988235, -60.1067352, 67.3391266
1: -30.8884201, 46.3548164, -19.5256310, 30.0672913, -60.9557114, 65.8804398
2: -31.6402645, 45.3322067, -20.0441341, 29.5605125, -61.2007751, 65.3763428
3: -37.8341980, 53.8254700, -23.9777222, 34.5819740, -72.4161682, 77.8031921
4: -35.6236992, 50.8980179, -22.6333179, 32.7782593, -68.4019623, 73.5313339

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A2_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -27.1459351, 49.4174004, -22.6010666, 41.0435104, -68.1894226, 72.0184555
1: -30.4976730, 45.8090210, -25.4470558, 38.4601746, -68.9578476, 71.2560730
2: -31.2257614, 44.8032074, -26.0237408, 37.6582336, -68.8839874, 70.8269424
3: -37.3664398, 53.2030945, -31.1604176, 44.4320755, -81.7985001, 84.3635101
4: -35.1837616, 50.2943001, -29.4234562, 42.0567856, -77.2405472, 79.7177582

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A2_A2_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -27.5079098, 49.9935341, -22.6010666, 41.0435104, -68.5513992, 72.5945816
1: -30.8884201, 46.3548164, -25.4470558, 38.4601746, -69.3485947, 71.8018723
2: -31.6402645, 45.3322067, -26.0237408, 37.6582336, -69.2984924, 71.3559494
3: -37.8341980, 53.8254700, -31.1604176, 44.4320755, -82.2662735, 84.9858856
4: -35.6236992, 50.8980179, -29.4234562, 42.0567856, -77.6804810, 80.3214722

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A2_A2_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -27.1459351, 49.4174004, -22.7344017, 41.2614822, -68.4074173, 72.1518021
1: -30.4976730, 45.8090210, -25.5986557, 38.6146011, -69.1122742, 71.4076767
2: -31.2257614, 44.8032074, -26.1717892, 37.8171082, -69.0428696, 70.9749985
3: -37.3664398, 53.2030945, -31.3480244, 44.6017151, -81.9681473, 84.5511169
4: -35.1837616, 50.2943001, -29.5632267, 42.2719345, -77.4556961, 79.8575211

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A2_A2_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -27.5079098, 49.9935341, -22.7344017, 41.2614822, -68.7693939, 72.7279205
1: -30.8884201, 46.3548164, -25.5986557, 38.6146011, -69.5030212, 71.9534607
2: -31.6402645, 45.3322067, -26.1717892, 37.8171082, -69.4573746, 71.5039978
3: -37.8341980, 53.8254700, -31.3480244, 44.6017151, -82.4359131, 85.1734924
4: -35.6236992, 50.8980179, -29.5632267, 42.2719345, -77.8956299, 80.4612350

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A2_A2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -24.9989815, 45.6509018, -19.4801235, 36.9443207, -61.9433022, 65.1310272
1: -28.0598755, 42.1997566, -21.8982525, 33.7243996, -61.7842712, 64.0980072
2: -28.7609997, 41.3211327, -22.4941502, 33.1094513, -61.8704529, 63.8152695
3: -34.3129692, 48.9402542, -26.9263992, 38.9688416, -73.2818069, 75.8666534
4: -32.3584633, 46.2080040, -25.3940353, 36.7423363, -69.1007996, 71.6020355

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1137219, upper bound: 54.1793317
time: 0.71 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1273714, upper bound: 54.1859352
time: 0.65 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -27.1733932, 49.4113731, -19.4801235, 36.9443207, -64.1177139, 68.8914948
1: -30.5476780, 45.8119125, -21.8982525, 33.7243996, -64.2720795, 67.7101669
2: -31.2554703, 44.8115463, -22.4941502, 33.1094513, -64.3649216, 67.3056946
3: -37.4488449, 53.1881409, -26.9263992, 38.9688416, -76.4176865, 80.1145401
4: -35.2109261, 50.3438759, -25.3940353, 36.7423363, -71.9532547, 75.7378998

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1137219, upper bound: 54.1793317
time: 0.63 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1273714, upper bound: 54.1859352
time: 0.75 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -24.9989815, 45.6509018, -21.6157990, 40.3355522, -65.3345261, 67.2667007
1: -28.0598755, 42.1997566, -24.3604546, 37.1097221, -65.1696014, 66.5602112
2: -28.7609997, 41.3211327, -24.9428616, 36.3737679, -65.1347656, 66.2639771
3: -34.3129692, 48.9402542, -29.9993553, 42.9932442, -77.3062134, 78.9395905
4: -32.3584633, 46.2080040, -28.1840897, 40.6615143, -73.0199738, 74.3920822

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B1_B2_A1_A1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1137219, upper bound: 54.1763869
time: 1.08 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A1_A2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1273714, upper bound: 54.1831708
time: 0.71 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -27.1733932, 49.4113731, -21.6157990, 40.3355522, -67.5089340, 71.0271759
1: -30.5476780, 45.8119125, -24.3604546, 37.1097221, -67.6574020, 70.1723633
2: -31.2554703, 44.8115463, -24.9428616, 36.3737679, -67.6292419, 69.7544098
3: -37.4488449, 53.1881409, -29.9993553, 42.9932442, -80.4420853, 83.1874924
4: -35.2109261, 50.3438759, -28.1840897, 40.6615143, -75.8724213, 78.5279541

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_A1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1137219, upper bound: 54.1763869
time: 0.51 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_A2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1273714, upper bound: 54.1831708
time: 0.66 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -24.9989815, 45.6509018, -24.8894806, 45.4667892, -70.4657593, 70.5403824
1: -28.0598755, 42.1997566, -27.9448166, 42.0506592, -70.1105347, 70.1445541
2: -28.7609997, 41.3211327, -28.6405411, 41.1779480, -69.9389496, 69.9616623
3: -34.3129692, 48.9402542, -34.1853867, 48.7700653, -83.0830307, 83.1256409
4: -32.3584633, 46.2080040, -32.2394066, 46.0572433, -78.4157104, 78.4474106

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1502495, upper bound: 54.1109822
time: 0.70 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -24.9989815, 45.6509018, -27.0217915, 49.0073776, -74.0063629, 72.6726913
1: -28.0598755, 42.1997566, -30.3840942, 45.4939613, -73.5538254, 72.5838470
2: -28.7609997, 41.3211327, -31.0853882, 44.5081253, -73.2691269, 72.4064941
3: -34.3129692, 48.9402542, -37.2629738, 52.8413620, -87.1543274, 86.2032089
4: -32.3584633, 46.2080040, -35.0376968, 50.0244102, -82.3828735, 81.2456970

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_A1

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1235675, upper bound: 54.1144192
time: 0.77 seconds

## Relational analysis of NS_A2_A2_B2_B2_A1_B2_A2

### Relational analysis result of NS_A2_A2_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1561819, upper bound: 54.1246349
time: 0.60 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -27.1733932, 49.4113731, -24.8894806, 45.4667892, -72.6401749, 74.3008575
1: -30.5476780, 45.8119125, -27.9448166, 42.0506592, -72.5983353, 73.7567215
2: -31.2554703, 44.8115463, -28.6405411, 41.1779480, -72.4334183, 73.4520874
3: -37.4488449, 53.1881409, -34.1853867, 48.7700653, -86.2189102, 87.3735275
4: -35.2109261, 50.3438759, -32.2394066, 46.0572433, -81.2681732, 82.5832825

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1064471, upper bound: 54.1064471
time: 0.83 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1203741, upper bound: 54.1203741
time: 0.73 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -27.1733932, 49.4113731, -27.0217915, 49.0073776, -76.1807709, 76.4331665
1: -30.5476780, 45.8119125, -30.3840942, 45.4939613, -76.0416412, 76.1960068
2: -31.2554703, 44.8115463, -31.0853882, 44.5081253, -75.7635956, 75.8969345
3: -37.4488449, 53.1881409, -37.2629738, 52.8413620, -90.2902069, 90.4511032
4: -35.2109261, 50.3438759, -35.0376968, 50.0244102, -85.2353287, 85.3815765

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1064471, upper bound: 54.1064471
time: 1.11 seconds

## Relational analysis of NS_A2_A2_B2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1203741, upper bound: 54.1203741
time: 0.68 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.28 seconds
NS_A1_A1_B1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.3779234, upper bound: 53.7968863
NS_A1_A1_B1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.3174578, upper bound: 53.3174578
NS_A1_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.7968863, upper bound: 53.4830774
NS_A1_A1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.3174578, upper bound: 53.4324896
NS_A1_A1_B1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.4830774, upper bound: 53.8185195
NS_A1_A1_B1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.4324896, upper bound: 53.3386869
NS_A1_A1_B1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.4830774, upper bound: 53.9147337
NS_A1_A1_B1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.4324896, upper bound: 53.4537588
NS_A1_A1_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0145101, upper bound: 53.9762009
NS_A1_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0729939, upper bound: 53.9872118
NS_A1_A1_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0145101, upper bound: 53.9762009
NS_A1_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0729939, upper bound: 53.9872118
NS_A1_A1_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0471852, upper bound: 53.9790208
NS_A1_A1_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0709575, upper bound: 53.9872291
NS_A1_A1_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0226416, upper bound: 53.9528134
NS_A1_A1_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0226416, upper bound: 53.9528134
NS_A1_A1_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0632378, upper bound: 53.9990815
NS_A1_A1_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0632378, upper bound: 54.0024346
NS_A1_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.7820183, upper bound: 53.9471489
NS_A1_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0615574, upper bound: 53.9698709
NS_A1_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.8999370, upper bound: 53.9759908
NS_A1_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.9842739, upper bound: 53.9829808
NS_A1_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.7896110, upper bound: 53.9474410
NS_A1_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0601265, upper bound: 53.9700306
NS_A1_A2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0630038, upper bound: 53.9758991
NS_A1_A2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0601266, upper bound: 53.9700306
NS_A1_A2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.7956376, upper bound: 53.9503084
NS_A1_A2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.9770613, upper bound: 53.9666365
NS_A1_A2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.9819010, upper bound: 53.9726152
NS_A1_A2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.9770613, upper bound: 53.9666365
NS_A1_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.9729870, upper bound: 53.9615425
NS_A1_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0554984, upper bound: 53.9701549
NS_A1_A2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0583985, upper bound: 53.9760185
NS_A1_A2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0554984, upper bound: 53.9701549
NS_A1_A2_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.2379381, upper bound: 54.2376635
NS_A1_A2_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.2379381, upper bound: 54.2376635
NS_A1_A2_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.2490936, upper bound: 54.2376635
NS_A1_A2_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.2490936, upper bound: 54.2376635
NS_A1_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.2452873, upper bound: 54.2456145
NS_A1_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.2456463, upper bound: 54.2455755
NS_A1_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.2452873, upper bound: 54.2456145
NS_A1_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.2456463, upper bound: 54.2455755
NS_A1_A2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.8763503, upper bound: 53.9673597
NS_A1_A2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1215487, upper bound: 54.1055263
NS_A1_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0362276, upper bound: 54.0855195
NS_A1_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.0362276, upper bound: 54.1440125
NS_A1_A2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1664991, upper bound: 54.1198930
NS_A1_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1664993, upper bound: 54.1198930
NS_A1_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.9605547, upper bound: 53.9565637
NS_A1_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -53.9453017, upper bound: 53.9228449
NS_A2_A2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1137219, upper bound: 54.1793317
NS_A2_A2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1273714, upper bound: 54.1859352
NS_A2_A2_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1137219, upper bound: 54.1793317
NS_A2_A2_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1273714, upper bound: 54.1859352
NS_A2_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1137219, upper bound: 54.1763869
NS_A2_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1273714, upper bound: 54.1831708
NS_A2_A2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1137219, upper bound: 54.1763869
NS_A2_A2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1273714, upper bound: 54.1831708
NS_A2_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1235675, upper bound: 54.1144192
NS_A2_A2_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1561819, upper bound: 54.1246349
NS_A2_A2_B2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1064471, upper bound: 54.1064471
NS_A2_A2_B2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1203741, upper bound: 54.1203741
NS_A2_A2_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1064471, upper bound: 54.1064471
NS_A2_A2_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.28
Output dim: 0, lower bound: -54.1203741, upper bound: 54.1203741

## BFS NS instance: NS_A1_A1_B1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -17.2079239, 32.4127541, -16.5884876, 31.3761501, -48.5840759, 49.0012360
1: -19.3610115, 29.9016571, -18.6643124, 28.9215679, -48.2825699, 48.5659637
2: -19.8891678, 29.3916550, -19.1935329, 28.4418259, -48.3309937, 48.5851898
3: -23.7634315, 34.3966904, -22.9113350, 33.2520561, -57.0154839, 57.3080254
4: -22.4781494, 32.5509262, -21.7176285, 31.4614983, -53.9396477, 54.2685547

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_A1_B1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3174578, upper bound: 53.3174578
time: 0.57 seconds

## Relational analysis of NS_A1_A1_B1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_A1_B1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3174578, upper bound: 53.3174578
time: 0.54 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -16.0735664, 30.4926338, -16.4342651, 31.8391838, -47.9127464, 46.9268951
1: -18.0788689, 28.0409260, -18.4533176, 29.4553871, -47.5342560, 46.4942322
2: -18.6049423, 27.5937386, -19.1077957, 28.8963013, -47.5012436, 46.7015343
3: -22.1799927, 32.2247009, -22.6051750, 34.0514908, -56.2314835, 54.8298759
4: -21.0285072, 30.4908714, -21.7474937, 31.7416763, -52.7701836, 52.2383652

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_A1_B1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3174578, upper bound: 53.3174578
time: 0.70 seconds

## Relational analysis of NS_A1_A1_B1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_A1_B1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3174578, upper bound: 53.3174578
time: 0.88 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -16.5884876, 31.3761501, -17.3456173, 32.5988235, -49.1873093, 48.7217674
1: -18.6643124, 28.9215679, -19.5256310, 30.0672913, -48.7316055, 48.4471931
2: -19.1935329, 28.4418259, -20.0441341, 29.5605125, -48.7540436, 48.4859581
3: -22.9113350, 33.2520561, -23.9777222, 34.5819740, -57.4933052, 57.2297783
4: -21.7176285, 31.4614983, -22.6333179, 32.7782593, -54.4958878, 54.0948181

Time for backsubstitution: 0.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_A1_B1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3386869, upper bound: 53.4324896
time: 0.81 seconds

## Relational analysis of NS_A1_A1_B1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_A1_B1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3386869, upper bound: 53.4324896
time: 0.78 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -16.4342651, 31.8391838, -16.2045097, 30.6650105, -47.0992661, 48.0436935
1: -18.4533176, 29.4553871, -18.2344341, 28.1944160, -46.6477280, 47.6898155
2: -19.1077957, 28.8963013, -18.7521515, 27.7514534, -46.8592377, 47.6484451
3: -22.6051750, 34.0514908, -22.3829823, 32.3954124, -55.0005836, 56.4344711
4: -21.7474937, 31.7416763, -21.1740704, 30.7019939, -52.4494858, 52.9157448

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_A1_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3386869, upper bound: 53.4324896
time: 0.57 seconds

## Relational analysis of NS_A1_A1_B1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_A1_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.3386869, upper bound: 53.4324896
time: 0.54 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -17.3456173, 32.5988235, -16.5884876, 31.3761501, -48.7217674, 49.1873093
1: -19.5256310, 30.0672913, -18.6643124, 28.9215679, -48.4471893, 48.7316055
2: -20.0441341, 29.5605125, -19.1935329, 28.4418259, -48.4859619, 48.7540436
3: -23.9777222, 34.5819740, -22.9113350, 33.2520561, -57.2297783, 57.4933090
4: -22.6333179, 32.7782593, -21.7176285, 31.4614983, -54.0948181, 54.4958878

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.05 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B1_A2_B1_B1_A1

### Relational analysis result of NS_A1_A1_B1_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4324896, upper bound: 53.3386869
time: 0.72 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2_B1_B1_A2

### Relational analysis result of NS_A1_A1_B1_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4324896, upper bound: 53.3386869
time: 0.56 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -16.2045097, 30.6650105, -16.4342651, 31.8391838, -48.0436935, 47.0992699
1: -18.2344341, 28.1944160, -18.4533176, 29.4553871, -47.6898155, 46.6477280
2: -18.7521515, 27.7514534, -19.1077957, 28.8963013, -47.6484489, 46.8592377
3: -22.3829823, 32.3954124, -22.6051750, 34.0514908, -56.4344711, 55.0005875
4: -21.1740704, 30.7019939, -21.7474937, 31.7416763, -52.9157448, 52.4494820

Time for backsubstitution: 0.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35

Time for candidate selection: 0.05 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B1_A2_B1_B2_A1

### Relational analysis result of NS_A1_A1_B1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4324896, upper bound: 53.3386869
time: 0.68 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2_B1_B2_A2

### Relational analysis result of NS_A1_A1_B1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4324896, upper bound: 53.3386869
time: 0.58 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -17.3456173, 32.5988235, -16.7108593, 31.5337410, -48.8793564, 49.3096848
1: -19.5256310, 30.0672913, -18.8120003, 29.0583382, -48.5839691, 48.8792801
2: -20.0441341, 29.5605125, -19.3308754, 28.5831833, -48.6273193, 48.8913879
3: -23.9777222, 34.5819740, -23.1052761, 33.4036255, -57.3813477, 57.6872482
4: -22.6333179, 32.7782593, -21.8513603, 31.6589775, -54.2922974, 54.6296196

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A1_B1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4537588, upper bound: 53.4537588
time: 0.89 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A1_B1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4537588, upper bound: 53.4537588
time: 1.14 seconds

## BFS NS instance: NS_A1_A1_B1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -16.2045097, 30.6650105, -16.6281986, 32.1600609, -48.3645706, 47.2931976
1: -18.2344341, 28.1944160, -18.6772747, 29.7974205, -48.0318527, 46.8716888
2: -18.7521515, 27.7514534, -19.3344898, 29.2237072, -47.9758415, 47.0859375
3: -22.3829823, 32.3954124, -22.8983631, 34.4550896, -56.8380737, 55.2937698
4: -21.1740704, 30.7019939, -22.0051689, 32.1175194, -53.2915840, 52.7071609

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A1_B1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4537588, upper bound: 53.4537588
time: 0.56 seconds

## Relational analysis of NS_A1_A1_B1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A1_B1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.4537588, upper bound: 53.4537588
time: 0.64 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -16.9287109, 31.9394436, -21.8661537, 40.8458481, -57.7745552, 53.8055954
1: -19.0467262, 29.4327621, -24.6239452, 37.5693855, -56.6161118, 54.0567055
2: -19.5714760, 28.9346294, -25.2314472, 36.8253403, -56.3968086, 54.1660767
3: -23.3712864, 33.8382339, -30.3098011, 43.5024376, -66.8737259, 64.1480331
4: -22.1158047, 32.0432358, -28.4925900, 41.1440392, -63.2598419, 60.5358276

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0560838, upper bound: 53.9804471
time: 0.59 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0560838, upper bound: 54.2096194
time: 0.68 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -17.0408573, 32.0815086, -22.0263748, 41.1121254, -58.1529846, 54.1078796
1: -19.1829624, 29.5541878, -24.7984657, 37.8397713, -57.0227356, 54.3526459
2: -19.6988220, 29.0600777, -25.4140110, 37.0871849, -56.7860031, 54.4740906
3: -23.5507507, 33.9842796, -30.5227909, 43.8176689, -67.3684082, 64.5070724
4: -22.2407513, 32.2225838, -28.6980667, 41.4341583, -63.6749115, 60.9206467

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0587672, upper bound: 53.9804471
time: 0.55 seconds

## Relational analysis of NS_A1_A1_B2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0587672, upper bound: 54.2602778
time: 0.62 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -16.9287109, 31.9394436, -20.4234734, 38.2742538, -55.2029648, 52.3629150
1: -19.0467262, 29.4327621, -22.9184647, 35.0434341, -54.0901604, 52.3512230
2: -19.5714760, 28.9346294, -23.5870457, 34.3698807, -53.9413528, 52.5216751
3: -23.3712864, 33.8382339, -28.1627884, 40.5659332, -63.9372177, 62.0010185
4: -22.1158047, 32.0432358, -26.6241360, 38.2799873, -60.3957901, 58.6673737

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A1_A1_B2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -17.0408573, 32.0815086, -20.6047363, 38.5830002, -55.6238518, 52.6862450
1: -19.1829624, 29.5541878, -23.1258659, 35.3523712, -54.5353279, 52.6800346
2: -19.6988220, 29.0600777, -23.7939873, 34.6715317, -54.3703537, 52.8540535
3: -23.5507507, 33.9842796, -28.4156456, 40.9272690, -64.4780121, 62.3999252
4: -22.2407513, 32.2225838, -26.8594589, 38.6202202, -60.8609695, 59.0820427

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A1_A1_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -16.3312263, 31.0369682, -20.4234734, 38.2742538, -54.6054802, 51.4604416
1: -18.3140697, 28.4707489, -22.9184647, 35.0434341, -53.3575020, 51.3892021
2: -18.9326515, 28.0173359, -23.5870457, 34.3698807, -53.3025322, 51.6043739
3: -22.4239197, 32.7802811, -28.1627884, 40.5659332, -62.9898529, 60.9430695
4: -21.4007607, 30.8737030, -26.6241360, 38.2799873, -59.6807404, 57.4978333

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

## BFS NS instance: NS_A1_A1_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -16.0499516, 30.5593204, -20.6047363, 38.5830002, -54.6329498, 51.1640549
1: -18.0038338, 27.9717197, -23.1258659, 35.3523712, -53.3562012, 51.0975800
2: -18.6115398, 27.5368252, -23.7939873, 34.6715317, -53.2830582, 51.3308067
3: -22.0404415, 32.1826096, -28.4156456, 40.9272690, -62.9677086, 60.5982437
4: -21.0241795, 30.3469429, -26.8594589, 38.6202202, -59.6444016, 57.2064018

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

## BFS NS instance: NS_A1_A1_B2_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -17.2079239, 32.4127541, -26.9132576, 49.0046349, -66.2125549, 59.3260002
1: -19.3610115, 29.9016571, -30.2366276, 45.4139519, -64.7749481, 60.1382828
2: -19.8891678, 29.3916550, -30.9580040, 44.4133453, -64.3025131, 60.3496475
3: -23.7634315, 34.3966904, -37.0427322, 52.7356339, -76.4990540, 71.4394226
4: -22.4781494, 32.5509262, -34.8777657, 49.8586960, -72.3368454, 67.4286957

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B2_B2_B1_B1_A1_A1

### Relational analysis result of NS_A1_A1_B2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0226416, upper bound: 53.9528134
time: 0.62 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_B1_A1_A2

### Relational analysis result of NS_A1_A1_B2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0226416, upper bound: 53.9528134
time: 0.67 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -17.3456173, 32.5988235, -26.9132576, 49.0046349, -66.3502350, 59.5120811
1: -19.5256310, 30.0672913, -30.2366276, 45.4139519, -64.9395676, 60.3039169
2: -20.0441341, 29.5605125, -30.9580040, 44.4133453, -64.4574814, 60.5185127
3: -23.9777222, 34.5819740, -37.0427322, 52.7356339, -76.7133560, 71.6247101
4: -22.6333179, 32.7782593, -34.8777657, 49.8586960, -72.4920120, 67.6560211

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B2_B2_B1_B1_A2_A1

### Relational analysis result of NS_A1_A1_B2_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0226416, upper bound: 53.9528134
time: 0.55 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_B1_A2_A2

### Relational analysis result of NS_A1_A1_B2_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0226416, upper bound: 53.9528134
time: 0.72 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -17.2079239, 32.4127541, -27.2705288, 49.5736427, -66.7815704, 59.6832695
1: -19.3610115, 29.9016571, -30.6212692, 45.9507065, -65.3117065, 60.5229263
2: -19.8891678, 29.3916550, -31.3666096, 44.9417458, -64.8309174, 60.7582512
3: -23.7634315, 34.3966904, -37.5039520, 53.3472824, -77.1106873, 71.9006424
4: -22.4781494, 32.5509262, -35.3105354, 50.4530678, -72.9312134, 67.8614655

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B2_B2_B1_B2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0226416, upper bound: 53.9990814
time: 0.89 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_B2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0226416, upper bound: 53.9990814
time: 0.64 seconds

## BFS NS instance: NS_A1_A1_B2_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -17.3456173, 32.5988235, -27.2705288, 49.5736427, -66.9192429, 59.8693542
1: -19.5256310, 30.0672913, -30.6212692, 45.9507065, -65.4763336, 60.6885605
2: -20.0441341, 29.5605125, -31.3666096, 44.9417458, -64.9858780, 60.9271164
3: -23.9777222, 34.5819740, -37.5039520, 53.3472824, -77.3249969, 72.0859222
4: -22.6333179, 32.7782593, -35.3105354, 50.4530678, -73.0863724, 68.0887909

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_B2_B2_B1_B2_A2_A1

### Relational analysis result of NS_A1_A1_B2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0226416, upper bound: 53.9879709
time: 0.50 seconds

## Relational analysis of NS_A1_A1_B2_B2_B1_B2_A2_A2

### Relational analysis result of NS_A1_A1_B2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0226416, upper bound: 53.9879709
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -18.2594929, 34.4946442, -17.0639935, 32.1195335, -50.3790245, 51.5586395
1: -20.5320511, 31.7705765, -19.2093735, 29.6289711, -50.1610222, 50.9799500
2: -21.1102276, 31.1934090, -19.7222977, 29.1342278, -50.2444534, 50.9157028
3: -25.2330132, 36.6993637, -23.5871830, 34.0640526, -59.2970657, 60.2865372
4: -23.9054470, 34.5767250, -22.2820511, 32.2885132, -56.1939621, 56.8587761

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_A1_B1

### Relational analysis result of NS_A1_A2_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7909493, upper bound: 54.0387754
time: 0.51 seconds

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_A1_B2

### Relational analysis result of NS_A1_A2_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7909493, upper bound: 54.0405295
time: 0.66 seconds

## BFS NS instance: NS_A1_A2_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -18.7372608, 35.2431030, -17.1906815, 32.3424530, -51.0797119, 52.4337845
1: -21.0704937, 32.5190468, -19.3532696, 29.8395538, -50.9100494, 51.8722992
2: -21.6533165, 31.9310799, -19.8666916, 29.3375969, -50.9909134, 51.7977600
3: -25.8942871, 37.5617523, -23.7667923, 34.3104286, -60.2047157, 61.3285446
4: -24.5054283, 35.4176102, -22.4430447, 32.5233994, -57.0288239, 57.8606567

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2093663, upper bound: 54.1616889
time: 0.88 seconds

## Relational analysis of NS_A1_A2_B1_A1_A1_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2093663, upper bound: 54.2552934
time: 1.08 seconds

## BFS NS instance: NS_A1_A2_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -17.6346836, 33.4764709, -22.6286945, 41.0853386, -58.7200241, 56.1051636
1: -19.8626423, 30.7618446, -25.4894161, 38.4517670, -58.3143997, 56.2512550
2: -20.4043922, 30.2237129, -26.0497303, 37.6586952, -58.0630722, 56.2734451
3: -24.4237328, 35.4744835, -31.2145634, 44.4052925, -68.8290100, 66.6890488
4: -23.1036224, 33.5145988, -29.4317265, 42.0964317, -65.2000580, 62.9463196

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_A1_A2_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8457636, upper bound: 53.9703905
time: 1.02 seconds

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_A1_A2_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8917099, upper bound: 53.9669941
time: 0.70 seconds

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A1_B2

### Relational analysis result of NS_A1_A2_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8844878, upper bound: 53.9608243
time: 0.62 seconds

## BFS NS instance: NS_A1_A2_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -18.2763691, 34.4988136, -21.9594727, 39.9534225, -58.2297897, 56.4582787
1: -20.5797901, 31.6758766, -24.7381535, 37.3512878, -57.9310684, 56.4140282
2: -21.1305180, 31.1160507, -25.2844086, 36.5963554, -57.7268639, 56.4004555
3: -25.2461052, 36.5167770, -30.2861462, 43.0969009, -68.3430023, 66.8029251
4: -23.9087524, 34.5566025, -28.5658512, 40.8812866, -64.7900391, 63.1224518

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_A2_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0321533, upper bound: 53.9794783
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A2_B1

### Relational analysis result of NS_A1_A2_B1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0567064, upper bound: 53.9742610
time: 0.55 seconds

## Relational analysis of NS_A1_A2_B1_A1_A1_B2_A2_B2

### Relational analysis result of NS_A1_A2_B1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0535186, upper bound: 53.9683974
time: 0.60 seconds

## BFS NS instance: NS_A1_A2_B1_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -19.9811535, 37.2443542, -15.9560719, 30.2128468, -50.1940002, 53.2004242
1: -22.4938202, 34.5990372, -17.9861526, 27.8775349, -50.3713531, 52.5851898
2: -23.0916672, 33.9181328, -18.4605026, 27.4238510, -50.5155106, 52.3786278
3: -27.5973644, 39.9768486, -22.0946636, 32.0054970, -59.6028481, 62.0714951
4: -26.1774139, 37.6664963, -20.9027195, 30.3448925, -56.5223083, 58.5692062

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7981281, upper bound: 54.0226138
time: 0.67 seconds

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7981281, upper bound: 54.0242108
time: 0.75 seconds

## BFS NS instance: NS_A1_A2_B1_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -20.5431213, 38.1284332, -16.0788250, 30.4293938, -50.9725151, 54.2072601
1: -23.1458492, 35.4681473, -18.1255379, 28.0806980, -51.2265472, 53.5936813
2: -23.7242355, 34.7695160, -18.6002522, 27.6212273, -51.3454628, 53.3697662
3: -28.4213028, 40.9844818, -22.2695885, 32.2428894, -60.6641922, 63.2540627
4: -26.8801613, 38.6938705, -21.0593300, 30.5727730, -57.4529266, 59.7532005

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1405878, upper bound: 54.1663544
time: 0.69 seconds

## Relational analysis of NS_A1_A2_B1_A1_A2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1405878, upper bound: 54.2114781
time: 0.63 seconds

## BFS NS instance: NS_A1_A2_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -20.5377445, 38.1393852, -20.3927879, 36.9931488, -57.5308914, 58.5321732
1: -23.1376629, 35.4562836, -22.9818230, 34.8790016, -58.0166512, 58.4380951
2: -23.7197495, 34.7573471, -23.4940701, 34.1815872, -57.9013290, 58.2514114
3: -28.4056892, 40.9692154, -28.1497002, 40.2158813, -68.6215668, 69.1189117
4: -26.8731117, 38.6734428, -26.6149807, 38.0689545, -64.9420624, 65.2884216

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7896110, upper bound: 53.9474410
time: 1.00 seconds

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7896110, upper bound: 53.9700306
time: 1.19 seconds

## BFS NS instance: NS_A1_A2_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -20.7641983, 38.5226555, -20.5425091, 37.2043304, -57.9685287, 59.0651512
1: -23.3978901, 35.8334427, -23.1527023, 35.0477905, -58.4456787, 58.9861450
2: -23.9760513, 35.1220207, -23.6604462, 34.3554153, -58.3314629, 58.7824669
3: -28.7319565, 41.4105492, -28.3608646, 40.4049835, -69.1369324, 69.7714157
4: -27.1647415, 39.1019020, -26.7745686, 38.3024635, -65.4672012, 65.8764572

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7896110, upper bound: 53.9474410
time: 0.68 seconds

## Relational analysis of NS_A1_A2_B1_A1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7896110, upper bound: 53.9700306
time: 0.69 seconds

## BFS NS instance: NS_A1_A2_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -21.5177326, 39.5863647, -16.6554890, 31.3871956, -52.9049225, 56.2418518
1: -24.1012955, 36.2955933, -18.7490883, 28.8750801, -52.9763641, 55.0446815
2: -24.7689857, 35.6702576, -19.2545891, 28.4133606, -53.1823425, 54.9248390
3: -29.5492287, 41.9749603, -23.0109482, 33.1787643, -62.7279854, 64.9859085
4: -27.6464901, 39.8186722, -21.7314129, 31.4852810, -59.1317673, 61.5500793

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A2_A1_B1_A1_B1

### Relational analysis result of NS_A1_A2_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8069084, upper bound: 54.0452482
time: 1.15 seconds

## Relational analysis of NS_A1_A2_B1_A2_A1_B1_A1_B2

### Relational analysis result of NS_A1_A2_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8069084, upper bound: 54.0470126
time: 0.56 seconds

## BFS NS instance: NS_A1_A2_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -21.8935757, 40.1844482, -16.7706890, 31.5923920, -53.4859695, 56.9551353
1: -24.5278053, 36.8838120, -18.8798580, 29.0675468, -53.5953522, 55.7636642
2: -25.2002926, 36.2418594, -19.3858051, 28.5994987, -53.7997894, 55.6276588
3: -30.0724201, 42.6608086, -23.1749821, 33.4037971, -63.4762154, 65.8357773
4: -28.1208591, 40.4774513, -21.8779640, 31.7004433, -59.8213043, 62.3553925

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A2_A1_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0353905, upper bound: 54.0975702
time: 0.87 seconds

## Relational analysis of NS_A1_A2_B1_A2_A1_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0353905, upper bound: 54.1569055
time: 0.83 seconds

## BFS NS instance: NS_A1_A2_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -21.9979630, 40.3879738, -21.5142746, 39.0813522, -61.0793152, 61.9022484
1: -24.6501980, 37.0597687, -24.2176666, 36.5150185, -61.1652107, 61.2774353
2: -25.3174782, 36.4139900, -24.7760887, 35.8018112, -61.1192894, 61.1900520
3: -30.2396545, 42.8625450, -29.6227779, 42.1228218, -72.3624725, 72.4853134
4: -28.2575779, 40.6974602, -27.9502010, 39.9431496, -68.2007294, 68.6476593

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A2_A1_B2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7956376, upper bound: 53.9503084
time: 0.96 seconds

## Relational analysis of NS_A1_A2_B1_A2_A1_B2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7956376, upper bound: 53.9666365
time: 0.86 seconds

## BFS NS instance: NS_A1_A2_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -22.1715717, 40.6826172, -21.7160587, 39.4203568, -61.5919266, 62.3986740
1: -24.8480091, 37.3488617, -24.4478703, 36.7910614, -61.6390686, 61.7967262
2: -25.5159740, 36.6913071, -25.0017624, 36.0772858, -61.5932617, 61.6930504
3: -30.4759750, 43.2018280, -29.9095020, 42.4352341, -72.9112015, 73.1113281
4: -28.4807472, 41.0084763, -28.1812973, 40.2889862, -68.7697296, 69.1897736

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A2_A1_B2_B2_A1

### Relational analysis result of NS_A1_A2_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7956376, upper bound: 53.9503084
time: 0.59 seconds

## Relational analysis of NS_A1_A2_B1_A2_A1_B2_B2_A2

### Relational analysis result of NS_A1_A2_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.7956376, upper bound: 53.9666365
time: 0.84 seconds

## BFS NS instance: NS_A1_A2_B1_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -21.3531857, 39.4255180, -15.5832958, 29.5360947, -50.8892822, 55.0088120
1: -23.9711761, 36.3942566, -17.5662613, 27.1941757, -51.1653481, 53.9605103
2: -24.6254883, 35.7032890, -18.0347176, 26.7685146, -51.3940048, 53.7379990
3: -29.3603554, 42.1154175, -21.5667915, 31.2085152, -60.5688705, 63.6822090
4: -27.6802502, 39.7563477, -20.4008961, 29.6103668, -57.2906189, 60.1572418

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A2_A2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B1_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9843637, upper bound: 54.0571359
time: 0.96 seconds

## Relational analysis of NS_A1_A2_B1_A2_A2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B1_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9843637, upper bound: 54.0589337
time: 0.70 seconds

## BFS NS instance: NS_A1_A2_B1_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -21.8781147, 40.2656288, -15.6944838, 29.7356110, -51.6137238, 55.9601097
1: -24.5684052, 37.2252998, -17.6921425, 27.3798275, -51.9482346, 54.9174423
2: -25.2189388, 36.5137901, -18.1617489, 26.9482918, -52.1672287, 54.6755371
3: -30.1138496, 43.1014023, -21.7249870, 31.4250374, -61.5388870, 64.8263855
4: -28.3416023, 40.7217026, -20.5435562, 29.8177338, -58.1593285, 61.2652588

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A2_A2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1249623, upper bound: 54.1422514
time: 0.54 seconds

## Relational analysis of NS_A1_A2_B1_A2_A2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1249623, upper bound: 54.1950293
time: 0.99 seconds

## BFS NS instance: NS_A1_A2_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -21.9744453, 40.4521866, -19.3243580, 35.0266380, -57.0010834, 59.7765427
1: -24.6770344, 37.3862991, -21.7734909, 32.9614830, -57.6385193, 59.1597862
2: -25.3296566, 36.6638832, -22.2742100, 32.3483276, -57.6779861, 58.9380951
3: -30.2410336, 43.2855835, -26.6503830, 37.9885292, -68.2295609, 69.9359665
4: -28.4697380, 40.8862724, -25.2028160, 35.9682312, -64.4379654, 66.0890884

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B1_A2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A2_A2_B2_B1_A1

### Relational analysis result of NS_A1_A2_B1_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9729870, upper bound: 53.9615425
time: 0.70 seconds

## Relational analysis of NS_A1_A2_B1_A2_A2_B2_B1_A2

### Relational analysis result of NS_A1_A2_B1_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9729870, upper bound: 53.9701549
time: 0.71 seconds

## BFS NS instance: NS_A1_A2_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -22.1560783, 40.7551384, -19.5620174, 35.4325485, -57.5886269, 60.3171501
1: -24.8856697, 37.6878090, -22.0424862, 33.3011398, -58.1868057, 59.7302933
2: -25.5352058, 36.9570618, -22.5412598, 32.6851692, -58.2203598, 59.4983177
3: -30.5033989, 43.6392708, -26.9838696, 38.3797417, -68.8831406, 70.6231384
4: -28.7018166, 41.2328529, -25.4840298, 36.3786354, -65.0804520, 66.7168732

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_A2_A2_B2_B2_A1

### Relational analysis result of NS_A1_A2_B1_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9729870, upper bound: 53.9615425
time: 0.72 seconds

## Relational analysis of NS_A1_A2_B1_A2_A2_B2_B2_A2

### Relational analysis result of NS_A1_A2_B1_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9729870, upper bound: 53.9701549
time: 0.68 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -19.9709053, 37.6195297, -18.0096893, 34.5045395, -54.4754448, 55.6292114
1: -22.4919643, 34.4881020, -20.2644501, 31.3356285, -53.8275909, 54.7525520
2: -23.0672455, 33.8458862, -20.8240395, 30.7965603, -53.8638077, 54.6699219
3: -27.6811600, 39.8655052, -24.9127598, 36.1377792, -63.8189392, 64.7782669
4: -26.0491295, 37.6885681, -23.5042210, 34.1120758, -60.1612053, 61.1927834

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2379381, upper bound: 54.2376635
time: 0.74 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2379381, upper bound: 54.2376635
time: 0.58 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -19.9709053, 37.6195297, -18.5929279, 35.4050598, -55.3759651, 56.2124557
1: -22.4919643, 34.4881020, -20.9081707, 32.2002640, -54.6922302, 55.3962708
2: -23.0672455, 33.8458862, -21.4890232, 31.6307678, -54.6980133, 55.3349037
3: -27.6811600, 39.8655052, -25.6358967, 37.1361847, -64.8173447, 65.5013962
4: -26.0491295, 37.6885681, -24.2581673, 35.0568504, -61.1059799, 61.9467354

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B2_B1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2379381, upper bound: 54.2376635
time: 0.72 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2379381, upper bound: 54.2376635
time: 0.63 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -20.7535648, 38.9688454, -20.1621380, 37.9425201, -58.6960831, 59.1309814
1: -23.3915901, 35.7409630, -22.7463379, 34.7554359, -58.1470261, 58.4872932
2: -23.9654617, 35.0568199, -23.2905464, 34.0983238, -58.0637817, 58.3473663
3: -28.8036175, 41.3473244, -28.0199013, 40.2061424, -69.0097580, 69.3672256
4: -27.0659561, 39.1405182, -26.3197308, 38.0713654, -65.1373138, 65.4602509

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2379381, upper bound: 54.2376635
time: 0.58 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2379381, upper bound: 54.2376635
time: 0.60 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -20.7535648, 38.9688454, -20.9512825, 39.1883583, -59.9419212, 59.9201279
1: -23.3915901, 35.7409630, -23.6223164, 35.9247818, -59.3163719, 59.3632698
2: -23.9654617, 35.0568199, -24.1859741, 35.2365913, -59.2020531, 59.2427940
3: -28.8036175, 41.3473244, -29.0305367, 41.5495491, -70.3531647, 70.3778610
4: -27.0659561, 39.1405182, -27.3181362, 39.3811989, -66.4471512, 66.4586563

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2379381, upper bound: 54.2376635
time: 0.61 seconds

## Relational analysis of NS_A1_A2_B2_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2379381, upper bound: 54.2376635
time: 0.55 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -18.6505756, 35.3713188, -19.7382240, 37.2840958, -55.9346619, 55.1095428
1: -21.0479298, 32.0889587, -22.2689476, 34.0633774, -55.1113052, 54.3579063
2: -21.5534916, 31.5764656, -22.8086624, 33.4443474, -54.9978409, 54.3851280
3: -25.8689842, 36.9811134, -27.4345913, 39.3556671, -65.2246552, 64.4157028
4: -24.3483124, 35.1917801, -25.7743301, 37.3207817, -61.6690903, 60.9661102

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2457871, upper bound: 54.2553831
time: 0.54 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2457871, upper bound: 54.2566935
time: 0.51 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -25.1167068, 45.9919701, -20.2739372, 38.1013718, -63.2180672, 66.2659073
1: -28.3256130, 42.3169441, -22.8562202, 34.9313622, -63.2569733, 65.1731644
2: -28.9228325, 41.4970016, -23.4192066, 34.2780228, -63.2008553, 64.9162064
3: -34.8510017, 48.8821526, -28.1498909, 40.3865738, -75.2375717, 77.0320282
4: -32.4884186, 46.6386871, -26.4399033, 38.2647095, -70.7531281, 73.0785904

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2461460, upper bound: 54.2553831
time: 0.72 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2461460, upper bound: 54.2566935
time: 0.62 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -18.6505756, 35.3713188, -20.4478760, 38.4054146, -57.0559845, 55.8191948
1: -21.0479298, 32.0889587, -23.0556812, 35.1232033, -56.1711349, 55.1446381
2: -21.5534916, 31.5764656, -23.6150513, 34.4752426, -56.0287323, 55.1915169
3: -25.8689842, 36.9811134, -28.3372707, 40.5794945, -66.4484711, 65.3183746
4: -24.3483124, 35.1917801, -26.6816387, 38.4946785, -62.8429909, 61.8734169

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2452873, upper bound: 54.2452873
time: 0.55 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2452873, upper bound: 54.2455755
time: 0.55 seconds

## BFS NS instance: NS_A1_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -25.1167068, 45.9919701, -20.8734493, 39.0232086, -64.1399155, 66.8654099
1: -28.3256130, 42.3169441, -23.5280571, 35.7789764, -64.1045914, 65.8450012
2: -28.9228325, 41.4970016, -24.0988865, 35.1076698, -64.0305023, 65.5958862
3: -34.8510017, 48.8821526, -28.9136734, 41.3458214, -76.1968231, 77.7958221
4: -32.4884186, 46.6386871, -27.2008305, 39.2318878, -71.7203064, 73.8395157

Time for backsubstitution: 0.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.2456463, upper bound: 54.2452873
time: 1.36 seconds

## Relational analysis of NS_A1_A2_B2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_A2_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0401668, upper bound: 54.2455755
time: 0.69 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -19.4903393, 36.8607979, -23.6614857, 43.2902374, -62.7805786, 60.5222778
1: -21.9923439, 33.6487503, -26.6263885, 39.9922523, -61.9845772, 60.2751312
2: -22.5264359, 33.0436897, -27.2357330, 39.2074242, -61.7338486, 60.2794037
3: -27.0915680, 38.8707809, -32.6592293, 46.2883720, -73.3799286, 71.5299988
4: -25.4556713, 36.8640060, -30.6805515, 43.9353943, -69.3910522, 67.5445557

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8440330, upper bound: 53.9673597
time: 1.10 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B1_B1_A2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.8440330, upper bound: 53.9673597
time: 1.52 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -19.6516113, 37.1267319, -24.0044708, 43.8054733, -63.4570847, 61.1311951
1: -22.1681080, 33.9189453, -26.9908524, 40.4479637, -62.6160698, 60.9097900
2: -22.7101269, 33.3050003, -27.6257915, 39.6623611, -62.3724861, 60.9307861
3: -27.3081188, 39.1869888, -33.1033134, 46.8393288, -74.1474457, 72.2902908
4: -25.6623764, 37.1555176, -31.0942192, 44.4795380, -70.1419144, 68.2497406

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B2_A1_B1_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0401159, upper bound: 54.0684349
time: 0.68 seconds

## Relational analysis of NS_A1_A2_B2_B2_A1_B1_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.0401159, upper bound: 54.1055265
time: 0.71 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -17.8160114, 34.0215607, -31.3419724, 56.5433960, -74.3593979, 65.3635330
1: -20.1211739, 30.8008003, -35.2866096, 52.5778503, -72.6990128, 66.0874100
2: -20.5978432, 30.3216705, -36.0381165, 51.3734932, -71.9713364, 66.3597794
3: -24.7908344, 35.4889793, -43.3227921, 60.9897804, -85.7806168, 78.8117523
4: -23.2771263, 33.7750626, -40.5259285, 57.9199066, -81.1970215, 74.3009796

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_A2_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -24.8240490, 45.6679764, -31.3419724, 56.5433960, -81.3674316, 77.0099487
1: -27.9966373, 42.0347786, -35.2866096, 52.5778503, -80.5744781, 77.3213806
2: -28.5971832, 41.1817169, -36.0381165, 51.3734932, -79.9706726, 77.2198334
3: -34.5088348, 48.6311493, -43.3227921, 60.9897804, -95.4985962, 91.9539185
4: -32.1483917, 46.2813225, -40.5259285, 57.9199066, -90.0682907, 86.8072433

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 40

## BFS NS instance: NS_A1_A2_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -18.5929279, 35.4050598, -23.7872734, 43.4955788, -62.0884972, 59.1923332
1: -20.9081707, 32.2002640, -26.6979084, 40.1541824, -61.0623550, 58.8981705
2: -21.4890232, 31.6307678, -27.3762989, 39.3467522, -60.8357735, 59.0070610
3: -25.6358967, 37.1361847, -32.6357689, 46.5428123, -72.1787033, 69.7719574
4: -24.2581673, 35.0568504, -30.8000050, 43.9359894, -68.1941452, 65.8568573

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1700078, upper bound: 54.1091309
time: 0.59 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1700078, upper bound: 54.1170340
time: 0.81 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -18.5929279, 35.4050598, -26.4535122, 48.1483879, -66.7413177, 61.8585739
1: -20.9081707, 32.2002640, -29.7406368, 44.5992050, -65.5073776, 61.9408836
2: -21.4890232, 31.6307678, -30.4283619, 43.6408653, -65.1298904, 62.0591278
3: -25.6358967, 37.1361847, -36.4532967, 51.7585182, -77.3944016, 73.5894775
4: -24.2581673, 35.0568504, -34.2767181, 49.0177383, -73.2758942, 69.3335724

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_A2_B2_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B2_A2_A1_B2_B1

### Relational analysis result of NS_A1_A2_B2_B2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9549775, upper bound: 54.1091309
time: 0.66 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_A1_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1700078, upper bound: 54.1170340
time: 0.70 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -20.9512825, 39.1883583, -26.7969837, 48.7744484, -69.7257156, 65.9853439
1: -23.6223164, 35.9247818, -30.1112366, 45.1614647, -68.7837830, 66.0360184
2: -24.1859741, 35.2365913, -30.8208256, 44.1752853, -68.3612595, 66.0574036
3: -29.0305367, 41.5495491, -36.8929901, 52.4046021, -81.4351349, 78.4425354
4: -27.3181362, 39.3811989, -34.7025490, 49.6211662, -76.9393005, 74.0837479

Time for backsubstitution: 0.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B1_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9453016, upper bound: 53.9228449
time: 0.62 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B1_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9453016, upper bound: 53.9228449
time: 0.68 seconds

## BFS NS instance: NS_A1_A2_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -20.1827927, 37.8689804, -25.0316982, 45.7600746, -65.9428711, 62.9006729
1: -22.7626076, 34.6402206, -28.0891056, 42.2439651, -65.0065536, 62.7293243
2: -23.3093891, 34.0071869, -28.8202438, 41.3737183, -64.6831055, 62.8274307
3: -27.9678421, 40.0422516, -34.3802948, 48.9190254, -76.8868484, 74.4225464
4: -26.3367558, 37.9633789, -32.4607544, 46.2781219, -72.6148682, 70.4241333

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_A2_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9453016, upper bound: 53.9228449
time: 0.72 seconds

## Relational analysis of NS_A1_A2_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_A2_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9453016, upper bound: 53.9228449
time: 0.82 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -21.7182884, 39.9148178, -18.3027992, 34.9557343, -56.6740227, 58.2176170
1: -24.4112740, 36.6826591, -20.5982208, 31.7524014, -56.1636734, 57.2808800
2: -25.0035496, 36.0107002, -21.1493454, 31.2120781, -56.2156296, 57.1600456
3: -29.8913231, 42.4065132, -25.3399696, 36.6314201, -66.5227432, 67.7464828
4: -28.1129322, 40.2179604, -23.8785839, 34.6144180, -62.7273483, 64.0965347

Time for backsubstitution: 0.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1323156, upper bound: 54.1945068
time: 0.88 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1323156, upper bound: 54.1958465
time: 0.58 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -28.3827515, 51.3638153, -18.7970486, 35.7294960, -64.1122437, 70.1608658
1: -31.9260483, 47.5119820, -21.1359081, 32.5490761, -64.4751205, 68.6478882
2: -32.6488037, 46.4751587, -21.7123966, 31.9728508, -64.6216583, 68.1875381
3: -39.1583519, 55.1777229, -25.9904900, 37.5815582, -76.7399139, 81.1682129
4: -36.6617470, 52.3159485, -24.4980526, 35.4578857, -72.1196289, 76.8140030

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1723155, upper bound: 54.2452719
time: 0.58 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1716101, upper bound: 54.2403009
time: 0.60 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -23.6792908, 43.2492867, -18.3027992, 34.9557343, -58.6350250, 61.5520782
1: -26.6681576, 39.9649696, -20.5982208, 31.7524014, -58.4205589, 60.5631714
2: -27.2509060, 39.1873512, -21.1493454, 31.2120781, -58.4629822, 60.3366966
3: -32.7342186, 46.2356606, -25.3399696, 36.6314201, -69.3656235, 71.5756302
4: -30.6890945, 43.9601936, -23.8785839, 34.6144180, -65.3035126, 67.8387680

Time for backsubstitution: 0.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A1_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1137201, upper bound: 54.1763453
time: 0.65 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A1_B2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1137201, upper bound: 54.1793317
time: 0.58 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -30.6565800, 55.3003387, -18.7970486, 35.7294960, -66.3860779, 74.0973816
1: -34.5245895, 51.4583435, -21.1359081, 32.5490761, -67.0736694, 72.5942383
2: -35.2512512, 50.2886353, -21.7123966, 31.9728508, -67.2241058, 72.0010147
3: -42.4021988, 59.6720428, -25.9904900, 37.5815582, -79.9837570, 85.6625366
4: -39.6480217, 56.6797676, -24.4980526, 35.4578857, -75.1059036, 81.1778183

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1257880, upper bound: 54.1763453
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1257881, upper bound: 54.1859352
time: 0.73 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -21.7182884, 39.9148178, -20.6888409, 38.7659187, -60.4842072, 60.6036606
1: -24.4112740, 36.6826591, -23.3301334, 35.5200081, -59.9312820, 60.0127945
2: -25.0035496, 36.0107002, -23.8820934, 34.8529854, -59.8565369, 59.8927879
3: -29.8913231, 42.4065132, -28.7433376, 41.1047173, -70.9960327, 71.1498489
4: -28.1129322, 40.2179604, -26.9765797, 38.9607620, -67.0736771, 67.1945419

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B1_B2_A1_A1_B1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1243831, upper bound: 54.1764285
time: 0.91 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A1_A1_B2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1243831, upper bound: 54.1764285
time: 0.69 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -28.3827515, 51.3638153, -20.9918365, 39.2206764, -67.6034241, 72.3556519
1: -31.9260483, 47.5119820, -23.6667500, 36.0564384, -67.9824829, 71.1787338
2: -32.6488037, 46.4751587, -24.2296429, 35.3577232, -68.0065308, 70.7047806
3: -39.1583519, 55.1777229, -29.1535378, 41.7501602, -80.9085083, 84.3312607
4: -36.6617470, 52.3159485, -27.3733215, 39.5229683, -76.1847153, 79.6892700

Time for backsubstitution: 0.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B1_B2_A1_A2_B1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1697270, upper bound: 54.2341620
time: 0.52 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A1_A2_B2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1697270, upper bound: 54.2341620
time: 0.58 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -23.6792908, 43.2492867, -20.6888409, 38.7659187, -62.4452095, 63.9381256
1: -26.6681576, 39.9649696, -23.3301334, 35.5200081, -62.1881599, 63.2950897
2: -27.2509060, 39.1873512, -23.8820934, 34.8529854, -62.1038857, 63.0694427
3: -32.7342186, 46.2356606, -28.7433376, 41.1047173, -73.8389130, 74.9789734
4: -30.6890945, 43.9601936, -26.9765797, 38.9607620, -69.6498566, 70.9367752

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_A1_A1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -53.9480797, upper bound: 53.8152173
time: 0.58 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_A1_A2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1106444, upper bound: 54.1733028
time: 1.26 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -30.6565800, 55.3003387, -20.9918365, 39.2206764, -69.8772583, 76.2921753
1: -34.5245895, 51.4583435, -23.6667500, 36.0564384, -70.5810242, 75.1250916
2: -35.2512512, 50.2886353, -24.2296429, 35.3577232, -70.6089783, 74.5182571
3: -42.4021988, 59.6720428, -29.1535378, 41.7501602, -84.1523590, 88.8255768
4: -39.6480217, 56.6797676, -27.3733215, 39.5229683, -79.1709900, 84.0530853

Time for backsubstitution: 0.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_A2_B1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1257881, upper bound: 54.1763453
time: 0.61 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -54.1257881, upper bound: 54.1831708
time: 0.60 seconds

## BFS NS instance: NS_A2_A2_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -21.7182884, 39.9148178, -25.8847637, 47.0278397, -68.7461243, 65.7995834
1: -24.4112740, 36.6826591, -29.1224136, 43.6017647, -68.0130386, 65.8050690
2: -25.0035496, 36.0107002, -29.7814522, 42.6866455, -67.6901932, 65.7921524
3: -29.8913231, 42.4065132, -35.7303314, 50.5833244, -80.4746475, 78.1368408
4: -28.1129322, 40.2179604, -33.5613823, 47.9584846, -76.0714188, 73.7793350

Time for backsubstitution: 0.81 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 2.21 + 417.90 = 420.11 seconds
