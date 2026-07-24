## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 7)
Time budget: 420 seconds
Split limit: 100
Threshold: 547.332881116455


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-131.2357788, 514.9561768, -131.2357788, 514.9561768, -646.1919556, 646.1919556)
1: (-85.4777908, 295.0576172, -85.4777908, 295.0576172, -380.5354004, 380.5354004)
2: (-46.5778465, 267.0964355, -46.5778465, 267.0964355, -313.6742859, 313.6742859)
3: (-62.6862679, 401.5845032, -62.6862679, 401.5845032, -464.2707825, 464.2707825)
4: (-84.1672974, 324.8345337, -84.1672974, 324.8345337, -409.0018311, 409.0018311)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.89 + 1.95 = 3.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -547.3383545, upper bound: 547.3383545

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366367, upper bound: 547.3364670
time: 0.76 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359296, upper bound: 547.3359296
time: 0.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.77 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 0, lower bound: -547.3366367, upper bound: 547.3364670
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.77
Output dim: 0, lower bound: -547.3359296, upper bound: 547.3359296

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -129.0341492, 506.1789551, -131.2357788, 514.9561768, -643.9902954, 637.4145508
1: -84.0021820, 289.9710388, -85.4777908, 295.0576172, -379.0597839, 375.4488220
2: -45.8047028, 262.5160522, -46.5778465, 267.0964355, -312.9011230, 309.0939026
3: -61.6502228, 394.5868225, -62.6862679, 401.5845032, -463.2347412, 457.2731018
4: -82.7363358, 319.2454529, -84.1672974, 324.8345337, -407.5708618, 403.4127502

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359296, upper bound: 547.3359296
time: 0.86 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359296, upper bound: 547.3359296
time: 0.85 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -185.2007751, 724.0216064, -126.8287354, 497.7982483, -682.9990234, 850.8503418
1: -120.3639984, 414.3753662, -82.6306229, 285.1797180, -405.5437012, 497.0059814
2: -65.6249084, 375.4895630, -45.0256424, 258.1870728, -323.8119812, 420.5151978
3: -88.0919418, 564.7359619, -60.5808945, 388.0799866, -476.1719360, 625.3168335
4: -118.0188751, 457.5017395, -81.3329620, 313.9377441, -431.9566040, 538.8346558

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359296, upper bound: 547.3359296
time: 0.82 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359296, upper bound: 547.3359296
time: 0.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.79 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.79
Output dim: 0, lower bound: -547.3359296, upper bound: 547.3359296
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.79
Output dim: 0, lower bound: -547.3359296, upper bound: 547.3359296
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.79
Output dim: 0, lower bound: -547.3359296, upper bound: 547.3359296
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.79
Output dim: 0, lower bound: -547.3359296, upper bound: 547.3359296

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -129.0341492, 506.1789551, -129.0341492, 506.1789551, -635.2128906, 635.2128906
1: -84.0021820, 289.9710388, -84.0021820, 289.9710388, -373.9731750, 373.9731750
2: -45.8047028, 262.5160522, -45.8047028, 262.5160522, -308.3207397, 308.3207397
3: -61.6502228, 394.5868225, -61.6502228, 394.5868225, -456.2370300, 456.2370300
4: -82.7363358, 319.2454529, -82.7363358, 319.2454529, -401.9817810, 401.9817810

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3365384, upper bound: 547.3362609
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366236, upper bound: 547.3363458
time: 0.93 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -129.0341492, 506.1789551, -185.2007751, 724.0216064, -853.0557251, 691.3796387
1: -84.0021820, 289.9710388, -120.3639984, 414.3753662, -498.3775635, 410.3350220
2: -45.8047028, 262.5160522, -65.6249084, 375.4895630, -421.2942505, 328.1409607
3: -61.6502228, 394.5868225, -88.0919418, 564.7359619, -626.3861694, 482.6787720
4: -82.7363358, 319.2454529, -118.0188751, 457.5017395, -540.2380981, 437.2642822

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3365384, upper bound: 547.3362609
time: 0.61 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366236, upper bound: 547.3363458
time: 0.73 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -185.2007751, 724.0216064, -129.0341492, 506.1789551, -691.3796387, 853.0557251
1: -120.3639984, 414.3753662, -84.0021820, 289.9710388, -410.3350220, 498.3775635
2: -65.6249084, 375.4895630, -45.8047028, 262.5160522, -328.1409607, 421.2942505
3: -88.0919418, 564.7359619, -61.6502228, 394.5868225, -482.6787720, 626.3861694
4: -118.0188751, 457.5017395, -82.7363358, 319.2454529, -437.2642822, 540.2380981

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354788, upper bound: 547.3354332
time: 0.66 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359222, upper bound: 547.3359222
time: 0.73 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -185.2007751, 724.0216064, -185.2007751, 724.0216064, -909.2224121, 909.2224121
1: -120.3639984, 414.3753662, -120.3639984, 414.3753662, -534.7393799, 534.7393799
2: -65.6249084, 375.4895630, -65.6249084, 375.4895630, -441.1144714, 441.1144714
3: -88.0919418, 564.7359619, -88.0919418, 564.7359619, -652.8278809, 652.8278809
4: -118.0188751, 457.5017395, -118.0188751, 457.5017395, -575.5206299, 575.5206299

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354768, upper bound: 547.3354788
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359222, upper bound: 547.3359222
time: 0.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.55 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -547.3365384, upper bound: 547.3362609
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -547.3366236, upper bound: 547.3363458
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -547.3365384, upper bound: 547.3362609
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -547.3366236, upper bound: 547.3363458
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -547.3354788, upper bound: 547.3354332
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -547.3359222, upper bound: 547.3359222
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -547.3354768, upper bound: 547.3354788
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.55
Output dim: 0, lower bound: -547.3359222, upper bound: 547.3359222

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -118.6048203, 465.9146423, -126.2176971, 495.3080139, -613.9128418, 592.1323242
1: -77.0634995, 266.3726807, -82.1137772, 283.5892639, -360.6527100, 348.4864502
2: -42.0011024, 241.4292908, -44.7792931, 256.8154907, -298.8165894, 286.2085876
3: -56.7006721, 362.1216431, -60.3147354, 385.7744751, -442.4751587, 422.4363708
4: -75.9416275, 293.3004456, -80.8956223, 312.2257690, -388.1673889, 374.1960449

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3375285, upper bound: 547.3375285
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3375285, upper bound: 547.3375285
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -147.3437195, 579.3598022, -124.0335541, 486.8173218, -634.1608887, 703.3933716
1: -95.6086121, 331.5986328, -80.6828690, 278.6952820, -374.3038940, 412.2814941
2: -52.2355576, 300.6344604, -44.0031319, 252.4457550, -304.6812439, 344.6375427
3: -70.5680695, 450.5860596, -59.2755470, 379.0448914, -449.6129456, 509.8615417
4: -94.3793259, 365.4140625, -79.4759445, 306.8617249, -401.2409973, 444.8900146

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3375285, upper bound: 547.3375285
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3375285, upper bound: 547.3375285
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -118.6048203, 465.9146423, -183.0251312, 715.4709473, -834.0757446, 648.9396973
1: -77.0634995, 266.3726807, -118.8954163, 409.4447021, -486.5081787, 385.2680969
2: -42.0011024, 241.4292908, -64.8371582, 371.0952148, -413.0963135, 306.2663574
3: -56.7006721, 362.1216431, -87.0844345, 557.9270020, -614.6276245, 449.2060852
4: -75.9416275, 293.3004456, -116.6060791, 452.0999146, -528.0415649, 409.9064941

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3362609
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3361488
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -147.3437195, 579.3598022, -179.4589386, 701.6898804, -849.0335083, 758.8187256
1: -95.6086121, 331.5986328, -116.5065155, 401.3684387, -496.9770203, 448.1051636
2: -52.2355576, 300.6344604, -63.5379372, 363.8156433, -416.0511780, 364.1723938
3: -70.5680695, 450.5860596, -85.3430939, 546.8251953, -617.3932495, 535.9291382
4: -94.3793259, 365.4140625, -114.2450256, 443.1549377, -537.5341187, 479.6590576

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -547.3150112, upper bound: 547.3121549
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3365198, upper bound: 547.3362668
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -185.2007751, 724.0216064, -127.3050919, 499.5563660, -684.7571411, 851.3267212
1: -120.3639984, 414.3753662, -82.8909454, 286.0963135, -406.4603271, 497.2662964
2: -65.6249084, 375.4895630, -45.2099571, 259.0346985, -324.6595764, 420.6995239
3: -88.0919418, 564.7359619, -60.8203354, 389.2677917, -477.3597412, 625.5562744
4: -118.0188751, 457.5017395, -81.6198502, 314.9862061, -433.0050049, 539.1215820

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354953, upper bound: 547.3354290
time: 0.59 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354403, upper bound: 547.3354332
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -185.2007751, 724.0216064, -128.9093781, 505.6997070, -690.9005127, 852.9309692
1: -120.3639984, 414.3753662, -83.9207764, 289.6883545, -410.0523682, 498.2961426
2: -65.6249084, 375.4895630, -45.7610207, 262.2616272, -327.8865356, 421.2505798
3: -88.0919418, 564.7359619, -61.5903130, 394.1990051, -482.2909546, 626.3262939
4: -118.0188751, 457.5017395, -82.6551514, 318.9341125, -436.9529114, 540.1568604

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364588, upper bound: 547.3365134
time: 0.73 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3363573, upper bound: 547.3365134
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -183.6526031, 718.0092163, -185.2007751, 724.0216064, -907.6741943, 903.2099609
1: -119.3616714, 410.8499146, -120.3639984, 414.3753662, -533.7369995, 531.2138672
2: -65.0968323, 372.3075867, -65.6249084, 375.4895630, -440.5863953, 437.9324951
3: -87.3466034, 559.9202271, -88.0919418, 564.7359619, -652.0825806, 648.0121460
4: -117.0054474, 453.6175537, -118.0188751, 457.5017395, -574.5072021, 571.6364136

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3351769, upper bound: 547.3351440
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354768, upper bound: 547.3354122
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -185.0939789, 723.6126709, -185.2007751, 724.0216064, -909.1156006, 908.8134766
1: -120.2936859, 414.1331787, -120.3639984, 414.3753662, -534.6690674, 534.4971924
2: -65.5872269, 375.2719727, -65.6249084, 375.4895630, -441.0767822, 440.8968811
3: -88.0401535, 564.4030762, -88.0919418, 564.7359619, -652.7761230, 652.4949951
4: -117.9489212, 457.2348938, -118.0188751, 457.5017395, -575.4506836, 575.2537842

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359222, upper bound: 547.3358292
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3358292, upper bound: 547.3358292
time: 0.68 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.96 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3375285, upper bound: 547.3375285
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3375285, upper bound: 547.3375285
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3375285, upper bound: 547.3375285
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3375285, upper bound: 547.3375285
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3362609
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3361488
NS_A1_B2_A2_A1, status: Status.VERIFIED, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3150112, upper bound: 547.3121549
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3365198, upper bound: 547.3362668
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3354953, upper bound: 547.3354290
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3354403, upper bound: 547.3354332
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3364588, upper bound: 547.3365134
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3363573, upper bound: 547.3365134
NS_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3351769, upper bound: 547.3351440
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3354768, upper bound: 547.3354122
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3359222, upper bound: 547.3358292
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.96
Output dim: 0, lower bound: -547.3358292, upper bound: 547.3358292

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -118.6048203, 465.9146423, -118.6048203, 465.9146423, -584.5194702, 584.5194702
1: -77.0634995, 266.3726807, -77.0634995, 266.3726807, -343.4361572, 343.4361572
2: -42.0011024, 241.4292908, -42.0011024, 241.4292908, -283.4303894, 283.4303894
3: -56.7006721, 362.1216431, -56.7006721, 362.1216431, -418.8223267, 418.8223267
4: -75.9416275, 293.3004456, -75.9416275, 293.3004456, -369.2420654, 369.2420654

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3377476, upper bound: 547.3376211
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3376670, upper bound: 547.3376211
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -118.6048203, 465.9146423, -147.3437195, 579.3598022, -697.9645996, 613.2582397
1: -77.0634995, 266.3726807, -95.6086121, 331.5986328, -408.6621399, 361.9812927
2: -42.0011024, 241.4292908, -52.2355576, 300.6344604, -342.6355591, 293.6647644
3: -56.7006721, 362.1216431, -70.5680695, 450.5860596, -507.2867432, 432.6896973
4: -75.9416275, 293.3004456, -94.3793259, 365.4140625, -441.3556824, 387.6797180

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3332174, upper bound: 547.3340315
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3373245, upper bound: 547.3372748
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -147.3437195, 579.3598022, -118.6048203, 465.9146423, -613.2583008, 697.9645996
1: -95.6086121, 331.5986328, -77.0634995, 266.3726807, -361.9812927, 408.6621399
2: -52.2355576, 300.6344604, -42.0011024, 241.4292908, -293.6647644, 342.6355591
3: -70.5680695, 450.5860596, -56.7006721, 362.1216431, -432.6896973, 507.2867126
4: -94.3793259, 365.4140625, -75.9416275, 293.3004456, -387.6797180, 441.3556824

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3375285, upper bound: 547.3374674
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3374674, upper bound: 547.3374674
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -147.3437195, 579.3598022, -147.3437195, 579.3598022, -726.7034302, 726.7034302
1: -95.6086121, 331.5986328, -95.6086121, 331.5986328, -427.2072449, 427.2072449
2: -52.2355576, 300.6344604, -52.2355576, 300.6344604, -352.8699646, 352.8699646
3: -70.5680695, 450.5860596, -70.5680695, 450.5860596, -521.1541138, 521.1541138
4: -94.3793259, 365.4140625, -94.3793259, 365.4140625, -459.7933655, 459.7933655

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350198, upper bound: 547.3362510
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3375099, upper bound: 547.3375099
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -115.4907532, 453.8346863, -172.5141296, 674.7101440, -790.2009277, 626.3488159
1: -75.0498810, 259.4769592, -112.0985107, 386.1789551, -461.2288208, 371.5754395
2: -40.9132156, 235.2121582, -61.1771317, 350.0958862, -391.0090942, 296.3892822
3: -55.2183952, 352.6703491, -82.0914764, 526.0520020, -581.2703857, 434.7618408
4: -73.9541550, 285.6864624, -109.9176178, 426.3868713, -500.3410339, 395.6040649

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3349525, upper bound: 547.3351709
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3362609
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -118.1786270, 464.2712402, -242.2734833, 944.4342651, -1062.6127930, 706.5447388
1: -76.7887878, 265.4143066, -158.8534393, 541.6646118, -618.4533691, 424.2677307
2: -41.8527870, 240.5549164, -86.7826920, 489.2894897, -531.1422729, 327.3375854
3: -56.5002213, 360.8206787, -115.6926651, 740.6927490, -797.1929321, 476.5132751
4: -75.6748886, 292.2388000, -155.2146606, 597.8735352, -673.5484009, 447.4534607

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3357758, upper bound: 547.3347304
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3361488
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -145.9341888, 573.8845825, -179.1086426, 700.3265381, -846.2606201, 752.9930420
1: -94.6927719, 328.3910522, -116.2803879, 400.5717468, -495.2644958, 444.6714478
2: -51.7378082, 297.7180481, -63.4156799, 363.0896912, -414.8275146, 361.1337280
3: -69.8980789, 446.2264404, -85.1768341, 545.7463989, -615.6444702, 531.4032593
4: -93.4728317, 361.8752136, -114.0226364, 442.2771606, -535.7500000, 475.8978577

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3353625, upper bound: 547.3353762
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3365079, upper bound: 547.3362533
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -174.7229156, 683.3986816, -124.0490646, 486.9956360, -661.7185669, 807.4477539
1: -113.5901871, 391.1820374, -80.7902145, 278.8891907, -392.4793091, 471.9722290
2: -61.9759560, 354.5487061, -44.0810776, 252.5281372, -314.5040894, 398.6296997
3: -83.1180649, 532.9846191, -59.2851448, 379.4100647, -462.5281372, 592.2697754
4: -111.3572845, 431.8741150, -79.5612183, 307.0381165, -418.3953857, 511.4352722

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354953, upper bound: 547.3353908
time: 0.92 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354105, upper bound: 547.3354290
time: 0.66 seconds

## BFS NS instance: NS_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -244.5253448, 953.3164673, -126.8096313, 497.6323547, -742.1577148, 1080.1258545
1: -160.4060059, 546.7990723, -82.5704880, 284.9830627, -445.3890686, 629.3695679
2: -87.6165009, 493.8547974, -45.0375786, 258.0229492, -345.6394653, 538.8923950
3: -116.7568817, 747.8021240, -60.5876312, 387.7551880, -504.5120850, 808.3896484
4: -156.7073059, 603.5449219, -81.3109894, 313.7564087, -470.4636841, 684.8558960

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3340009, upper bound: 547.3347739
time: 0.86 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3340009, upper bound: 547.3353058
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -174.7229156, 683.3986816, -125.6508713, 493.1234131, -667.8463135, 809.0495605
1: -113.5901871, 391.1820374, -81.8181305, 282.4687195, -396.0588074, 473.0001831
2: -61.9759560, 354.5487061, -44.6310806, 255.7377777, -317.7137451, 399.1797485
3: -83.1180649, 532.9846191, -60.0536537, 384.3278809, -467.4459229, 593.0382690
4: -111.3572845, 431.8741150, -80.5943298, 310.9646912, -422.3219604, 512.4683838

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3361725, upper bound: 547.3358266
time: 0.75 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3360878, upper bound: 547.3358547
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -244.5253448, 953.3164673, -128.4327087, 503.8807068, -748.4060669, 1081.7491455
1: -160.4060059, 546.7990723, -83.6143341, 288.6073914, -449.0133972, 630.4132690
2: -87.6165009, 493.8547974, -45.5962448, 261.2676086, -348.8840942, 539.4510498
3: -116.7568817, 747.8021240, -61.3680077, 392.7404785, -509.4973755, 809.1700439
4: -156.7073059, 603.5449219, -82.3613586, 317.7351379, -474.4424438, 685.9062500

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3348227, upper bound: 547.3358347
time: 0.69 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3363573, upper bound: 547.3365134
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -173.0486145, 676.8876343, -181.9783173, 711.5118408, -884.5603638, 858.8659668
1: -112.5027695, 387.3728638, -118.2814789, 407.2424622, -519.7452393, 505.6543579
2: -61.4033241, 351.1094055, -64.4988632, 369.0508423, -430.4541321, 415.6082153
3: -82.3088455, 527.7659302, -86.5605774, 554.9689331, -637.2777710, 614.3265381
4: -110.2561493, 427.6663513, -115.9682083, 449.6191101, -559.8752441, 543.6345215

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A1_A1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3351769, upper bound: 547.3351440
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3351769, upper bound: 547.3351440
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -242.7575989, 946.4368896, -184.6181946, 721.7727661, -964.5303955, 1131.0550537
1: -159.2518768, 542.7948608, -119.9890747, 413.0648499, -572.3167114, 662.7839355
2: -87.0053558, 490.2557983, -65.4220505, 374.2977600, -461.3031006, 555.6778564
3: -115.8999023, 742.3205566, -87.8173370, 562.9545898, -678.8544922, 830.1377563
4: -155.5369720, 599.1447144, -117.6505508, 456.0552368, -611.5922241, 716.7952271

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_A2_A1

### Relational analysis result of NS_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3339001, upper bound: 547.3346045
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_A2_A2

### Relational analysis result of NS_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354432, upper bound: 547.3354094
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -174.6156464, 682.9879761, -181.9783173, 711.5118408, -886.1273804, 864.9663086
1: -113.5199203, 390.9392090, -118.2814789, 407.2424622, -520.7623901, 509.2207031
2: -61.9382172, 354.3305054, -64.4988632, 369.0508423, -430.9890747, 418.8293457
3: -83.0661392, 532.6502686, -86.5605774, 554.9689331, -638.0350952, 619.2107544
4: -111.2870941, 431.6065063, -115.9682083, 449.6191101, -560.9061279, 547.5746460

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3358292, upper bound: 547.3358292
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3358292, upper bound: 547.3358292
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -244.4250793, 952.9324951, -184.6181946, 721.7727661, -966.1978760, 1137.5506592
1: -160.3404388, 546.5708618, -119.9890747, 413.0648499, -573.4052124, 666.5599365
2: -87.5812073, 493.6492004, -65.4220505, 374.2977600, -461.8789673, 559.0711670
3: -116.7085190, 747.4884033, -87.8173370, 562.9545898, -679.6630859, 835.3056030
4: -156.6418457, 603.2935181, -117.6505508, 456.0552368, -612.6970825, 720.9440308

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344067, upper bound: 547.3351294
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3358292, upper bound: 547.3358292
time: 0.74 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.81 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3377476, upper bound: 547.3376211
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3376670, upper bound: 547.3376211
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3332174, upper bound: 547.3340315
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3373245, upper bound: 547.3372748
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3375285, upper bound: 547.3374674
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3374674, upper bound: 547.3374674
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3350198, upper bound: 547.3362510
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3375099, upper bound: 547.3375099
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3349525, upper bound: 547.3351709
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3362609
NS_A1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3357758, upper bound: 547.3347304
NS_A1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3361488
NS_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3353625, upper bound: 547.3353762
NS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3365079, upper bound: 547.3362533
NS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3354953, upper bound: 547.3353908
NS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3354105, upper bound: 547.3354290
NS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3340009, upper bound: 547.3347739
NS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3340009, upper bound: 547.3353058
NS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3361725, upper bound: 547.3358266
NS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3360878, upper bound: 547.3358547
NS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3348227, upper bound: 547.3358347
NS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3363573, upper bound: 547.3365134
NS_A2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3351769, upper bound: 547.3351440
NS_A2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3351769, upper bound: 547.3351440
NS_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3339001, upper bound: 547.3346045
NS_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3354432, upper bound: 547.3354094
NS_A2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3358292, upper bound: 547.3358292
NS_A2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3358292, upper bound: 547.3358292
NS_A2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3344067, upper bound: 547.3351294
NS_A2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 0, lower bound: -547.3358292, upper bound: 547.3358292

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -108.7559891, 427.6845703, -115.4907532, 453.8346863, -562.5906372, 543.1752930
1: -70.7119446, 244.6304626, -75.0498810, 259.4769592, -330.1887817, 319.6803284
2: -38.5643959, 221.8440704, -40.9132156, 235.2121582, -273.7765503, 262.7572937
3: -52.0113754, 332.3046570, -55.2183952, 352.6703491, -404.6816711, 387.5230408
4: -69.6674957, 269.3063049, -73.9541550, 285.6864624, -355.3539429, 343.2604675

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3381676, upper bound: 547.3381676
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3381676, upper bound: 547.3381676
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -179.8776245, 698.5099487, -118.1786270, 464.2712402, -644.1488647, 816.6884766
1: -118.1231613, 402.5992737, -76.7887878, 265.4143066, -383.5374146, 479.3880310
2: -64.3650055, 363.9567566, -41.8527870, 240.5549164, -304.9198914, 405.8095398
3: -85.6724854, 549.7713013, -56.5002213, 360.8206787, -446.4931335, 606.2714844
4: -115.1039124, 444.3533325, -75.6748886, 292.2388000, -407.3427124, 520.0281982

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3355385, upper bound: 547.3364369
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3373922, upper bound: 547.3373922
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -102.9554672, 404.4634094, -137.5620575, 541.4196777, -644.3750610, 542.0254517
1: -67.3843536, 232.0990143, -88.9998627, 309.5718689, -376.9562378, 321.0988770
2: -36.5505371, 211.8211823, -48.6455765, 280.8981018, -317.4486389, 260.4667358
3: -48.9901962, 314.8679504, -65.8205719, 420.1401367, -469.1303101, 380.6885376
4: -65.6655426, 256.1583862, -87.9153900, 341.0643005, -406.7298279, 344.0736694

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -547.3323732, upper bound: 547.3321063
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3330249, upper bound: 547.3339239
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -115.4919281, 454.0259094, -146.2902222, 575.3966675, -690.8886108, 600.3161621
1: -75.0351257, 259.3142700, -94.9097824, 329.2571716, -404.2922974, 354.2240295
2: -40.8721123, 235.1239319, -51.8424110, 298.5605164, -339.4326172, 286.9662781
3: -55.1936913, 352.4516296, -70.0583191, 447.3514099, -502.5450745, 422.5099182
4: -73.8746719, 285.6011658, -93.6824265, 362.8189392, -436.6936035, 379.2835999

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3360081, upper bound: 547.3356947
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372640, upper bound: 547.3372748
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372640, upper bound: 547.3371425
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -136.8784637, 538.5484009, -115.4907532, 453.8346863, -590.7130737, 654.0391235
1: -88.7612076, 308.3933105, -75.0498810, 259.4769592, -348.2381592, 383.4431763
2: -48.5535889, 279.7159729, -40.9132156, 235.2121582, -283.7657471, 320.6291504
3: -65.5735931, 418.7428894, -55.2183952, 352.6703491, -418.2439575, 473.9612732
4: -87.6646500, 339.7084351, -73.9541550, 285.6864624, -373.3511047, 413.6625671

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3374253, upper bound: 547.3373284
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3361095, upper bound: 547.3365363
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -207.1838379, 808.2752686, -118.1786270, 464.2712402, -671.4550781, 926.4538574
1: -136.3029175, 464.7058716, -76.7887878, 265.4143066, -401.7171631, 541.4946289
2: -74.3993683, 420.1829224, -41.8527870, 240.5549164, -314.9542542, 462.0357056
3: -99.1875381, 634.7614136, -56.5002213, 360.8206787, -460.0081482, 691.2615356
4: -133.0923920, 513.3529663, -75.6748886, 292.2388000, -425.3311768, 589.0278320

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A2_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350198, upper bound: 547.3358856
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3375982, upper bound: 547.3376460
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -146.2970581, 575.4581299, -147.3437195, 579.3598022, -725.6568604, 722.8018188
1: -94.9409409, 329.2826233, -95.6086121, 331.5986328, -426.5395508, 424.8912048
2: -51.8796577, 298.5939941, -52.2355576, 300.6344604, -352.5141296, 350.8294678
3: -70.0485916, 447.3705750, -70.5680695, 450.5860596, -520.6346436, 517.9386597
4: -93.6734238, 362.8653564, -94.3793259, 365.4140625, -459.0874939, 457.2446594

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3352074, upper bound: 547.3345227
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3357557, upper bound: 547.3362195
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -147.2199554, 578.8848877, -147.3437195, 579.3598022, -726.5797729, 726.2285156
1: -95.5281525, 331.3215332, -95.6086121, 331.5986328, -427.1267700, 426.9301453
2: -52.1919632, 300.3858337, -52.2355576, 300.6344604, -352.8264160, 352.6213074
3: -70.5088043, 450.2049255, -70.5680695, 450.5860596, -521.0948486, 520.7730103
4: -94.2991867, 365.1076050, -94.3793259, 365.4140625, -459.7132568, 459.4869080

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3374482, upper bound: 547.3375099
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3374482, upper bound: 547.3374482
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -108.5239182, 426.4191284, -169.0599976, 661.1000977, -769.6239624, 595.4790649
1: -70.3758774, 243.9613037, -109.8154678, 378.4408569, -448.8167419, 353.7767639
2: -38.3077965, 221.2121582, -59.9218178, 343.1178589, -381.4256592, 281.1339722
3: -51.8373451, 331.5334473, -80.4220276, 515.5190430, -567.3563843, 411.9554443
4: -69.4342041, 268.5819397, -107.6770706, 417.8942566, -487.3284607, 376.2590027

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 22

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -115.0886688, 452.2629089, -172.5141296, 674.7101440, -789.7988281, 624.7770386
1: -74.7898636, 258.5757751, -112.0985107, 386.1789551, -460.9688110, 370.6742554
2: -40.7720642, 234.3995514, -61.1771317, 350.0958862, -390.8679199, 295.5766907
3: -55.0218239, 351.4407654, -82.0914764, 526.0520020, -581.0738525, 433.5322266
4: -73.6909561, 284.7017517, -109.9176178, 426.3868713, -500.0778198, 394.6193848

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B1_A2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3362609
time: 1.21 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3362609
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -115.5188675, 453.8701172, -234.6864471, 912.4656982, -1027.9843750, 688.5565186
1: -75.0096893, 259.5158386, -153.5632324, 525.3931274, -600.4028320, 413.0790710
2: -40.8565712, 235.2480316, -83.8469467, 475.0604248, -515.9169922, 319.0949707
3: -55.1925430, 352.7630615, -111.6276855, 717.7794189, -772.9718628, 464.3907471
4: -73.9278259, 285.7382812, -149.8214417, 580.1142578, -654.0421143, 435.5597229

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3347344, upper bound: 547.3342917
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3347344, upper bound: 547.3347304
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -118.1786270, 464.2712402, -241.9587555, 943.1670532, -1061.3455811, 706.2299805
1: -76.7887878, 265.4143066, -158.6507874, 540.9490967, -617.7377930, 424.0650330
2: -41.8527870, 240.5549164, -86.6736526, 488.6405945, -530.4934082, 327.2285767
3: -56.5002213, 360.8206787, -115.5419312, 739.7202759, -796.2205200, 476.3625488
4: -75.6748886, 292.2388000, -155.0135803, 597.0939941, -672.7688599, 447.2523193

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3361487
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3361488
time: 1.04 seconds

## BFS NS instance: NS_A1_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -144.9432220, 570.1853027, -179.1086426, 700.3265381, -845.2696533, 749.2938232
1: -94.0605774, 326.2134705, -116.2803879, 400.5717468, -494.6323242, 442.4938660
2: -51.4008789, 295.8113403, -63.4156799, 363.0896912, -414.4905701, 359.2270203
3: -69.4038010, 443.1940308, -85.1768341, 545.7463989, -615.1501465, 528.3708496
4: -92.7992935, 359.4843750, -114.0226364, 442.2771606, -535.0764160, 473.5070190

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3353532, upper bound: 547.3353762
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3353625, upper bound: 547.3353241
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -145.8082123, 573.4008789, -179.1086426, 700.3265381, -846.1346436, 752.5094604
1: -94.6108475, 328.1087036, -116.2803879, 400.5717468, -495.1825256, 444.3890991
2: -51.6934624, 297.4647827, -63.4156799, 363.0896912, -414.7831116, 360.8804626
3: -69.8378601, 445.8383179, -85.1768341, 545.7463989, -615.5842285, 531.0151367
4: -93.3912582, 361.5633240, -114.0226364, 442.2771606, -535.6683960, 475.5859680

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364177, upper bound: 547.3362533
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364177, upper bound: 547.3361778
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -172.0180054, 672.7317505, -124.0490646, 486.9956360, -659.0136719, 796.7807617
1: -111.9222412, 385.0429688, -80.7902145, 278.8891907, -390.8113708, 465.8331299
2: -61.0769844, 348.8662415, -44.0810776, 252.5281372, -313.6050720, 392.9472656
3: -81.8933868, 524.8155518, -59.2851448, 379.4100647, -461.3034668, 584.1007080
4: -109.7489929, 425.1504822, -79.5612183, 307.0381165, -416.7870789, 504.7117004

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3349563, upper bound: 547.3350790
time: 0.68 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3349563, upper bound: 547.3353908
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -159.6043091, 625.1950073, -121.5694809, 477.2576904, -636.8619995, 746.7642822
1: -104.4380341, 358.5418701, -79.2096786, 273.3335571, -377.7716064, 437.7515564
2: -56.8458138, 324.5385132, -43.2022362, 247.4823456, -304.3281555, 367.7407532
3: -76.0734711, 489.2314453, -58.0992775, 371.8829041, -447.9563599, 547.3307495
4: -102.4814453, 395.4103394, -77.9874191, 300.8895264, -403.3709717, 473.3977661

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A1_A2_A1

### Relational analysis result of NS_A2_B1_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350534, upper bound: 547.3348251
time: 0.63 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3349563, upper bound: 547.3351287
time: 0.67 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3349563, upper bound: 547.3354290
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -236.8008728, 920.8035278, -124.0638046, 486.8692932, -723.6699219, 1044.8673096
1: -155.0174866, 530.2402954, -80.7336655, 278.8927612, -433.9102478, 610.9737549
2: -84.6329422, 479.3962708, -44.0090332, 252.5409851, -337.1739197, 523.4052734
3: -112.6263580, 724.4938354, -59.2365875, 379.4322510, -492.0585938, 783.7302856
4: -151.2003784, 585.4775391, -79.5154419, 307.0363770, -458.2367554, 664.9929810

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A2_A1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3340009, upper bound: 547.3347284
time: 0.72 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3337920, upper bound: 547.3347737
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -244.2141571, 952.0649414, -126.8096313, 497.6323547, -741.8464966, 1078.8745117
1: -160.2055054, 546.0914307, -82.5704880, 284.9830627, -445.1885376, 628.6619263
2: -87.5085297, 493.2127075, -45.0375786, 258.0229492, -345.5314941, 538.2503052
3: -116.6075974, 746.8405151, -60.5876312, 387.7551880, -504.3627930, 807.4281006
4: -156.5083008, 602.7738037, -81.3109894, 313.7564087, -470.2646484, 684.0847778

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A2_A2_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354204, upper bound: 547.3352774
time: 0.65 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3353439, upper bound: 547.3353058
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -172.0180054, 672.7317505, -125.6508713, 493.1234131, -665.1414185, 798.3826294
1: -111.9222412, 385.0429688, -81.8181305, 282.4687195, -394.3908691, 466.8610840
2: -61.0769844, 348.8662415, -44.6310806, 255.7377777, -316.8146973, 393.4973145
3: -81.8933868, 524.8155518, -60.0536537, 384.3278809, -466.2212524, 584.8692017
4: -109.7489929, 425.1504822, -80.5943298, 310.9646912, -420.7136536, 505.7448120

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B2_A1_A1_B1

### Relational analysis result of NS_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3361696, upper bound: 547.3358266
time: 1.02 seconds

## Relational analysis of NS_A2_B1_B2_A1_A1_B2

### Relational analysis result of NS_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3361696, upper bound: 547.3358266
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -159.6043091, 625.1950073, -123.1080856, 483.1737366, -642.7780762, 748.3029785
1: -104.4380341, 358.5418701, -80.1988983, 276.7862549, -381.2242432, 438.7407837
2: -56.8458138, 324.5385132, -43.7291489, 250.5762482, -307.4220581, 368.2676392
3: -76.0734711, 489.2314453, -58.8367043, 376.6250305, -452.6984863, 548.0681152
4: -102.4814453, 395.4103394, -78.9801712, 304.6763000, -407.1577454, 474.3905029

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B2_A1_A2_B1

### Relational analysis result of NS_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3360877, upper bound: 547.3358547
time: 0.67 seconds

## Relational analysis of NS_A2_B1_B2_A1_A2_B2

### Relational analysis result of NS_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3360877, upper bound: 547.3358547
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -236.8008728, 920.8035278, -125.7112198, 493.2101746, -730.0109253, 1046.5147705
1: -155.0174866, 530.2402954, -81.7930908, 282.5653381, -437.5827942, 612.0331421
2: -84.6329422, 479.3962708, -44.5765915, 255.8278656, -340.4607544, 523.9727783
3: -112.6263580, 724.4938354, -60.0293770, 384.4851990, -497.1115417, 784.5231934
4: -151.2003784, 585.4775391, -80.5823593, 311.0671082, -462.2674866, 666.0598755

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A2_A1_A1

### Relational analysis result of NS_A2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344873, upper bound: 547.3351428
time: 0.76 seconds

## Relational analysis of NS_A2_B1_B2_A2_A1_A2

### Relational analysis result of NS_A2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3343477, upper bound: 547.3351915
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -244.2141571, 952.0649414, -128.4327087, 503.8807068, -748.0948486, 1080.4976807
1: -160.2055054, 546.0914307, -83.6143341, 288.6073914, -448.8128662, 629.7057495
2: -87.5085297, 493.2127075, -45.5962448, 261.2676086, -348.7761230, 538.8089600
3: -116.6075974, 746.8405151, -61.3680077, 392.7404785, -509.3480835, 808.2084961
4: -156.5083008, 602.7738037, -82.3613586, 317.7351379, -474.2433777, 685.1351318

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B2_A2_A2_A1

### Relational analysis result of NS_A2_B1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3360819, upper bound: 547.3358266
time: 0.68 seconds

## Relational analysis of NS_A2_B1_B2_A2_A2_A2

### Relational analysis result of NS_A2_B1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3360066, upper bound: 547.3358547
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -173.0486145, 676.8876343, -174.7229156, 683.3986816, -856.4472656, 851.6105347
1: -112.5027695, 387.3728638, -113.5901871, 391.1820374, -503.6848145, 500.9630127
2: -61.4033241, 351.1094055, -61.9759560, 354.5487061, -415.9519653, 413.0853577
3: -82.3088455, 527.7659302, -83.1180649, 532.9846191, -615.2933960, 610.8839111
4: -110.2561493, 427.6663513, -111.3572845, 431.8741150, -542.1301880, 539.0234985

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344930, upper bound: 547.3343767
time: 0.59 seconds

## Relational analysis of NS_A2_B2_A1_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344930, upper bound: 547.3351440
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -173.0486145, 676.8876343, -244.5253448, 953.3164673, -1126.3649902, 921.4129639
1: -112.5027695, 387.3728638, -160.4060059, 546.7990723, -659.3017578, 547.7788086
2: -61.4033241, 351.1094055, -87.6165009, 493.8547974, -555.2580566, 438.7258911
3: -82.3088455, 527.7659302, -116.7568817, 747.8021240, -830.1108398, 644.5227661
4: -110.2561493, 427.6663513, -156.7073059, 603.5449219, -713.8010864, 584.3735352

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344930, upper bound: 547.3347727
time: 0.92 seconds

## Relational analysis of NS_A2_B2_A1_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344930, upper bound: 547.3351440
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -235.0265503, 913.9286499, -181.5809021, 709.8593750, -944.8859253, 1095.5091553
1: -153.8684387, 526.1890869, -117.9696274, 406.2825012, -560.1508789, 644.1586304
2: -84.0225525, 475.7658997, -64.3216095, 368.1724548, -452.1949768, 540.0875244
3: -111.7723846, 718.9353027, -86.3501968, 553.7274170, -665.4998169, 805.2854614
4: -150.0388794, 581.0574951, -115.6889343, 448.5836487, -598.6224365, 696.7463379

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_A2_A1_B1

### Relational analysis result of NS_A2_B2_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3338574, upper bound: 547.3346046
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_A2_A1_B2

### Relational analysis result of NS_A2_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3338874, upper bound: 547.3345585
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -242.4530945, 945.2138672, -184.6181946, 721.7727661, -964.2258301, 1129.8320312
1: -159.0559235, 542.1032715, -119.9890747, 413.0648499, -572.1207886, 662.0922852
2: -86.8998871, 489.6285706, -65.4220505, 374.2977600, -461.1976318, 555.0505371
3: -115.7540359, 741.3802490, -87.8173370, 562.9545898, -678.7086182, 829.1974487
4: -155.3423004, 598.3914185, -117.6505508, 456.0552368, -611.3975220, 716.0419312

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_A2_A2_B1

### Relational analysis result of NS_A2_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354110, upper bound: 547.3354094
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_A2_A2_B2

### Relational analysis result of NS_A2_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3354432, upper bound: 547.3353311
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -174.6156464, 682.9879761, -174.7229156, 683.3986816, -858.0142822, 857.7108765
1: -113.5199203, 390.9392090, -113.5901871, 391.1820374, -504.7019653, 504.5293579
2: -61.9382172, 354.3305054, -61.9759560, 354.5487061, -416.4869385, 416.3064575
3: -83.0661392, 532.6502686, -83.1180649, 532.9846191, -616.0507812, 615.7681274
4: -111.2870941, 431.6065063, -111.3572845, 431.8741150, -543.1610718, 542.9636230

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359222, upper bound: 547.3357312
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3358521, upper bound: 547.3357622
time: 0.87 seconds

## BFS NS instance: NS_A2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -174.6156464, 682.9879761, -244.5253448, 953.3164673, -1127.9317627, 927.5133057
1: -113.5199203, 390.9392090, -160.4060059, 546.7990723, -660.3188477, 551.3452148
2: -61.9382172, 354.3305054, -87.6165009, 493.8547974, -555.7930298, 441.9470215
3: -83.0661392, 532.6502686, -116.7568817, 747.8021240, -830.8682251, 649.4069824
4: -111.2870941, 431.6065063, -156.7073059, 603.5449219, -714.8319702, 588.3137207

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359222, upper bound: 547.3357312
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3358521, upper bound: 547.3357622
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -236.7053680, 920.4384766, -181.5809021, 709.8593750, -946.5647583, 1102.0191650
1: -154.9550476, 530.0231934, -117.9696274, 406.2825012, -561.2375488, 647.9926758
2: -84.5993195, 479.2006836, -64.3216095, 368.1724548, -452.7717590, 543.5222778
3: -112.5802765, 724.1959839, -86.3501968, 553.7274170, -666.3076782, 810.5460815
4: -151.1379852, 585.2386475, -115.6889343, 448.5836487, -599.7216187, 700.9275513

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_A2_A1_B1

### Relational analysis result of NS_A2_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3340427, upper bound: 547.3351293
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A2_A2_A1_B2

### Relational analysis result of NS_A2_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3340714, upper bound: 547.3350842
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -244.1137390, 951.6805420, -184.6181946, 721.7727661, -965.8864746, 1136.2987061
1: -160.1398468, 545.8630371, -119.9890747, 413.0648499, -573.2047119, 665.8521118
2: -87.4731979, 493.0069275, -65.4220505, 374.2977600, -461.7709656, 558.4289551
3: -116.5592117, 746.5264282, -87.8173370, 562.9545898, -679.5137939, 834.3436279
4: -156.4427490, 602.5220337, -117.6505508, 456.0552368, -612.4979248, 720.1726074

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_A2_A2_B1

### Relational analysis result of NS_A2_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3357312, upper bound: 547.3358292
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A2_A2_A2_B2

### Relational analysis result of NS_A2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3357622, upper bound: 547.3357622
time: 0.93 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.05 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3381676, upper bound: 547.3381676
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3381676, upper bound: 547.3381676
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3355385, upper bound: 547.3364369
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3373922, upper bound: 547.3373922
NS_A1_B1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3323732, upper bound: 547.3321063
NS_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3330249, upper bound: 547.3339239
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3372640, upper bound: 547.3372748
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3372640, upper bound: 547.3371425
NS_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3374253, upper bound: 547.3373284
NS_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3361095, upper bound: 547.3365363
NS_A1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3350198, upper bound: 547.3358856
NS_A1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3375982, upper bound: 547.3376460
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3352074, upper bound: 547.3345227
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3357557, upper bound: 547.3362195
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3374482, upper bound: 547.3375099
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3374482, upper bound: 547.3374482
NS_A1_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3362609
NS_A1_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3362609
NS_A1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3347344, upper bound: 547.3342917
NS_A1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3347344, upper bound: 547.3347304
NS_A1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3361487
NS_A1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3364384, upper bound: 547.3361488
NS_A1_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3353532, upper bound: 547.3353762
NS_A1_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3353625, upper bound: 547.3353241
NS_A1_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3364177, upper bound: 547.3362533
NS_A1_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3364177, upper bound: 547.3361778
NS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3349563, upper bound: 547.3350790
NS_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3349563, upper bound: 547.3353908
NS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3349563, upper bound: 547.3351287
NS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3349563, upper bound: 547.3354290
NS_A2_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3340009, upper bound: 547.3347284
NS_A2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3337920, upper bound: 547.3347737
NS_A2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3354204, upper bound: 547.3352774
NS_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3353439, upper bound: 547.3353058
NS_A2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3361696, upper bound: 547.3358266
NS_A2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3361696, upper bound: 547.3358266
NS_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3360877, upper bound: 547.3358547
NS_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3360877, upper bound: 547.3358547
NS_A2_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3344873, upper bound: 547.3351428
NS_A2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3343477, upper bound: 547.3351915
NS_A2_B1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3360819, upper bound: 547.3358266
NS_A2_B1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3360066, upper bound: 547.3358547
NS_A2_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3344930, upper bound: 547.3343767
NS_A2_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3344930, upper bound: 547.3351440
NS_A2_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3344930, upper bound: 547.3347727
NS_A2_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3344930, upper bound: 547.3351440
NS_A2_B2_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3338574, upper bound: 547.3346046
NS_A2_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3338874, upper bound: 547.3345585
NS_A2_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3354110, upper bound: 547.3354094
NS_A2_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3354432, upper bound: 547.3353311
NS_A2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3359222, upper bound: 547.3357312
NS_A2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3358521, upper bound: 547.3357622
NS_A2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3359222, upper bound: 547.3357312
NS_A2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3358521, upper bound: 547.3357622
NS_A2_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3340427, upper bound: 547.3351293
NS_A2_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3340714, upper bound: 547.3350842
NS_A2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3357312, upper bound: 547.3358292
NS_A2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.05
Output dim: 0, lower bound: -547.3357622, upper bound: 547.3357622

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -108.7559891, 427.6845703, -108.7559891, 427.6845703, -536.4405518, 536.4405518
1: -70.7119446, 244.6304626, -70.7119446, 244.6304626, -315.3422546, 315.3422852
2: -38.5643959, 221.8440704, -38.5643959, 221.8440704, -260.4084778, 260.4084778
3: -52.0113754, 332.3046570, -52.0113754, 332.3046570, -384.3159790, 384.3159790
4: -69.6674957, 269.3063049, -69.6674957, 269.3063049, -338.9738159, 338.9738159

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3379400, upper bound: 547.3373505
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3381236, upper bound: 547.3380853
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -108.7559891, 427.6845703, -179.8776245, 698.5099487, -807.2659302, 607.5621948
1: -70.7119446, 244.6304626, -118.1231613, 402.5992737, -473.3111267, 362.7535706
2: -38.5643959, 221.8440704, -64.3650055, 363.9567566, -402.5211487, 286.2090759
3: -52.0113754, 332.3046570, -85.6724854, 549.7713013, -601.7826538, 417.9771423
4: -69.6674957, 269.3063049, -115.1039124, 444.3533325, -514.0208130, 384.4102173

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364372, upper bound: 547.3355385
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364086, upper bound: 547.3373922
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -169.5915527, 659.1163940, -102.6120224, 403.1592407, -572.7507324, 761.7283936
1: -111.3898849, 379.4786987, -67.1684875, 231.3465118, -342.7363892, 446.6471558
2: -60.6673584, 343.2260437, -36.4319000, 211.1386414, -271.8059692, 379.6579285
3: -80.7265930, 518.0808105, -48.8294525, 313.8450317, -394.5716248, 566.9102783
4: -108.5351944, 419.0343323, -65.4546204, 255.3256226, -363.8608093, 484.4889526

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344686, upper bound: 547.3344686
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344686, upper bound: 547.3363673
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -178.4896240, 693.2478638, -115.0817947, 452.4363098, -630.9257812, 808.3295288
1: -117.2097092, 399.4404297, -74.7697983, 258.3942871, -375.6040039, 474.2102356
2: -63.8669205, 361.1285706, -40.7289505, 234.2878418, -298.1547546, 401.8575134
3: -85.0251694, 545.4519043, -55.0002098, 351.1993713, -436.2245483, 600.4520874
4: -114.2165756, 440.8938904, -73.6155624, 284.5831604, -398.7996521, 514.5094604

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3363673, upper bound: 547.3355385
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3363673, upper bound: 547.3373922
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -106.9980316, 418.0610962, -135.5341492, 533.5296631, -640.5276489, 553.5951538
1: -69.9164276, 239.3899384, -87.6787033, 304.9389343, -374.8553467, 327.0686340
2: -38.1590729, 218.3709259, -47.9146690, 276.7015991, -314.8606567, 266.2855835
3: -50.9343872, 324.9395752, -64.8237381, 413.8484192, -464.7828064, 389.7633057
4: -68.0475540, 264.3279419, -86.5827713, 335.9777527, -404.0252686, 350.9107056

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3322706, upper bound: 547.3330032
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -547.3323759, upper bound: 547.3318631
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -112.3978424, 441.9947510, -135.8733826, 534.7472534, -647.1450195, 577.8681030
1: -73.0343933, 252.4578400, -88.0908432, 306.1489868, -379.1833801, 340.5486450
2: -39.7932053, 228.9405975, -48.1765938, 277.7227783, -317.5158997, 277.1171570
3: -53.7187881, 343.0549011, -65.0847168, 415.6481934, -469.3669739, 408.1395874
4: -71.8950043, 278.0288391, -86.9971695, 337.2511902, -409.1461792, 365.0260010

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3323091, upper bound: 547.3334554
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3366586, upper bound: 547.3369918
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3342925, upper bound: 547.3338260
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -115.0817947, 452.4363098, -206.1238251, 804.2843628, -919.3660889, 658.5601196
1: -74.7697983, 258.3942871, -135.6202240, 462.3178711, -537.0875854, 394.0144958
2: -40.7289505, 234.2878418, -74.0207291, 418.0561523, -458.7850952, 308.3085632
3: -55.0002098, 351.1993713, -98.6917648, 631.4942627, -686.4944458, 449.8911438
4: -73.6155624, 284.5831604, -132.4195099, 510.7575073, -584.3729858, 417.0026245

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3363729, upper bound: 547.3356349
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3372640, upper bound: 547.3371319
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -126.5937500, 500.1033020, -110.0016861, 432.5065308, -559.1001587, 610.1049805
1: -82.1735764, 286.2380066, -71.4776535, 247.0712891, -329.2448425, 357.7156372
2: -44.8229370, 260.1660461, -38.9591446, 224.0068665, -268.8297424, 299.1251831
3: -60.2654152, 388.0087280, -52.5825539, 335.7175903, -395.9830017, 440.5912781
4: -80.8068542, 315.3588562, -70.3855286, 271.9840698, -352.7908630, 385.7443848

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3374253, upper bound: 547.3373284
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3374253, upper bound: 547.3373284
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -134.8220062, 530.5065308, -114.7112198, 450.8707886, -585.6928101, 645.2177124
1: -87.4035873, 303.6937866, -74.5485840, 257.7186279, -345.1222229, 378.2423706
2: -47.8266335, 275.3960876, -40.6456528, 233.5998688, -281.4265137, 316.0417480
3: -64.6317673, 412.3696289, -54.8524857, 350.2806091, -414.9123840, 467.2221069
4: -86.4008865, 334.4718933, -73.4746933, 283.7297058, -370.1305542, 407.9465332

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3352138, upper bound: 547.3338225
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3361095, upper bound: 547.3365363
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3361095, upper bound: 547.3365363
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -206.0284424, 803.8794556, -118.1786270, 464.2712402, -670.2996826, 922.0579834
1: -135.5406342, 462.0937805, -76.7887878, 265.4143066, -400.9548340, 538.8825684
2: -73.9904633, 417.8884888, -41.8527870, 240.5549164, -314.5453491, 459.7412720
3: -98.6049500, 631.1180420, -56.5002213, 360.8206787, -459.4255981, 687.6181641
4: -132.3034821, 510.5125427, -75.6748886, 292.2388000, -424.5422974, 586.1874390

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3343000, upper bound: 547.3357885
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344700, upper bound: 547.3357438
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -207.0769653, 807.8642578, -118.1786270, 464.2712402, -671.3482056, 926.0428467
1: -136.2335205, 464.4653320, -76.7887878, 265.4143066, -401.6477051, 541.2540283
2: -74.3618546, 419.9668579, -41.8527870, 240.5549164, -314.9166870, 461.8196411
3: -99.1360474, 634.4314575, -56.5002213, 360.8206787, -459.9566650, 690.9315796
4: -133.0228729, 513.0885010, -75.6748886, 292.2388000, -425.2616577, 588.7633057

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3359255, upper bound: 547.3367366
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3375884, upper bound: 547.3376460
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -143.3329773, 563.9201050, -139.0476837, 546.8877563, -690.2207031, 702.9677124
1: -92.9642334, 322.6900024, -90.0746765, 313.0296021, -405.9938354, 412.7646790
2: -50.7818336, 292.6382446, -49.1942024, 283.7967834, -334.5785828, 341.8324280
3: -68.6086197, 438.3780518, -66.5968933, 425.3624573, -493.9710693, 504.9748840
4: -91.7601852, 355.5841370, -89.0939636, 344.9230347, -436.6831970, 444.6781006

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3339776, upper bound: 547.3342488
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3339776, upper bound: 547.3345227
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -146.2970581, 575.4581299, -146.9592133, 577.8666992, -724.1636963, 722.4173584
1: -94.9409409, 329.2826233, -95.3567734, 330.7406616, -425.6815491, 424.6394043
2: -51.8796577, 298.5939941, -52.0991516, 299.8654175, -351.7450562, 350.6931458
3: -70.0485916, 447.3705750, -70.3796997, 449.4067688, -519.4553223, 517.7502441
4: -93.6734238, 362.8653564, -94.1226044, 364.4730530, -458.1464844, 456.9879761

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344902, upper bound: 547.3359449
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3344902, upper bound: 547.3362195
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -144.0029297, 566.3407593, -136.8784637, 538.5484009, -682.5512085, 703.2191772
1: -93.4244385, 324.1745300, -88.7612076, 308.3933105, -401.8177490, 412.9357300
2: -51.0601273, 293.9448547, -48.5535889, 279.7159729, -330.7760620, 342.4984436
3: -68.9747238, 440.4020996, -65.5735931, 418.7428894, -487.7176208, 505.9757080
4: -92.2325439, 357.1856079, -87.6646500, 339.7084351, -431.9409790, 444.8502502

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3374482, upper bound: 547.3374482
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3374482, upper bound: 547.3374482
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -146.6602173, 576.7144775, -207.1838379, 808.2752686, -954.9354858, 783.8983154
1: -95.1653976, 330.0547485, -136.3029175, 464.7058716, -559.8712769, 466.3576660
2: -51.9960251, 299.2279663, -74.3993683, 420.1829224, -472.1789551, 373.6273193
3: -70.2444534, 448.4895935, -99.1875381, 634.7614136, -705.0057983, 547.6771240
4: -93.9456711, 363.7011719, -133.0923920, 513.3529663, -607.2985229, 496.7935791

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3365621, upper bound: 547.3357520
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3374336, upper bound: 547.3374336
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -108.3764572, 426.1965027, -172.5141296, 674.7101440, -783.0866089, 598.7106323
1: -70.4663162, 243.7790985, -112.0985107, 386.1789551, -456.6452637, 355.8775635
2: -38.4310455, 221.0779877, -61.1771317, 350.0958862, -388.5268555, 282.2551270
3: -51.8251228, 331.1454468, -82.0914764, 526.0520020, -577.8771362, 413.2369385
4: -69.4179153, 268.3781128, -109.9176178, 426.3868713, -495.8047180, 378.2957153

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 22

## BFS NS instance: NS_A1_B2_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -178.9822235, 694.9783936, -172.5141296, 674.7101440, -853.6923218, 867.4925537
1: -117.5343704, 400.5572815, -112.0985107, 386.1789551, -503.7133179, 512.6557617
2: -64.0445938, 362.1350403, -61.1771317, 350.0958862, -414.1404724, 423.3121643
3: -85.2236862, 546.9774780, -82.0914764, 526.0520020, -611.2756958, 629.0689697
4: -114.4789124, 442.1475525, -109.9176178, 426.3868713, -540.8657837, 552.0651855

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## BFS NS instance: NS_A1_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -109.7459869, 431.1549072, -234.6864471, 912.4656982, -1022.2116089, 665.8413696
1: -71.1608276, 246.5720215, -153.5632324, 525.3931274, -596.5538940, 400.1352539
2: -38.7356796, 223.5456390, -83.8469467, 475.0604248, -513.7960815, 307.3925781
3: -52.4377518, 335.1505737, -111.6276855, 717.7794189, -770.2171021, 446.7782593
4: -70.2108688, 271.4703979, -149.8214417, 580.1142578, -650.3251343, 421.2918396

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 22

## BFS NS instance: NS_A1_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -117.6707001, 462.2816467, -234.6864471, 912.4656982, -1030.1362305, 696.9680786
1: -76.4589996, 264.2695618, -153.5632324, 525.3931274, -601.8519897, 417.8327942
2: -41.6734848, 239.5236511, -83.8469467, 475.0604248, -516.7338867, 323.3706055
3: -56.2516937, 359.2589722, -111.6276855, 717.7794189, -774.0311279, 470.8866577
4: -75.3398361, 290.9872131, -149.8214417, 580.1142578, -655.4541016, 440.8086548

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 22

## BFS NS instance: NS_A1_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -108.7559891, 427.6845703, -241.9587555, 943.1670532, -1051.9227295, 669.6433105
1: -70.7119446, 244.6304626, -158.6507874, 540.9490967, -611.6608887, 403.2812500
2: -38.5643959, 221.8440704, -86.6736526, 488.6405945, -527.2049561, 308.5177307
3: -52.0113754, 332.3046570, -115.5419312, 739.7202759, -791.7316284, 447.8465576
4: -69.6674957, 269.3063049, -155.0135803, 597.0939941, -666.7614746, 424.3198547

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 22

## BFS NS instance: NS_A1_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -179.8776245, 698.5099487, -241.9587555, 943.1670532, -1123.0444336, 940.4685669
1: -118.1231613, 402.5992737, -158.6507874, 540.9490967, -659.0722656, 561.2500610
2: -64.3650055, 363.9567566, -86.6736526, 488.6405945, -553.0056152, 450.6304016
3: -85.6724854, 549.7713013, -115.5419312, 739.7202759, -825.3927612, 665.3132324
4: -115.1039124, 444.3533325, -155.0135803, 597.0939941, -712.1978760, 599.3669434

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 22

## BFS NS instance: NS_A1_B2_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -141.6840515, 557.4783936, -168.4913940, 659.1652222, -800.8492432, 725.9697266
1: -91.9300308, 318.9647827, -109.4170151, 377.0559082, -468.9859314, 428.3818054
2: -50.2553596, 289.2765808, -59.7179184, 341.8567505, -392.1121216, 348.9945068
3: -67.8500824, 433.2506104, -80.1343842, 513.5354614, -581.3855591, 513.3850098
4: -90.7080917, 351.4519348, -107.2683029, 416.2887878, -506.9968872, 458.7201538

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3353322, upper bound: 547.3353762
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B2_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3353532, upper bound: 547.3352770
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -144.4079132, 568.0770874, -238.3710480, 929.2578125, -1073.6657715, 806.4481201
1: -93.7108841, 325.0086060, -156.2562714, 532.7672119, -626.4780884, 481.2648926
2: -51.2123947, 294.7198486, -85.3621979, 481.2583313, -532.4707031, 380.0820312
3: -69.1507874, 441.5514221, -113.7864914, 728.4608765, -797.6116943, 555.3378906
4: -92.4588852, 358.1494446, -152.6455383, 588.0037231, -680.4625244, 510.7949829

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3346821, upper bound: 547.3338918
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3352266, upper bound: 547.3353136
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -142.5798035, 560.8112793, -168.4913940, 659.1652222, -801.7449951, 729.3026123
1: -92.5000381, 320.9349060, -109.4170151, 377.0559082, -469.5559082, 430.3519287
2: -50.5578346, 291.0006409, -59.7179184, 341.8567505, -392.4145813, 350.7185669
3: -68.2978973, 435.9991760, -80.1343842, 513.5354614, -581.8333740, 516.1335449
4: -91.3173370, 353.6139832, -107.2683029, 416.2887878, -507.6061401, 460.8822021

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 14

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_A2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364177, upper bound: 547.3361778
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_A2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364177, upper bound: 547.3361778
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -145.2854767, 571.3545532, -238.3710480, 929.2578125, -1074.5432129, 809.7255859
1: -94.2708054, 326.9363098, -156.2562714, 532.7672119, -627.0380249, 483.1925659
2: -51.5100861, 296.4004822, -85.3621979, 481.2583313, -532.7684326, 381.7626038
3: -69.5904465, 444.2435608, -113.7864914, 728.4608765, -798.0513306, 558.0299683
4: -93.0584793, 360.2657776, -152.6455383, 588.0037231, -681.0620728, 512.9113159

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3357306, upper bound: 547.3346249
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3364177, upper bound: 547.3361778
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -172.0180054, 672.7317505, -117.5099258, 461.5703430, -633.5883789, 790.2416382
1: -111.9222412, 385.0429688, -76.7350616, 264.2547302, -376.1769409, 461.7780151
2: -61.0769844, 348.8662415, -41.8458023, 239.1187439, -300.1957092, 390.7120361
3: -81.8933868, 524.8155518, -56.1830330, 359.8187866, -441.7121582, 580.9985962
4: -109.7489929, 425.1504822, -75.5324173, 290.9871826, -400.7361145, 500.6828918

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3349476, upper bound: 547.3348724
time: 0.71 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 22

## BFS NS instance: NS_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -172.0180054, 672.7317505, -126.2806396, 494.5905151, -666.6085205, 799.0123291
1: -111.9222412, 385.0429688, -82.4975433, 283.9230347, -395.8452454, 467.5404968
2: -61.0769844, 348.8662415, -44.9068184, 256.8228455, -317.8997498, 393.7730103
3: -81.8933868, 524.8155518, -60.4459000, 386.7215881, -468.6149902, 585.2614746
4: -109.7489929, 425.1504822, -81.3113708, 312.3058167, -422.0547791, 506.4618530

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A1_A1_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3349476, upper bound: 547.3351815
time: 0.68 seconds

## Relational analysis of NS_A2_B1_B1_A1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 22

## BFS NS instance: NS_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -159.6043091, 625.1950073, -117.5099258, 461.5703430, -621.1746826, 742.7048340
1: -104.4380341, 358.5418701, -76.7350616, 264.2547302, -368.6927490, 435.2769165
2: -56.8458138, 324.5385132, -41.8458023, 239.1187439, -295.9645691, 366.3843079
3: -76.0734711, 489.2314453, -56.1830330, 359.8187866, -435.8922729, 545.4144897
4: -102.4814453, 395.4103394, -75.5324173, 290.9871826, -393.4686279, 470.9427490

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3349338, upper bound: 547.3349270
time: 0.71 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3342485, upper bound: 547.3344064
time: 0.69 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B1_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3342485, upper bound: 547.3351287
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -159.6043091, 625.1950073, -126.2806396, 494.5905151, -654.1948242, 751.4754639
1: -104.4380341, 358.5418701, -82.4975433, 283.9230347, -388.3610840, 441.0394287
2: -56.8458138, 324.5385132, -44.9068184, 256.8228455, -313.6686401, 369.4453125
3: -76.0734711, 489.2314453, -60.4459000, 386.7215881, -462.7950439, 549.6773682
4: -102.4814453, 395.4103394, -81.3113708, 312.3058167, -414.7872620, 476.7217102

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 32

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3349338, upper bound: 547.3352323
time: 0.77 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_A1

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3342485, upper bound: 547.3347150
time: 0.74 seconds

## Relational analysis of NS_A2_B1_B1_A1_A2_B2_A2

### Relational analysis result of NS_A2_B1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3342485, upper bound: 547.3354290
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -233.9964142, 909.6813965, -124.0638046, 486.8692932, -720.8656006, 1033.7452393
1: -153.2933807, 523.8526001, -80.7336655, 278.8927612, -432.1861572, 604.5861816
2: -83.6931381, 473.4965515, -44.0090332, 252.5409851, -336.2341309, 517.5056152
3: -111.3346481, 716.0031738, -59.2365875, 379.4322510, -490.7669067, 775.2396851
4: -149.5280151, 578.4881592, -79.5154419, 307.0363770, -456.5643921, 658.0036011

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A2_A1_A1_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3335342, upper bound: 547.3344556
time: 0.66 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_A1_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3335342, upper bound: 547.3347284
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -221.7172546, 861.8583984, -121.6602554, 477.4425659, -699.1597900, 983.5186768
1: -145.7553101, 497.2147827, -79.2030640, 273.5027466, -419.2580566, 576.4178467
2: -79.3978806, 449.0965881, -43.1562347, 247.6395874, -327.0374146, 492.2528076
3: -105.4179688, 680.1787720, -58.0847855, 372.1412964, -477.5592651, 738.2635498
4: -142.1562500, 548.6702271, -77.9917221, 301.0716553, -443.2278137, 626.6619263

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_B1

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3335342, upper bound: 547.3345010
time: 0.78 seconds

## Relational analysis of NS_A2_B1_B1_A2_A1_A2_B2

### Relational analysis result of NS_A2_B1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3335342, upper bound: 547.3347737
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -241.3837585, 940.8493652, -126.8096313, 497.6323547, -739.0160522, 1067.6589355
1: -158.4499664, 539.6751099, -82.5704880, 284.9830627, -443.4329834, 622.2456055
2: -86.5554886, 487.2823181, -45.0375786, 258.0229492, -344.5784302, 532.3198853
3: -115.2975082, 738.2872925, -60.5876312, 387.7551880, -503.0527039, 798.8748779
4: -154.8145142, 595.7329102, -81.3109894, 313.7564087, -468.5709229, 677.0438843

Time for backsubstitution: 1.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A2_B1_B1_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_B1_A2_A2_A1_A1

### Relational analysis result of NS_A2_B1_B1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3350471, upper bound: 547.3348377
time: 0.90 seconds

## Relational analysis of NS_A2_B1_B1_A2_A2_A1_A2

### Relational analysis result of NS_A2_B1_B1_A2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -547.3348776, upper bound: 547.3348365
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_B1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -229.6193695, 895.0529785, -124.3716202, 488.0603638, -717.6797485, 1019.4246216
1: -151.2301331, 514.0949097, -81.0175323, 279.5207520, -430.7508545, 595.1124268
2: -82.4409027, 463.8516541, -44.1724472, 253.0618896, -335.5027771, 508.0240784
3: -109.6196365, 703.8653564, -59.4197464, 380.3572083, -489.9768372, 763.2850952
4: -147.6854553, 567.2424316, -79.7635040, 307.7118835, -455.3972473, 647.0059204

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 22

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_B1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.84 + 416.30 = 420.14 seconds
