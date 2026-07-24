## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_9.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 2.87951805


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.9435281, 2.0473528, -1.9435281, 2.0473528, -3.9908810, 3.9908807)
1: (-14.0070305, 4.5551777, -14.0070305, 4.5551777, -18.5622082, 18.5622082)
2: (-7.3652477, 4.6843076, -7.3652477, 4.6843076, -12.0495548, 12.0495548)
3: (-9.4486265, 3.4397838, -9.4486265, 3.4397838, -12.8884106, 12.8884106)
4: (-5.1081634, 3.8832443, -5.1081634, 3.8832443, -8.9914055, 8.9914055)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.73 + 1.52 = 3.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -3.1994645, upper bound: 3.1994645

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1988951, upper bound: 3.1985747
time: 0.53 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1988951, upper bound: 3.1994500
time: 0.77 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.45 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.45
Output dim: 0, lower bound: -3.1988951, upper bound: 3.1985747
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.45
Output dim: 0, lower bound: -3.1988951, upper bound: 3.1994500

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2.1213088, 2.1787734, -1.9261450, 2.0277045, -4.1490135, 4.1049185
1: -14.8673372, 4.9773970, -13.8772249, 4.5137887, -19.3811245, 18.8546219
2: -7.9877892, 5.0327754, -7.2978525, 4.6415148, -12.6293030, 12.3306274
3: -10.1005116, 3.7247717, -9.3617010, 3.4083967, -13.5089073, 13.0864725
4: -5.5334530, 4.1442666, -5.0629334, 3.8441474, -9.3775988, 9.2072001

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1860130, upper bound: 3.1921458
time: 0.48 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1980815, upper bound: 3.1979272
time: 0.45 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1.9334546, 2.0380311, -1.9435281, 2.0473528, -3.9808073, 3.9815593
1: -13.9576826, 4.5331197, -14.0070305, 4.5551777, -18.5128593, 18.5401497
2: -7.3311300, 4.6637788, -7.3652477, 4.6843076, -12.0154362, 12.0290260
3: -9.4111109, 3.4248106, -9.4486265, 3.4397838, -12.8508949, 12.8734360
4: -5.0822468, 3.8663940, -5.1081634, 3.8832443, -8.9654875, 8.9745569

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0158260, upper bound: 3.0083021
time: 0.61 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9502864, upper bound: 2.9502864
time: 0.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.07 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.07
Output dim: 0, lower bound: -3.1860130, upper bound: 3.1921458
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.07
Output dim: 0, lower bound: -3.1980815, upper bound: 3.1979272
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.07
Output dim: 0, lower bound: -3.0158260, upper bound: 3.0083021
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.07
Output dim: 0, lower bound: -2.9502864, upper bound: 2.9502864

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -1.3349209, 1.5186436, -1.6938899, 1.8315389, -3.1664591, 3.2125335
1: -11.2477865, 3.2469873, -12.8497686, 4.0141029, -15.2618876, 16.0967560
2: -5.3283162, 3.4665937, -6.5294662, 4.1885080, -9.5168247, 9.9960594
3: -7.2802510, 2.6008565, -8.5586472, 3.0841200, -10.3643713, 11.1595039
4: -3.5412154, 2.9175963, -4.4667511, 3.4913642, -7.0325794, 7.3843474

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1810530, upper bound: 3.1810845
time: 0.63 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1810530, upper bound: 3.1921458
time: 0.51 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -1.7701690, 1.8591411, -1.8042581, 1.9206820, -3.6908507, 3.6633992
1: -13.2305899, 4.2308602, -13.3354750, 4.2590227, -17.4896126, 17.5663357
2: -6.8237677, 4.3444471, -6.9057693, 4.4055033, -11.2292681, 11.2502165
3: -8.8474836, 3.2288947, -8.9413738, 3.2397654, -12.0872488, 12.1702690
4: -4.6314192, 3.5444698, -4.7499304, 3.6564410, -8.2878599, 8.2944002

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1935591, upper bound: 3.1874308
time: 0.57 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1935591, upper bound: 3.1979272
time: 0.51 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -1.9220721, 2.0263727, -1.9435281, 2.0473528, -3.9694240, 3.9699004
1: -13.8792534, 4.5051818, -14.0070305, 4.5551777, -18.4344311, 18.5122089
2: -7.2866850, 4.6374779, -7.3652477, 4.6843076, -11.9709930, 12.0027256
3: -9.3558979, 3.4047275, -9.4486265, 3.4397838, -12.7956820, 12.8533535
4: -5.0532274, 3.8445261, -5.1081634, 3.8832443, -8.9364719, 8.9526892

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9849013, upper bound: 2.9770458
time: 0.50 seconds

## Relational analysis of NS_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0022766, upper bound: 2.9950744
time: 0.55 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0005733, upper bound: 2.9937537
time: 0.51 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -2.6896131, 2.6356165, -1.8352126, 1.9538312, -4.6434441, 4.4708290
1: -16.2351837, 6.0631890, -13.4819002, 4.3207245, -20.5559082, 19.5450859
2: -9.4925098, 5.9271960, -7.0039692, 4.4659581, -13.9584665, 12.9311638
3: -11.3058796, 4.3154712, -9.0458164, 3.2822721, -14.5881519, 13.3612862
4: -6.8836098, 4.9848065, -4.8290610, 3.7184114, -10.6020193, 9.8138676

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9502864, upper bound: 2.9502864
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9502864, upper bound: 2.9502864
time: 0.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.63 seconds
NS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -3.1810530, upper bound: 3.1810845
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -3.1810530, upper bound: 3.1921458
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -3.1935591, upper bound: 3.1874308
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -3.1935591, upper bound: 3.1979272
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -3.0022766, upper bound: 2.9950744
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -3.0005733, upper bound: 2.9937537
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -2.9502864, upper bound: 2.9502864
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 2.63
Output dim: 0, lower bound: -2.9502864, upper bound: 2.9502864

## BFS NS instance: NS_A1_A1_B1

### Backsubstitution after applying NS history:
0: -1.3349209, 1.5186436, -1.2860099, 1.5389785, -2.8738992, 2.8046534
1: -11.2477865, 3.2469873, -11.5187521, 3.1315110, -14.3792973, 14.7657394
2: -5.3283162, 3.4665937, -5.2274418, 3.4460998, -8.7744160, 8.6940355
3: -7.2802510, 2.6008565, -7.3820977, 2.5598743, -9.8401251, 9.9829540
4: -3.5412154, 2.9175963, -3.4740272, 2.9503121, -6.4915266, 6.3916230

Time for backsubstitution: 1.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1798722, upper bound: 3.1798722
time: 0.47 seconds

## Relational analysis of NS_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1798722, upper bound: 3.1810845
time: 0.75 seconds

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -1.3349209, 1.5186436, -1.4598157, 1.6357009, -2.9706216, 2.9784594
1: -11.2477865, 3.2469873, -11.7198057, 3.5114472, -14.7592335, 14.9667931
2: -5.3283162, 3.4665937, -5.7456656, 3.7168524, -9.0451689, 9.2122593
3: -7.2802510, 2.6008565, -7.7010450, 2.7486060, -10.0288572, 10.3019018
4: -3.5412154, 2.9175963, -3.8724203, 3.1371367, -6.6783524, 6.7900162

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1798722, upper bound: 3.1918055
time: 0.46 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1798722, upper bound: 3.1921458
time: 0.61 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -1.7701690, 1.8591411, -1.2860099, 1.5389785, -3.3091474, 3.1451504
1: -13.2305899, 4.2308602, -11.5187521, 3.1315110, -16.3621006, 15.7496128
2: -6.8237677, 4.3444471, -5.2274418, 3.4460998, -10.2698669, 9.5718889
3: -8.8474836, 3.2288947, -7.3820977, 2.5598743, -11.4073572, 10.6109924
4: -4.6314192, 3.5444698, -3.4740272, 2.9503121, -7.5817308, 7.0184970

Time for backsubstitution: 1.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1918055, upper bound: 3.1855311
time: 0.42 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1918055, upper bound: 3.1874308
time: 0.55 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -1.7701690, 1.8591411, -1.4598157, 1.6357009, -3.4058700, 3.3189564
1: -13.2305899, 4.2308602, -11.7198057, 3.5114472, -16.7420368, 15.9506664
2: -6.8237677, 4.3444471, -5.7456656, 3.7168524, -10.5406189, 10.0901108
3: -8.8474836, 3.2288947, -7.7010450, 2.7486060, -11.5960894, 10.9299393
4: -4.6314192, 3.5444698, -3.8724203, 3.1371367, -7.7685556, 7.4168901

Time for backsubstitution: 1.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1918055, upper bound: 3.1974644
time: 0.60 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1918055, upper bound: 3.1975419
time: 0.54 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -1.9220721, 2.0263727, -1.9251477, 2.0266030, -3.9486742, 3.9515200
1: -13.8792534, 4.5051818, -13.8458185, 4.5031648, -18.3824177, 18.3509979
2: -7.2866850, 4.6374779, -7.2787046, 4.6328998, -11.9195843, 11.9161825
3: -9.3558979, 3.4047275, -9.3393803, 3.4006855, -12.7565832, 12.7441072
4: -5.0532274, 3.8445261, -5.0584903, 3.8413539, -8.8945808, 8.9030161

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9906601, upper bound: 2.9825135
time: 0.54 seconds

## Relational analysis of NS_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9447863, upper bound: 2.9461303
time: 0.52 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -1.8393987, 1.9637308, -2.6540117, 2.7679081, -4.6073070, 4.6177421
1: -13.5945253, 4.3393850, -17.9867840, 6.1723847, -19.7669106, 22.3261662
2: -7.0390635, 4.4954371, -9.7749577, 6.2292504, -13.2683125, 14.2703953
3: -9.1152821, 3.3056707, -12.3311377, 4.5841088, -13.6993904, 15.6368084
4: -4.8455300, 3.7324488, -6.8970947, 5.2046299, -10.0501595, 10.6295424

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0005733, upper bound: 2.9937537
time: 0.51 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0005733, upper bound: 2.9937537
time: 0.58 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -2.6896131, 2.6356165, -1.9307721, 2.0346673, -4.7242804, 4.5663886
1: -16.2351837, 6.0631890, -13.9230022, 4.5244322, -20.7596169, 19.9861908
2: -9.4925098, 5.9271960, -7.3163295, 4.6555662, -14.1480761, 13.2435226
3: -11.3058796, 4.3154712, -9.3885994, 3.4179974, -14.7238750, 13.7040682
4: -6.8836098, 4.9848065, -5.0755973, 3.8596251, -10.7432337, 10.0604038

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9116224, upper bound: 2.9123700
time: 0.50 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9055274, upper bound: 2.9055274
time: 0.58 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2.6896131, 2.6356165, -2.7000341, 2.6453857, -5.3349991, 5.3356504
1: -16.2351837, 6.0631890, -16.2870407, 6.0864477, -22.3216324, 22.3502293
2: -9.4925098, 5.9271960, -9.5287228, 5.9486632, -15.4411736, 15.4559174
3: -11.3058796, 4.3154712, -11.3451939, 4.3309646, -15.6368446, 15.6606655
4: -6.8836098, 4.9848065, -6.9104242, 5.0029554, -11.8865652, 11.8952312

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9116224, upper bound: 2.9123700
time: 0.57 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9055274, upper bound: 2.9055274
time: 0.51 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.65 seconds
NS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -3.1798722, upper bound: 3.1798722
NS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -3.1798722, upper bound: 3.1810845
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -3.1798722, upper bound: 3.1918055
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -3.1798722, upper bound: 3.1921458
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -3.1918055, upper bound: 3.1855311
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -3.1918055, upper bound: 3.1874308
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -3.1918055, upper bound: 3.1974644
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -3.1918055, upper bound: 3.1975419
NS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -2.9906601, upper bound: 2.9825135
NS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -2.9447863, upper bound: 2.9461303
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -3.0005733, upper bound: 2.9937537
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -3.0005733, upper bound: 2.9937537
NS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -2.9116224, upper bound: 2.9123700
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -2.9055274, upper bound: 2.9055274
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -2.9116224, upper bound: 2.9123700
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.65
Output dim: 0, lower bound: -2.9055274, upper bound: 2.9055274

## BFS NS instance: NS_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1.3349209, 1.5186436, -1.3188236, 1.5078458, -2.8427660, 2.8374672
1: -11.2477865, 3.2469873, -11.1919317, 3.2141700, -14.4619560, 14.4389191
2: -5.3283162, 3.4665937, -5.2784224, 3.4400685, -8.7683840, 8.7450161
3: -7.2802510, 2.6008565, -7.2318840, 2.5812876, -9.8615379, 9.8327408
4: -3.5412154, 2.9175963, -3.5033674, 2.8986609, -6.4398756, 6.4209638

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## BFS NS instance: NS_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1.3349209, 1.5186436, -1.2942671, 1.5527681, -2.8876877, 2.8129106
1: -11.2477865, 3.2469873, -11.6150370, 3.1514075, -14.3991938, 14.8620234
2: -5.3283162, 3.4665937, -5.2630239, 3.4723229, -8.8006392, 8.7296162
3: -7.2802510, 2.6008565, -7.4421010, 2.5789058, -9.8591566, 10.0429573
4: -3.5412154, 2.9175963, -3.4972386, 2.9769883, -6.5182028, 6.4148345

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1.3349209, 1.5186436, -1.7422082, 1.8350716, -3.1699920, 3.2608519
1: -11.2477865, 3.2469873, -13.1207714, 4.1725268, -15.4203129, 16.3677597
2: -5.3283162, 3.4665937, -6.7326460, 4.2947063, -9.6230221, 10.1992397
3: -7.2802510, 2.6008565, -8.7552519, 3.1917338, -10.4719849, 11.3561087
4: -3.5412154, 2.9175963, -4.5626755, 3.5016878, -7.0429020, 7.4802718

Time for backsubstitution: 0.97 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1.3349209, 1.5186436, -1.4676726, 1.6474935, -2.9824140, 2.9863162
1: -11.2477865, 3.2469873, -11.7992563, 3.5307832, -14.7785683, 15.0462437
2: -5.3283162, 3.4665937, -5.7789025, 3.7390904, -9.0674067, 9.2454958
3: -7.2802510, 2.6008565, -7.7517548, 2.7654195, -10.0456705, 10.3526115
4: -3.5412154, 2.9175963, -3.8944521, 3.1599698, -6.7011852, 6.8120484

Time for backsubstitution: 1.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1.7701690, 1.8591411, -1.3188236, 1.5078458, -3.2780144, 3.1779642
1: -13.2305899, 4.2308602, -11.1919317, 3.2141700, -16.4447594, 15.4227915
2: -6.8237677, 4.3444471, -5.2784224, 3.4400685, -10.2638359, 9.6228695
3: -8.8474836, 3.2288947, -7.2318840, 2.5812876, -11.4287701, 10.4607782
4: -4.6314192, 3.5444698, -3.5033674, 2.8986609, -7.5300798, 7.0478373

Time for backsubstitution: 1.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 10

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -1.7701690, 1.8591411, -1.2942671, 1.5527681, -3.3229365, 3.1534081
1: -13.2305899, 4.2308602, -11.6150370, 3.1514075, -16.3819981, 15.8458967
2: -6.8237677, 4.3444471, -5.2630239, 3.4723229, -10.2960911, 9.6074705
3: -8.8474836, 3.2288947, -7.4421010, 2.5789058, -11.4263897, 10.6709957
4: -4.6314192, 3.5444698, -3.4972386, 2.9769883, -7.6084075, 7.0417085

Time for backsubstitution: 1.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -1.7701690, 1.8591411, -1.7422082, 1.8350716, -3.6052406, 3.6013494
1: -13.2305899, 4.2308602, -13.1207714, 4.1725268, -17.4031143, 17.3516312
2: -6.8237677, 4.3444471, -6.7326460, 4.2947063, -11.1184740, 11.0770931
3: -8.8474836, 3.2288947, -8.7552519, 3.1917338, -12.0392170, 11.9841461
4: -4.6314192, 3.5444698, -4.5626755, 3.5016878, -8.1331072, 8.1071453

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 31

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -1.7701690, 1.8591411, -1.4676726, 1.6474935, -3.4176624, 3.3268132
1: -13.2305899, 4.2308602, -11.7992563, 3.5307832, -16.7613716, 16.0301170
2: -6.8237677, 4.3444471, -5.7789025, 3.7390904, -10.5628576, 10.1233492
3: -8.8474836, 3.2288947, -7.7517548, 2.7654195, -11.6129036, 10.9806490
4: -4.6314192, 3.5444698, -3.8944521, 3.1599698, -7.7913890, 7.4389219

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.8129830, 1.9389775, -1.9153241, 2.0185990, -3.8315821, 3.8543015
1: -13.4384003, 4.2745852, -13.8026924, 4.4822054, -17.9206047, 18.0772781
2: -6.9212117, 4.4285445, -7.2453585, 4.6140461, -11.5352573, 11.6739025
3: -8.9884882, 3.2641001, -9.3042870, 3.3875411, -12.3760290, 12.5683870
4: -4.7688537, 3.6727877, -5.0325994, 3.8258624, -8.5947151, 8.7053843

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_B1_A1_A1

### Relational analysis result of NS_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9906601, upper bound: 2.9825135
time: 0.59 seconds

## Relational analysis of NS_A2_A1_B1_A1_A2

### Relational analysis result of NS_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9906601, upper bound: 2.9825135
time: 0.61 seconds

## BFS NS instance: NS_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.1293900, 2.2349179, -1.7652862, 1.9015038, -4.0308938, 4.0002041
1: -14.5943747, 4.8916764, -13.1317959, 4.1535931, -18.7479668, 18.0234718
2: -7.7622652, 5.0148454, -6.7152748, 4.3227682, -12.0850315, 11.7301197
3: -9.8308678, 3.6656387, -8.7604294, 3.1801939, -13.0110617, 12.4260674
4: -5.5195856, 4.2246475, -4.6356406, 3.6172650, -9.1368484, 8.8602867

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9447863, upper bound: 2.9461303
time: 0.66 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9447863, upper bound: 2.9461303
time: 0.63 seconds

## BFS NS instance: NS_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1.8393987, 1.9637308, -2.6458046, 2.7587752, -4.5981731, 4.6095343
1: -13.5945253, 4.3393850, -17.9170513, 6.1506925, -19.7452183, 22.2564316
2: -7.0390635, 4.4954371, -9.7384319, 6.2080016, -13.2470636, 14.2338696
3: -9.1152821, 3.3056707, -12.2838078, 4.5681729, -13.6834545, 15.5894785
4: -4.8455300, 3.7324488, -6.8752337, 5.1878710, -10.0333996, 10.6076803

Time for backsubstitution: 1.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0005733, upper bound: 2.9937537
time: 0.49 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0005733, upper bound: 2.9937537
time: 0.62 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1.8393987, 1.9637308, -2.1483908, 2.3201027, -4.1595016, 4.1121202
1: -13.5945253, 4.3393850, -15.4047813, 5.0400186, -18.6345444, 19.7441654
2: -7.0390635, 4.4954371, -7.9953690, 5.1692677, -12.2083302, 12.4908066
3: -9.1152821, 3.3056707, -10.3681421, 3.8439002, -12.9591827, 13.6738129
4: -4.8455300, 3.7324488, -5.5482206, 4.3784294, -9.2239590, 9.2806692

Time for backsubstitution: 0.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9882491, upper bound: 2.9784832
time: 0.46 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9401950, upper bound: 2.9393100
time: 0.52 seconds

## BFS NS instance: NS_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.0833223, 2.1455822, -1.7006555, 1.8408617, -3.9241838, 3.8462377
1: -14.2000761, 4.7480178, -12.9049931, 4.0289664, -18.2290421, 17.6530094
2: -7.5650716, 4.8196573, -6.5549712, 4.2062535, -11.7713242, 11.3746281
3: -9.5536375, 3.5391467, -8.5941515, 3.0964940, -12.6501312, 12.1332970
4: -5.3569546, 4.0517697, -4.4844275, 3.5102279, -8.8671808, 8.5361958

Time for backsubstitution: 1.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_A1_A1

### Relational analysis result of NS_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9695306, upper bound: 2.9764384
time: 0.49 seconds

## Relational analysis of NS_A2_A2_B1_A1_A2

### Relational analysis result of NS_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9361975, upper bound: 2.9403219
time: 0.59 seconds

## BFS NS instance: NS_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.2594872, 2.2634561, -1.8098053, 1.9286797, -4.1881661, 4.0732603
1: -14.4038410, 5.1338015, -13.3851814, 4.2715273, -18.6753674, 18.5189800
2: -8.0932121, 5.1095738, -6.9274268, 4.4212623, -12.5144739, 12.0370007
3: -9.8800411, 3.7371893, -8.9723406, 3.2506132, -13.1306534, 12.7095280
4: -5.7751522, 4.2898674, -4.7649026, 3.6735415, -9.4486904, 9.0547695

Time for backsubstitution: 1.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_A2_A1

### Relational analysis result of NS_A2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9663558, upper bound: 2.9736170
time: 0.70 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2

### Relational analysis result of NS_A2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9701288, upper bound: 2.9760518
time: 0.50 seconds

## BFS NS instance: NS_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.0833223, 2.1455822, -2.4810145, 2.4569707, -4.5402923, 4.6265965
1: -14.2000761, 4.7480178, -15.3699923, 5.6047215, -19.8047981, 20.1180077
2: -7.5650716, 4.8196573, -8.7853794, 5.5279059, -13.0929775, 13.6050367
3: -9.5536375, 3.5391467, -10.6177053, 4.0322013, -13.5858383, 14.1568508
4: -5.3569546, 4.0517697, -6.3465481, 4.6479459, -10.0049000, 10.3983173

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9051538, upper bound: 2.9051538
time: 0.54 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9051538, upper bound: 2.9055029
time: 0.53 seconds

## BFS NS instance: NS_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.2594872, 2.2634561, -2.5894976, 2.5508728, -4.8103600, 4.8529520
1: -14.4038410, 5.1338015, -15.8313503, 5.8459487, -20.2497902, 20.9651489
2: -8.0932121, 5.1095738, -9.1599817, 5.7405429, -13.8337555, 14.2695551
3: -9.8800411, 3.7371893, -10.9848747, 4.1838026, -14.0638428, 14.7220631
4: -5.7751522, 4.2898674, -6.6262608, 4.8269958, -10.6021471, 10.9161272

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9055029, upper bound: 2.9051538
time: 0.57 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9055029, upper bound: 2.9055274
time: 0.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.63 seconds
NS_A2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -2.9906601, upper bound: 2.9825135
NS_A2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -2.9906601, upper bound: 2.9825135
NS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -2.9447863, upper bound: 2.9461303
NS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -2.9447863, upper bound: 2.9461303
NS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -3.0005733, upper bound: 2.9937537
NS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -3.0005733, upper bound: 2.9937537
NS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -2.9882491, upper bound: 2.9784832
NS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -2.9401950, upper bound: 2.9393100
NS_A2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -2.9695306, upper bound: 2.9764384
NS_A2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -2.9361975, upper bound: 2.9403219
NS_A2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -2.9663558, upper bound: 2.9736170
NS_A2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -2.9701288, upper bound: 2.9760518
NS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -2.9051538, upper bound: 2.9051538
NS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -2.9051538, upper bound: 2.9055029
NS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -2.9055029, upper bound: 2.9051538
NS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.63
Output dim: 0, lower bound: -2.9055029, upper bound: 2.9055274

## BFS NS instance: NS_A2_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -1.7959572, 1.9190217, -1.9153241, 2.0185990, -3.8145559, 3.8343458
1: -13.2782602, 4.2251544, -13.8026924, 4.4822054, -17.7604656, 18.0278473
2: -6.8384409, 4.3792624, -7.2453585, 4.6140461, -11.4524870, 11.6246204
3: -8.8802767, 3.2261412, -9.3042870, 3.3875411, -12.2678175, 12.5304279
4: -4.7219625, 3.6334414, -5.0325994, 3.8258624, -8.5478239, 8.6660376

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B1_A1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9906601, upper bound: 2.9825135
time: 0.52 seconds

## Relational analysis of NS_A2_A1_B1_A1_A1_B2

### Relational analysis result of NS_A2_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9906601, upper bound: 2.9825135
time: 0.58 seconds

## BFS NS instance: NS_A2_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -2.3728447, 2.5504818, -1.9153241, 2.0185990, -4.3914437, 4.4658055
1: -16.6400108, 5.5490460, -13.8026924, 4.4822054, -21.1222153, 19.3517361
2: -8.7922268, 5.7044611, -7.2453585, 4.6140461, -13.4062729, 12.9498196
3: -11.2823620, 4.1978235, -9.3042870, 3.3875411, -14.6699028, 13.5021095
4: -6.1612353, 4.7964120, -5.0325994, 3.8258624, -9.9870968, 9.8290119

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9744610, upper bound: 2.9716190
time: 0.53 seconds

## Relational analysis of NS_A2_A1_B1_A1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9900155, upper bound: 2.9824924
time: 0.53 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.1293900, 2.2349179, -1.7549748, 1.8903695, -4.0197597, 3.9898927
1: -14.5943747, 4.8916764, -13.0587168, 4.1280794, -18.7224541, 17.9503937
2: -7.7622652, 5.0148454, -6.6745663, 4.2985334, -12.0607986, 11.6894112
3: -9.8308678, 3.6656387, -8.7093115, 3.1616354, -12.9925032, 12.3749495
4: -5.5195856, 4.2246475, -4.6091156, 3.5969360, -9.1165209, 8.8337622

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9447863, upper bound: 2.9461303
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9447863, upper bound: 2.9461303
time: 0.47 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.1293900, 2.2349179, -2.5327098, 2.5164015, -4.6457915, 4.7676277
1: -14.5943747, 4.8916764, -15.3509846, 5.6931643, -20.2875385, 20.2426605
2: -7.7622652, 5.0148454, -8.8888292, 5.6209764, -13.3832407, 13.9036751
3: -9.8308678, 3.6656387, -10.6476154, 4.0716743, -13.9025402, 14.3132534
4: -5.5195856, 4.2246475, -6.4740644, 4.7529926, -10.2725773, 10.6987114

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9350177, upper bound: 2.9369849
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9447077, upper bound: 2.9454682
time: 0.54 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -1.9039500, 2.0058410, -2.6458046, 2.7587752, -4.6627245, 4.6516457
1: -13.7189922, 4.4536676, -17.9170513, 6.1506925, -19.8696842, 22.3707142
2: -7.2008629, 4.5865922, -9.7384319, 6.2080016, -13.4088640, 14.3250237
3: -9.2474880, 3.3659515, -12.2838078, 4.5681729, -13.8156586, 15.6497593
4: -5.0041809, 3.8030198, -6.8752337, 5.1878710, -10.1920490, 10.6782503

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_B1_A1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1910693, upper bound: 3.1911738
time: 0.62 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1918094, upper bound: 3.1918095
time: 0.82 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -2.6357431, 2.7493997, -2.6458046, 2.7587752, -5.3945169, 5.3952045
1: -17.8663235, 6.1284413, -17.9170513, 6.1506925, -24.0170155, 24.0454922
2: -9.7037020, 6.1873755, -9.7384319, 6.2080016, -15.9117031, 15.9258080
3: -12.2456245, 4.5530963, -12.2838078, 4.5681729, -16.8137951, 16.8369045
4: -6.8494673, 5.1703362, -6.8752337, 5.1878710, -12.0373363, 12.0455675

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_B1_A2_A1

### Relational analysis result of NS_A2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0159639, upper bound: 3.0058669
time: 0.48 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9560975, upper bound: 2.9560975
time: 0.50 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1.7369432, 1.8790492, -2.1401753, 2.3133917, -4.0503349, 4.0192232
1: -13.1850300, 4.1176491, -15.3669500, 5.0218735, -18.2069035, 19.4846001
2: -6.6866794, 4.2947388, -7.9661827, 5.1530037, -11.8396807, 12.2609215
3: -8.7718611, 3.1723588, -10.3377438, 3.8326349, -12.6044960, 13.5101013
4: -4.5826802, 3.5669212, -5.5266538, 4.3651185, -8.9477987, 9.0935745

Time for backsubstitution: 1.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_B2_B2_A1_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9882491, upper bound: 2.9784832
time: 0.50 seconds

## Relational analysis of NS_A2_A1_B2_B2_A1_A2

### Relational analysis result of NS_A2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9882491, upper bound: 2.9784832
time: 0.74 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -2.0554333, 2.1841476, -2.0096445, 2.2189198, -4.2743526, 4.1937923
1: -14.3743000, 4.7521515, -14.7633963, 4.7296729, -19.1039734, 19.5155487
2: -7.5562272, 4.8977270, -7.4906263, 4.8983436, -12.4545689, 12.3883533
3: -9.6404610, 3.5852549, -9.8512650, 3.6509440, -13.2914047, 13.4365187
4: -5.3387203, 4.1287117, -5.1797986, 4.1852674, -9.5239878, 9.3085098

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_B2_B2_A2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9401950, upper bound: 2.9393100
time: 0.49 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9401950, upper bound: 2.9393100
time: 0.48 seconds

## BFS NS instance: NS_A2_A2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -2.0678477, 2.1272078, -1.7006555, 1.8408617, -3.9087095, 3.8278632
1: -14.0492592, 4.7090015, -12.9049931, 4.0289664, -18.0782261, 17.6139946
2: -7.4866199, 4.7734518, -6.5549712, 4.2062535, -11.6928711, 11.3284216
3: -9.4518328, 3.5033979, -8.5941515, 3.0964940, -12.5483265, 12.0975485
4: -5.3139734, 4.0146866, -4.4844275, 3.5102279, -8.8241997, 8.4991121

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_A1_A1_B1

### Relational analysis result of NS_A2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9565138, upper bound: 2.9629371
time: 0.60 seconds

## Relational analysis of NS_A2_A2_B1_A1_A1_B2

### Relational analysis result of NS_A2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9125602, upper bound: 2.9100332
time: 0.66 seconds

## BFS NS instance: NS_A2_A2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -2.1822925, 2.3305235, -1.6227819, 1.7852201, -3.9675126, 3.9533052
1: -15.2766113, 5.0384221, -12.6406565, 3.8683457, -19.1449547, 17.6790791
2: -7.9910593, 5.1700749, -6.3172922, 4.0699444, -12.0610018, 11.4873676
3: -10.2413445, 3.8145323, -8.3669548, 3.0020125, -13.2433567, 12.1814852
4: -5.6203971, 4.3862247, -4.2963929, 3.4065793, -9.0269766, 8.6826172

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_A1_A2_B1

### Relational analysis result of NS_A2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9206896, upper bound: 2.9246245
time: 0.52 seconds

## Relational analysis of NS_A2_A2_B1_A1_A2_B2

### Relational analysis result of NS_A2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8856543, upper bound: 2.8840707
time: 0.44 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -2.2408328, 2.2427104, -1.8098053, 1.9286797, -4.1695118, 4.0525150
1: -14.2440710, 5.0820642, -13.3851814, 4.2715273, -18.5155983, 18.4672451
2: -8.0058117, 5.0579734, -6.9274268, 4.4212623, -12.4270716, 11.9854002
3: -9.7713165, 3.6978149, -8.9723406, 3.2506132, -13.0219297, 12.6701546
4: -5.7251172, 4.2483997, -4.7649026, 3.6735415, -9.3986568, 9.0133018

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_A2_A1_B1

### Relational analysis result of NS_A2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9505135, upper bound: 2.9594961
time: 1.03 seconds

## Relational analysis of NS_A2_A2_B1_A2_A1_B2

### Relational analysis result of NS_A2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9074824, upper bound: 2.9062682
time: 0.48 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -2.6423931, 2.7257440, -1.7371800, 1.8760290, -4.5184212, 4.4629240
1: -16.9941216, 6.0547299, -13.1358700, 4.1192489, -21.1133690, 19.1905937
2: -9.4413252, 6.0478277, -6.7008786, 4.2919807, -13.7333059, 12.7487040
3: -11.6565275, 4.4472141, -8.7572212, 3.1615908, -14.8181181, 13.2044315
4: -6.7116580, 5.1233096, -4.5860085, 3.5739474, -10.2856054, 9.7093172

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9502666, upper bound: 2.9599122
time: 0.55 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2_B2

### Relational analysis result of NS_A2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9041972, upper bound: 2.9049984
time: 0.51 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.0833223, 2.1455822, -2.0934548, 2.1546290, -4.2379513, 4.2390370
1: -14.2000761, 4.7480178, -14.2446384, 4.7703662, -18.9704418, 18.9926567
2: -7.5650716, 4.8196573, -7.5984497, 4.8396955, -12.4047670, 12.4181061
3: -9.5536375, 3.5391467, -9.5880299, 3.5535471, -13.1071844, 13.1271753
4: -5.3569546, 4.0517697, -5.3828845, 4.0688171, -9.4257717, 9.4346542

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8832720, upper bound: 2.8827417
time: 0.63 seconds

## Relational analysis of NS_A2_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_A2_B2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8635728, upper bound: 2.8635728
time: 0.50 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.0833223, 2.1455822, -2.2697210, 2.2730820, -4.3564043, 4.4153032
1: -14.2000761, 4.7480178, -14.4549141, 5.1565537, -19.3566303, 19.2029324
2: -7.5650716, 4.8196573, -8.1283484, 5.1307917, -12.6958637, 12.9480038
3: -9.5536375, 3.5391467, -9.9185419, 3.7525237, -13.3061609, 13.4576874
4: -5.3569546, 4.0517697, -5.8014603, 4.3078966, -9.6648502, 9.8532295

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8832720, upper bound: 2.8829231
time: 0.54 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8635728, upper bound: 2.8840964
time: 0.59 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.2594872, 2.2634561, -2.0934548, 2.1546290, -4.4141159, 4.3569107
1: -14.4038410, 5.1338015, -14.2446384, 4.7703662, -19.1742058, 19.3784409
2: -8.0932121, 5.1095738, -7.5984497, 4.8396955, -12.9329071, 12.7080231
3: -9.8800411, 3.7371893, -9.5880299, 3.5535471, -13.4335880, 13.3252182
4: -5.7751522, 4.2898674, -5.3828845, 4.0688171, -9.8439693, 9.6727524

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8639780, upper bound: 2.8635708
time: 0.60 seconds

## Relational analysis of NS_A2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8653327, upper bound: 2.8641592
time: 0.53 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.2594872, 2.2634561, -2.2697210, 2.2730820, -4.5325680, 4.5331774
1: -14.4038410, 5.1338015, -14.4549141, 5.1565537, -19.5603905, 19.5887127
2: -8.0932121, 5.1095738, -8.1283484, 5.1307917, -13.2240038, 13.2379198
3: -9.8800411, 3.7371893, -9.9185419, 3.7525237, -13.6325645, 13.6557312
4: -5.7751522, 4.2898674, -5.8014603, 4.3078966, -10.0830488, 10.0913277

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9033766, upper bound: 2.9029323
time: 0.57 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8653328, upper bound: 2.9047607
time: 0.58 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.97 seconds
NS_A2_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9906601, upper bound: 2.9825135
NS_A2_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9906601, upper bound: 2.9825135
NS_A2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9744610, upper bound: 2.9716190
NS_A2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9900155, upper bound: 2.9824924
NS_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9447863, upper bound: 2.9461303
NS_A2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9447863, upper bound: 2.9461303
NS_A2_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9350177, upper bound: 2.9369849
NS_A2_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9447077, upper bound: 2.9454682
NS_A2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -3.1910693, upper bound: 3.1911738
NS_A2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -3.1918094, upper bound: 3.1918095
NS_A2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -3.0159639, upper bound: 3.0058669
NS_A2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9560975, upper bound: 2.9560975
NS_A2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9882491, upper bound: 2.9784832
NS_A2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9882491, upper bound: 2.9784832
NS_A2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9401950, upper bound: 2.9393100
NS_A2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9401950, upper bound: 2.9393100
NS_A2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9565138, upper bound: 2.9629371
NS_A2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9125602, upper bound: 2.9100332
NS_A2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9206896, upper bound: 2.9246245
NS_A2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.8856543, upper bound: 2.8840707
NS_A2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9505135, upper bound: 2.9594961
NS_A2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9074824, upper bound: 2.9062682
NS_A2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9502666, upper bound: 2.9599122
NS_A2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9041972, upper bound: 2.9049984
NS_A2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.8832720, upper bound: 2.8827417
NS_A2_A2_B2_A1_B1_B2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.8635728, upper bound: 2.8635728
NS_A2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.8832720, upper bound: 2.8829231
NS_A2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.8635728, upper bound: 2.8840964
NS_A2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.8639780, upper bound: 2.8635708
NS_A2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.8653327, upper bound: 2.8641592
NS_A2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.9033766, upper bound: 2.9029323
NS_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.97
Output dim: 0, lower bound: -2.8653328, upper bound: 2.9047607

## BFS NS instance: NS_A2_A1_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -1.7959572, 1.9190217, -1.9040809, 2.0070453, -3.8030024, 3.8231022
1: -13.2782602, 4.2251544, -13.7247610, 4.4545255, -17.7327862, 17.9499149
2: -6.8384409, 4.3792624, -7.2012973, 4.5880661, -11.4265060, 11.5805578
3: -8.8802767, 3.2261412, -9.2495537, 3.3676190, -12.2478952, 12.4756947
4: -4.7219625, 3.6334414, -5.0039005, 3.8041811, -8.5261431, 8.6373425

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A1_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9881482, upper bound: 2.9861838
time: 0.50 seconds

## Relational analysis of NS_A2_A1_B1_A1_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0023003, upper bound: 2.9938073
time: 0.78 seconds

## BFS NS instance: NS_A2_A1_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -1.7959572, 1.9190217, -2.6750817, 2.6194918, -4.4154487, 4.5941029
1: -13.2782602, 4.2251544, -16.0992641, 6.0263038, -19.3045635, 20.3244171
2: -6.8384409, 4.3792624, -9.4290981, 5.8846731, -12.7231140, 13.8083591
3: -8.8802767, 3.2261412, -11.2136631, 4.2826376, -13.1629133, 14.4398041
4: -4.7219625, 3.6334414, -6.8438745, 4.9509997, -9.6729622, 10.4773159

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B1

### Relational analysis result of NS_A2_A1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9595238, upper bound: 2.9554012
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B2

### Relational analysis result of NS_A2_A1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9850570, upper bound: 2.9785356
time: 0.74 seconds

## BFS NS instance: NS_A2_A1_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -2.3550174, 2.5322535, -1.8496228, 1.9516122, -4.3066287, 4.3818765
1: -16.5332947, 5.5083947, -13.4200811, 4.3260489, -20.8593445, 18.9284744
2: -8.7309828, 5.6641741, -7.0109439, 4.4613056, -13.1922884, 12.6751156
3: -11.2074366, 4.1673164, -9.0306787, 3.2733746, -14.4808111, 13.1979952
4: -6.1186752, 4.7620635, -4.8672514, 3.6966228, -9.8152981, 9.6293144

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A1_A2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9743050, upper bound: 2.9708704
time: 0.60 seconds

## Relational analysis of NS_A2_A1_B1_A1_A2_B1_A2

### Relational analysis result of NS_A2_A1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9743050, upper bound: 2.9716190
time: 0.71 seconds

## BFS NS instance: NS_A2_A1_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -2.3449044, 2.5272570, -2.1299081, 2.2304029, -4.5753074, 4.6571651
1: -16.5224686, 5.4884405, -14.9557381, 4.9491320, -21.4715939, 20.4441795
2: -8.6993217, 5.6510386, -7.9431586, 5.0810685, -13.7803898, 13.5941973
3: -11.1883469, 4.1597919, -10.1578035, 3.7156720, -14.9040184, 14.3175955
4: -6.0913224, 4.7505078, -5.5885935, 4.2150021, -10.3063240, 10.3391018

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_A1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9523401, upper bound: 2.9490847
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_A1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9774858, upper bound: 2.9689760
time: 0.72 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.1092110, 2.2126124, -1.7549748, 1.8903695, -3.9995804, 3.9675872
1: -14.4327822, 4.8408551, -13.0587168, 4.1280794, -18.5608616, 17.8995724
2: -7.6697903, 4.9607201, -6.6745663, 4.2985334, -11.9683218, 11.6352863
3: -9.7186594, 3.6245968, -8.7093115, 3.1616354, -12.8802948, 12.3339081
4: -5.4652557, 4.1807156, -4.6091156, 3.5969360, -9.0621901, 8.7898302

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9935960, upper bound: 3.0021734
time: 0.54 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9935960, upper bound: 3.0021734
time: 0.56 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.7934542, 2.9153845, -1.7549748, 1.8903695, -4.6838236, 4.6703596
1: -18.2395229, 6.4129181, -13.0587168, 4.1280794, -22.3676033, 19.4716339
2: -10.0061789, 6.4529142, -6.6745663, 4.2985334, -14.3047123, 13.1274805
3: -12.4730673, 4.7330294, -8.7093115, 3.1616354, -15.6347027, 13.4423409
4: -7.1816974, 5.4638925, -4.6091156, 3.5969360, -10.7786322, 10.0730057

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9935960, upper bound: 3.0021734
time: 0.53 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9935960, upper bound: 3.0021734
time: 0.48 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -2.1109746, 2.2161088, -2.4724131, 2.4546075, -4.5655813, 4.6885214
1: -14.4878464, 4.8481565, -14.9953041, 5.5545402, -20.0423870, 19.8434601
2: -7.6958413, 4.9715037, -8.6756477, 5.4774966, -13.1733379, 13.6471519
3: -9.7543020, 3.6343887, -10.3921795, 3.9663358, -13.7206383, 14.0265675
4: -5.4732900, 4.1880951, -6.3221169, 4.6313901, -10.1046801, 10.5102100

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9114727, upper bound: 2.9119081
time: 0.62 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9347997, upper bound: 2.9367597
time: 0.82 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -2.0928807, 2.2076397, -2.6739552, 2.6736033, -4.7664838, 4.8815947
1: -14.4670286, 4.8187366, -16.1279907, 6.0188484, -20.4858761, 20.9467278
2: -7.6524057, 4.9516473, -9.3692913, 5.9500470, -13.6024513, 14.3209381
3: -9.7259016, 3.6208010, -11.2110310, 4.3031039, -14.0290051, 14.8318319
4: -5.4345155, 4.1710734, -6.8456731, 5.0370574, -10.4715719, 11.0167465

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A2_B2_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9322764, upper bound: 2.9318362
time: 0.62 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_B2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9321929, upper bound: 2.9325358
time: 0.55 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -1.8397497, 1.9400724, -2.6277161, 2.7395263, -4.5792761, 4.5677886
1: -13.3433590, 4.3021183, -17.8013191, 6.1072965, -19.4506531, 22.1034374
2: -6.9713058, 4.4367280, -9.6723394, 6.1656489, -13.1369524, 14.1090679
3: -8.9795017, 3.2538586, -12.2020245, 4.5369039, -13.5164051, 15.4558830
4: -4.8427324, 3.6758676, -6.8289614, 5.1520758, -9.9948082, 10.5048256

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1847639, upper bound: 3.1878984
time: 0.51 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_B1_A1_A1_B1

### Relational analysis result of NS_A2_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0663514, upper bound: 3.0712177
time: 0.48 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_A1_B2

### Relational analysis result of NS_A2_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9971101, upper bound: 2.9910866
time: 0.47 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -2.1192787, 2.2180717, -2.6157134, 2.7339842, -4.8532629, 4.8337841
1: -14.8751602, 4.9235916, -17.7920570, 6.0875754, -20.9627361, 22.7156487
2: -7.9000721, 5.0546503, -9.6429987, 6.1528578, -14.0529261, 14.6976490
3: -10.1030350, 3.6952286, -12.1856079, 4.5275812, -14.6306152, 15.8808346
4: -5.5616851, 4.1925569, -6.8029108, 5.1402907, -10.7019758, 10.9954672

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B2_B1_A1_A2_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1357538, upper bound: 3.1314000
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_A2_A2

### Relational analysis result of NS_A2_A1_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0976143, upper bound: 3.0982118
time: 0.63 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -2.5251026, 2.6602459, -2.6361778, 2.7510164, -5.2761178, 5.2964239
1: -17.3686161, 5.8911114, -17.8738518, 6.1300683, -23.4986820, 23.7649574
2: -9.3235998, 5.9766178, -9.7054644, 6.1896853, -15.5132847, 15.6820822
3: -11.8521128, 4.4046597, -12.2496490, 4.5551605, -16.4072704, 16.6543083
4: -6.5626884, 5.0010352, -6.8502684, 5.1731253, -11.7358131, 11.8513031

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A2_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9560975, upper bound: 2.9560975
time: 0.58 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9560975, upper bound: 2.9560975
time: 0.57 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -2.2124963, 2.4248025, -2.4810939, 2.6279359, -4.8404307, 4.9058962
1: -15.9951668, 5.1879196, -17.1552715, 5.7845197, -21.7796860, 22.3431911
2: -8.1758146, 5.3815579, -9.1439638, 5.8895912, -14.0654058, 14.5255222
3: -10.7052555, 3.9843285, -11.6749802, 4.3415513, -15.0468063, 15.6593075
4: -5.7329583, 4.5626316, -6.4428463, 4.9381318, -10.6710892, 11.0054779

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_A2_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9560975, upper bound: 2.9560975
time: 0.53 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_A2_B2

### Relational analysis result of NS_A2_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9560975, upper bound: 2.9560975
time: 0.58 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A1_A1

### Backsubstitution after applying NS history:
0: -1.7959572, 1.9190217, -2.1401753, 2.3133917, -4.1093483, 4.0591965
1: -13.2782602, 4.2251544, -15.3669500, 5.0218735, -18.3001328, 19.5921040
2: -6.8384409, 4.3792624, -7.9661827, 5.1530037, -11.9914417, 12.3454437
3: -8.8802767, 3.2261412, -10.3377438, 3.8326349, -12.7129116, 13.5638847
4: -4.7219625, 3.6334414, -5.5266538, 4.3651185, -9.0870800, 9.1600952

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_B2_A1_A1_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9773815, upper bound: 2.9704078
time: 0.59 seconds

## Relational analysis of NS_A2_A1_B2_B2_A1_A1_A2

### Relational analysis result of NS_A2_A1_B2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9875393, upper bound: 2.9784173
time: 0.52 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A1_A2

### Backsubstitution after applying NS history:
0: -2.3728447, 2.5504818, -2.1401753, 2.3133917, -4.6862364, 4.6906552
1: -16.6400108, 5.5490460, -15.3669500, 5.0218735, -21.6618824, 20.9159966
2: -8.7922268, 5.7044611, -7.9661827, 5.1530037, -13.9452286, 13.6706438
3: -11.2823620, 4.1978235, -10.3377438, 3.8326349, -15.1149969, 14.5355654
4: -6.1612353, 4.7964120, -5.5266538, 4.3651185, -10.5263529, 10.3230658

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_B2_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B2_B2_A1_A2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9790038, upper bound: 2.9691574
time: 0.67 seconds

## Relational analysis of NS_A2_A1_B2_B2_A1_A2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9780033, upper bound: 2.9681066
time: 0.88 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -2.1091456, 2.2125309, -2.0096445, 2.2189198, -4.3280654, 4.2221756
1: -14.4324675, 4.8407121, -14.7633963, 4.7296729, -19.1621399, 19.6041069
2: -7.6695867, 4.9605532, -7.4906263, 4.8983436, -12.5679274, 12.4511795
3: -9.7184286, 3.6244886, -9.8512650, 3.6509440, -13.3693724, 13.4757519
4: -5.4650998, 4.1805377, -5.1797986, 4.1852674, -9.6503649, 9.3603354

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_B2_A2_A1_A1

### Relational analysis result of NS_A2_A1_B2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9317742, upper bound: 2.9293122
time: 0.52 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_A1_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9399356, upper bound: 2.9392441
time: 0.61 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -2.7451444, 2.8868370, -2.0096445, 2.2189198, -4.9640622, 4.8964815
1: -17.9817410, 6.2977901, -14.7633963, 4.7296729, -22.7114143, 21.0611839
2: -9.8358545, 6.3788757, -7.4906263, 4.8983436, -14.7341976, 13.8695011
3: -12.2976103, 4.6581421, -9.8512650, 3.6509440, -15.9485483, 14.5094070
4: -7.0678420, 5.4120045, -5.1797986, 4.1852674, -11.2531090, 10.5918016

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_A2_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9292207, upper bound: 2.9297111
time: 0.51 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_B2

### Relational analysis result of NS_A2_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9399356, upper bound: 2.9392441
time: 0.76 seconds

## BFS NS instance: NS_A2_A2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -2.0612693, 2.1220024, -1.5973790, 1.7605880, -3.8218572, 3.7193813
1: -14.0231590, 4.6942406, -12.4944897, 3.8070154, -17.8301697, 17.1887302
2: -7.4641562, 4.7609158, -6.2060432, 4.0071645, -11.4713173, 10.9669590
3: -9.4306917, 3.4947269, -8.2526093, 2.9624693, -12.3931608, 11.7473354
4: -5.2970600, 4.0044351, -4.2234912, 3.3510537, -8.6481123, 8.2279263

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9287207, upper bound: 2.9289777
time: 0.49 seconds

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_A2

### Relational analysis result of NS_A2_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9564907, upper bound: 2.9624769
time: 0.45 seconds

## BFS NS instance: NS_A2_A2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -1.9833176, 2.0608768, -1.9116489, 2.0479732, -4.0312910, 3.9725256
1: -13.6093569, 4.5156498, -13.6810322, 4.4257345, -18.0350914, 18.1966820
2: -7.1625166, 4.6010995, -7.0505514, 4.5941458, -11.7566624, 11.6516504
3: -9.1050110, 3.3799171, -9.1101999, 3.3701062, -12.4751167, 12.4901152
4: -5.0826159, 3.8833764, -4.9589124, 3.8798699, -8.9624863, 8.8422871

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_A2_B1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B1_A1_A1_B2_A1

### Relational analysis result of NS_A2_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8952903, upper bound: 2.8925920
time: 0.53 seconds

## Relational analysis of NS_A2_A2_B1_A1_A1_B2_A2

### Relational analysis result of NS_A2_A2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9118273, upper bound: 2.9099582
time: 0.53 seconds

## BFS NS instance: NS_A2_A2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -2.1760516, 2.3254528, -1.5233233, 1.7074714, -3.8835230, 3.8487754
1: -15.2513132, 5.0243521, -12.2375393, 3.6507308, -18.9020443, 17.2618904
2: -7.9691935, 5.1579289, -5.9725041, 3.8769076, -11.8460999, 11.1304312
3: -10.2199030, 3.8061888, -8.0359402, 2.8714507, -13.0913515, 11.8421278
4: -5.6044588, 4.3764486, -4.0419426, 3.2540121, -8.8584700, 8.4183893

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9206896, upper bound: 2.9246245
time: 0.58 seconds

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9206896, upper bound: 2.9246245
time: 0.56 seconds

## BFS NS instance: NS_A2_A2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -2.0990887, 2.2650456, -1.8305641, 1.9900273, -4.0891147, 4.0956092
1: -14.8277655, 4.8458567, -13.4426231, 4.2733111, -19.1010742, 18.2884769
2: -7.6656880, 4.9996347, -6.8237357, 4.4634433, -12.1291313, 11.8233700
3: -9.8915491, 3.6923342, -8.9045506, 3.2796845, -13.1712341, 12.5968847
4: -5.3928647, 4.2588215, -4.7600226, 3.7751892, -9.1680536, 9.0188437

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B1

### Relational analysis result of NS_A2_A2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8856543, upper bound: 2.8840707
time: 0.68 seconds

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B2

### Relational analysis result of NS_A2_A2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8856543, upper bound: 2.8840707
time: 0.48 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -2.2344041, 2.2373164, -1.7104865, 1.8500137, -4.0844173, 3.9478030
1: -14.2142410, 5.0679126, -12.9893637, 4.0576301, -18.2718697, 18.0572758
2: -7.9826784, 5.0450783, -6.5864182, 4.2285590, -12.2112360, 11.6314964
3: -9.7465935, 3.6892681, -8.6405954, 3.1226335, -12.8692265, 12.3298607
4: -5.7081428, 4.2379875, -4.5102730, 3.5168526, -9.2249956, 8.7482605

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A1

### Relational analysis result of NS_A2_A2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9442392, upper bound: 2.9477114
time: 0.51 seconds

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9504592, upper bound: 2.9587228
time: 0.53 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -2.1516609, 2.1726165, -2.0268896, 2.1481004, -4.2997613, 4.1995058
1: -13.7794094, 4.8757076, -14.1655407, 4.6802516, -18.4596596, 19.0412445
2: -7.6612616, 4.8746061, -7.4390416, 4.8215342, -12.4827957, 12.3136482
3: -9.4062119, 3.5658131, -9.4918566, 3.5307240, -12.9369354, 13.0576677
4: -5.4828506, 4.1076727, -5.2550297, 4.0622358, -9.5450859, 9.3627024

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_A2_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B1_A2_A1_B2_A1

### Relational analysis result of NS_A2_A2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9031308, upper bound: 2.9008826
time: 0.55 seconds

## Relational analysis of NS_A2_A2_B1_A2_A1_B2_A2

### Relational analysis result of NS_A2_A2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9048207, upper bound: 2.9041448
time: 0.59 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -2.6361475, 2.7204149, -1.6391797, 1.7980583, -4.4342055, 4.3595943
1: -16.9633293, 6.0411367, -12.7435722, 3.9090948, -20.8724232, 18.7847080
2: -9.4186182, 6.0352888, -6.3640122, 4.1034050, -13.5220222, 12.3993015
3: -11.6321449, 4.4388990, -8.4318638, 3.0351679, -14.6673117, 12.8707619
4: -6.6953144, 5.1131492, -4.3318009, 3.4214664, -10.1167803, 9.4449501

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_A2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B1_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9467285, upper bound: 2.9540017
time: 0.52 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9501735, upper bound: 2.9589872
time: 0.53 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2.5500095, 2.6531904, -1.9547460, 2.0971751, -4.6471844, 4.6079364
1: -16.5122185, 5.8414030, -13.9443283, 4.5413580, -21.0535774, 19.7857323
2: -9.0861216, 5.8590479, -7.2319736, 4.7038183, -13.7899380, 13.0910187
3: -11.2791872, 4.3115034, -9.3015623, 3.4498291, -14.7290163, 13.6130657
4: -6.4651127, 4.9779038, -5.0734386, 3.9687941, -10.4339056, 10.0513420

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A2_A2_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B1_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8993193, upper bound: 2.9012946
time: 0.56 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9029690, upper bound: 2.9029989
time: 0.62 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -2.0833223, 2.1455822, -2.0778031, 2.1360955, -4.2194176, 4.2233853
1: -14.2000761, 4.7480178, -14.0930901, 4.7309556, -18.9310322, 18.8411064
2: -7.5650716, 4.8196573, -7.5194149, 4.7931437, -12.3582153, 12.3390713
3: -9.5536375, 3.5391467, -9.4859676, 3.5175509, -13.0711870, 13.0251122
4: -5.3569546, 4.0517697, -5.3394384, 4.0314279, -9.3883820, 9.3912058

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B2_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_A2_B2_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_A1_B1_B1_A1

### Relational analysis result of NS_A2_A2_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8635728, upper bound: 2.8635728
time: 0.70 seconds

## Relational analysis of NS_A2_A2_B2_A1_B1_B1_A2

### Relational analysis result of NS_A2_A2_B2_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2.8635728, upper bound: 2.8635728
time: 0.54 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -2.0833223, 2.1455822, -2.2509525, 2.2522371, -4.3355594, 4.3965349
1: -14.2000761, 4.7480178, -14.2948370, 5.1047606, -19.3048363, 19.0428524
2: -7.5650716, 4.8196573, -8.0405874, 5.0789738, -12.6440439, 12.8602448
3: -9.5536375, 3.5391467, -9.8095264, 3.7130065, -13.2666426, 13.3486719
4: -5.3569546, 4.0517697, -5.7511282, 4.2662520, -9.6232071, 9.8028965

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8832266, upper bound: 2.8829231
time: 0.64 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8832266, upper bound: 2.8829231
time: 0.50 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1.9885240, 2.0802901, -2.6526113, 2.7354038, -4.7239275, 4.7329006
1: -13.9209661, 4.5636716, -17.0463524, 6.0773921, -19.9983559, 21.6100197
2: -7.2961192, 4.6671147, -9.4769955, 6.0689521, -13.3650713, 14.1441097
3: -9.3161392, 3.4338171, -11.6956482, 4.4625092, -13.7786484, 15.1294651
4: -5.1243529, 3.9271591, -6.7376823, 5.1412826, -10.2656355, 10.6648417

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8835034, upper bound: 2.8840963
time: 0.49 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8835034, upper bound: 2.8840963
time: 0.63 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -2.2594872, 2.2634561, -2.2509525, 2.2522371, -4.5117235, 4.5144076
1: -14.4038410, 5.1338015, -14.2948370, 5.1047606, -19.5086021, 19.4286346
2: -8.0932121, 5.1095738, -8.0405874, 5.0789738, -13.1721859, 13.1501617
3: -9.8800411, 3.7371893, -9.8095264, 3.7130065, -13.5930462, 13.5467148
4: -5.7751522, 4.2898674, -5.7511282, 4.2662520, -10.0414047, 10.0409956

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B2_A2_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8975289, upper bound: 2.8980538
time: 0.46 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9036597, upper bound: 2.9024400
time: 0.63 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -2.1739652, 2.2048795, -2.6526113, 2.7354038, -4.9093690, 4.8574905
1: -14.1537066, 4.9699831, -17.0463524, 6.0773921, -20.2310963, 22.0163345
2: -7.8492742, 4.9720616, -9.4769955, 6.0689521, -13.9182253, 14.4490566
3: -9.6616173, 3.6432304, -11.6956482, 4.4625092, -14.1241264, 15.3388786
4: -5.5655050, 4.1780581, -6.7376823, 5.1412826, -10.7067871, 10.9157410

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_A2_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A2_B2_A2_B2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8990342, upper bound: 2.8978956
time: 0.57 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9046864, upper bound: 2.9046550
time: 0.65 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.09 seconds
NS_A2_A1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9881482, upper bound: 2.9861838
NS_A2_A1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -3.0023003, upper bound: 2.9938073
NS_A2_A1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9595238, upper bound: 2.9554012
NS_A2_A1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9850570, upper bound: 2.9785356
NS_A2_A1_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9743050, upper bound: 2.9708704
NS_A2_A1_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9743050, upper bound: 2.9716190
NS_A2_A1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9523401, upper bound: 2.9490847
NS_A2_A1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9774858, upper bound: 2.9689760
NS_A2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9935960, upper bound: 3.0021734
NS_A2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9935960, upper bound: 3.0021734
NS_A2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9935960, upper bound: 3.0021734
NS_A2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9935960, upper bound: 3.0021734
NS_A2_A1_B1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9114727, upper bound: 2.9119081
NS_A2_A1_B1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9347997, upper bound: 2.9367597
NS_A2_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9322764, upper bound: 2.9318362
NS_A2_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9321929, upper bound: 2.9325358
NS_A2_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -3.0663514, upper bound: 3.0712177
NS_A2_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9971101, upper bound: 2.9910866
NS_A2_A1_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -3.1357538, upper bound: 3.1314000
NS_A2_A1_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -3.0976143, upper bound: 3.0982118
NS_A2_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9560975, upper bound: 2.9560975
NS_A2_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9560975, upper bound: 2.9560975
NS_A2_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9560975, upper bound: 2.9560975
NS_A2_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9560975, upper bound: 2.9560975
NS_A2_A1_B2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9773815, upper bound: 2.9704078
NS_A2_A1_B2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9875393, upper bound: 2.9784173
NS_A2_A1_B2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9790038, upper bound: 2.9691574
NS_A2_A1_B2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9780033, upper bound: 2.9681066
NS_A2_A1_B2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9317742, upper bound: 2.9293122
NS_A2_A1_B2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9399356, upper bound: 2.9392441
NS_A2_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9292207, upper bound: 2.9297111
NS_A2_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9399356, upper bound: 2.9392441
NS_A2_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9287207, upper bound: 2.9289777
NS_A2_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9564907, upper bound: 2.9624769
NS_A2_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.8952903, upper bound: 2.8925920
NS_A2_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9118273, upper bound: 2.9099582
NS_A2_A2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9206896, upper bound: 2.9246245
NS_A2_A2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9206896, upper bound: 2.9246245
NS_A2_A2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.8856543, upper bound: 2.8840707
NS_A2_A2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.8856543, upper bound: 2.8840707
NS_A2_A2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9442392, upper bound: 2.9477114
NS_A2_A2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9504592, upper bound: 2.9587228
NS_A2_A2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9031308, upper bound: 2.9008826
NS_A2_A2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9048207, upper bound: 2.9041448
NS_A2_A2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9467285, upper bound: 2.9540017
NS_A2_A2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9501735, upper bound: 2.9589872
NS_A2_A2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.8993193, upper bound: 2.9012946
NS_A2_A2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9029690, upper bound: 2.9029989
NS_A2_A2_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.8635728, upper bound: 2.8635728
NS_A2_A2_B2_A1_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.8635728, upper bound: 2.8635728
NS_A2_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.8832266, upper bound: 2.8829231
NS_A2_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.8832266, upper bound: 2.8829231
NS_A2_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.8835034, upper bound: 2.8840963
NS_A2_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.8835034, upper bound: 2.8840963
NS_A2_A2_B2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.8975289, upper bound: 2.8980538
NS_A2_A2_B2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9036597, upper bound: 2.9024400
NS_A2_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.8990342, upper bound: 2.8978956
NS_A2_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.09
Output dim: 0, lower bound: -2.9046864, upper bound: 2.9046550

## BFS NS instance: NS_A2_A1_B1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1.7777362, 1.8999175, -1.8395889, 1.9409758, -3.7187116, 3.7395062
1: -13.1698208, 4.1813879, -13.3471212, 4.3018446, -17.4716644, 17.5285072
2: -6.7727790, 4.3360476, -6.9706259, 4.4374962, -11.2102757, 11.3066711
3: -8.8031225, 3.1937301, -8.9801636, 3.2549741, -12.0580969, 12.1738920
4: -4.6761446, 3.5959404, -4.8416691, 3.6764965, -8.3526411, 8.4376087

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B1_A1_A1_B1_B1_B1

### Relational analysis result of NS_A2_A1_B1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1003718, upper bound: 3.0967365
time: 0.54 seconds

## Relational analysis of NS_A2_A1_B1_A1_A1_B1_B1_B2

### Relational analysis result of NS_A2_A1_B1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.1095800, upper bound: 3.1064433
time: 0.65 seconds

## BFS NS instance: NS_A2_A1_B1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1.7594070, 1.8912321, -2.1191084, 2.2191188, -3.9785259, 4.0103397
1: -13.1522436, 4.1529422, -14.8797398, 4.9236155, -18.0758533, 19.0326824
2: -6.7289658, 4.3150082, -7.9000893, 5.0556765, -11.7846422, 12.2150955
3: -8.7754984, 3.1815600, -10.1046782, 3.6964872, -12.4719849, 13.2862377
4: -4.6368275, 3.5798306, -5.5608983, 4.1942301, -8.8310566, 9.1407290

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A1_A1_B1_B2_B1

### Relational analysis result of NS_A2_A1_B1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0402554, upper bound: 3.0361215
time: 0.53 seconds

## Relational analysis of NS_A2_A1_B1_A1_A1_B1_B2_B2

### Relational analysis result of NS_A2_A1_B1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0409058, upper bound: 3.0334168
time: 0.55 seconds

## BFS NS instance: NS_A2_A1_B1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1.7959572, 1.9190217, -2.6532898, 2.5996008, -4.3955564, 4.5723114
1: -13.2782602, 4.2251544, -15.9867201, 5.9769850, -19.2552452, 20.2118721
2: -6.8384409, 4.3792624, -9.3527851, 5.8394175, -12.6778574, 13.7320461
3: -8.8802767, 3.2261412, -11.1292238, 4.2499280, -13.1302042, 14.3553648
4: -4.7219625, 3.6334414, -6.7888937, 4.9136925, -9.6356544, 10.4223337

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9566533, upper bound: 2.9524780
time: 0.52 seconds

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9595238, upper bound: 2.9554012
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9595238, upper bound: 2.9554012
time: 0.55 seconds

## BFS NS instance: NS_A2_A1_B1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1.7262304, 1.8578496, -2.5024533, 2.4969919, -4.2232218, 4.3603020
1: -12.9718695, 4.0805202, -15.4832325, 5.6391306, -18.6110001, 19.5637531
2: -6.6195965, 4.2449627, -8.8576975, 5.5864110, -12.2060070, 13.1026592
3: -8.6439171, 3.1308858, -10.6924553, 4.0460644, -12.6899805, 13.8233404
4: -4.5482588, 3.5267537, -6.4284043, 4.7088642, -9.2571230, 9.9551563

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B2_B1

### Relational analysis result of NS_A2_A1_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9730418, upper bound: 2.9707984
time: 0.66 seconds

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9542803, upper bound: 2.9499161
time: 0.67 seconds

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B2_B1

### Relational analysis result of NS_A2_A1_B1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9505082, upper bound: 2.9470764
time: 0.53 seconds

## Relational analysis of NS_A2_A1_B1_A1_A1_B2_B2_B2

### Relational analysis result of NS_A2_A1_B1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9849030, upper bound: 2.9783358
time: 0.85 seconds

## BFS NS instance: NS_A2_A1_B1_A1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.3153086, 2.4915786, -1.8496228, 1.9516122, -4.2669210, 4.3412013
1: -16.2921238, 5.4180355, -13.4200811, 4.3260489, -20.6181717, 18.8381157
2: -8.5939922, 5.5744619, -7.0109439, 4.4613056, -13.0552979, 12.5854044
3: -11.0384893, 4.0991883, -9.0306787, 3.2733746, -14.3118639, 13.1298656
4: -6.0233488, 4.6855993, -4.8672514, 3.6966228, -9.7199717, 9.5528507

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B1_A1_A2_B1_A1_B1

### Relational analysis result of NS_A2_A1_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9743050, upper bound: 2.9708704
time: 0.52 seconds

## Relational analysis of NS_A2_A1_B1_A1_A2_B1_A1_B2

### Relational analysis result of NS_A2_A1_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9743050, upper bound: 2.9708704
time: 0.51 seconds

## BFS NS instance: NS_A2_A1_B1_A1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.5811772, 2.7676311, -1.8496228, 1.9516122, -4.5327892, 4.6172538
1: -17.6782207, 6.0132995, -13.4200811, 4.3260489, -22.0042686, 19.4333801
2: -9.4589796, 6.1568718, -7.0109439, 4.4613056, -13.9202852, 13.1678152
3: -12.0554218, 4.5169253, -9.0306787, 3.2733746, -15.3287964, 13.5476036
4: -6.6995144, 5.1893520, -4.8672514, 3.6966228, -10.3961372, 10.0566025

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 10

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_A1_B1_A1_A2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9743050, upper bound: 2.9716190
time: 0.52 seconds

## Relational analysis of NS_A2_A1_B1_A1_A2_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9743050, upper bound: 2.9716190
time: 0.54 seconds

## BFS NS instance: NS_A2_A1_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -2.3449044, 2.5272570, -2.1050887, 2.2080569, -4.5529613, 4.6323457
1: -16.5224686, 5.4884405, -14.8277979, 4.8933835, -21.4158516, 20.3162384
2: -8.6993217, 5.6510386, -7.8575525, 5.0297475, -13.7290678, 13.5085907
3: -11.1883469, 4.1597919, -10.0619450, 3.6790802, -14.8674269, 14.2217360
4: -6.0913224, 4.7505078, -5.5254812, 4.1725492, -10.2638721, 10.2759895

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A1_A2_B2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9523401, upper bound: 2.9490847
time: 0.74 seconds

## Relational analysis of NS_A2_A1_B1_A1_A2_B2_B1_A2

### Relational analysis result of NS_A2_A1_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9523401, upper bound: 2.9490847
time: 0.57 seconds

## BFS NS instance: NS_A2_A1_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -2.2793381, 2.4687793, -2.0200939, 2.1592615, -4.4385986, 4.4888735
1: -16.2126102, 5.3465853, -14.7359104, 4.7128830, -20.9254894, 20.0824966
2: -8.4819260, 5.5215120, -7.6275854, 4.9062076, -13.3881340, 13.1490955
3: -10.9515924, 4.0685000, -9.9154949, 3.5805266, -14.5321188, 13.9839954
4: -5.9232349, 4.6415167, -5.3296552, 4.0661378, -9.9893713, 9.9711704

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A1_A2_B2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9774858, upper bound: 2.9689761
time: 0.60 seconds

## Relational analysis of NS_A2_A1_B1_A1_A2_B2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9774858, upper bound: 2.9689761
time: 0.61 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.1092110, 2.2126124, -1.8050538, 1.9276863, -4.0368972, 4.0176663
1: -14.4327822, 4.8408551, -13.3232136, 4.2450600, -18.6778412, 18.1640682
2: -7.6697903, 4.9607201, -6.8692470, 4.3981552, -12.0679436, 11.8299656
3: -9.7186594, 3.6245968, -8.9142818, 3.2397642, -12.9584236, 12.5388775
4: -5.4652557, 4.1807156, -4.7453728, 3.6497135, -9.1149693, 8.9260883

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0933517, upper bound: 3.0920976
time: 0.72 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0890392, upper bound: 3.0890392
time: 0.61 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.1092110, 2.2126124, -2.1046832, 2.2105956, -4.3198066, 4.3172956
1: -14.4327822, 4.8408551, -14.4092226, 4.8321228, -19.2649040, 19.2500744
2: -7.6697903, 4.9607201, -7.6558414, 4.9541669, -12.6239567, 12.6165619
3: -9.7186594, 3.6245968, -9.6986160, 3.6205781, -13.3392372, 13.3232126
4: -5.4652557, 4.1807156, -5.4534521, 4.1773834, -9.6426392, 9.6341677

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0920976, upper bound: 3.0933517
time: 0.50 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0890392, upper bound: 3.0890392
time: 0.63 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.7934542, 2.9153845, -1.8050538, 1.9276863, -4.7211404, 4.7204385
1: -18.2395229, 6.4129181, -13.3232136, 4.2450600, -22.4845772, 19.7361317
2: -10.0061789, 6.4529142, -6.8692470, 4.3981552, -14.4043341, 13.3221598
3: -12.4730673, 4.7330294, -8.9142818, 3.2397642, -15.7128315, 13.6473093
4: -7.1816974, 5.4638925, -4.7453728, 3.6497135, -10.8314114, 10.2092619

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9910866, upper bound: 2.9971101
time: 0.47 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9935174, upper bound: 3.0003877
time: 0.50 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.7934542, 2.9153845, -2.1046832, 2.2105956, -5.0040493, 5.0200677
1: -18.2395229, 6.4129181, -14.4092226, 4.8321228, -23.0716419, 20.8221397
2: -10.0061789, 6.4529142, -7.6558414, 4.9541669, -14.9603462, 14.1087551
3: -12.4730673, 4.7330294, -9.6986160, 3.6205781, -16.0936451, 14.4316444
4: -7.1816974, 5.4638925, -5.4534521, 4.1773834, -11.3590813, 10.9173422

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9910866, upper bound: 2.9971101
time: 0.55 seconds

## Relational analysis of NS_A2_A1_B1_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_A1_B1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9935174, upper bound: 3.0003877
time: 0.59 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -2.1075068, 2.2132399, -2.5383427, 2.5303669, -4.6378736, 4.7515821
1: -14.4739656, 4.8403945, -15.3228159, 5.6945863, -20.1685486, 20.1632099
2: -7.6844358, 4.9649448, -8.8617821, 5.6279678, -13.3124037, 13.8267269
3: -9.7425833, 3.6298954, -10.6072998, 4.0770254, -13.8196087, 14.2371950
4: -5.4639330, 4.1827493, -6.4638538, 4.7828798, -10.2468128, 10.6466026

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9114727, upper bound: 2.9119081
time: 0.64 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9114727, upper bound: 2.9119081
time: 0.58 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -2.1109746, 2.2161088, -2.4713106, 2.4532707, -4.5642452, 4.6874194
1: -14.4878464, 4.8481565, -14.9859705, 5.5518827, -20.0397301, 19.8341217
2: -7.6958413, 4.9715037, -8.6712818, 5.4745903, -13.1704311, 13.6427860
3: -9.7543020, 3.6343887, -10.3856764, 3.9641805, -13.7184830, 14.0200644
4: -5.4732900, 4.1880951, -6.3192039, 4.6289368, -10.1022263, 10.5072975

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9347997, upper bound: 2.9367597
time: 0.60 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9347997, upper bound: 2.9367597
time: 0.77 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -2.0928807, 2.2076397, -2.6521854, 2.6533329, -4.7462134, 4.8598251
1: -14.4670286, 4.8187366, -16.0131664, 5.9698524, -20.4368782, 20.8319016
2: -7.6524057, 4.9516473, -9.2932329, 5.9041309, -13.5565367, 14.2448807
3: -9.7259016, 3.6208010, -11.1256371, 4.2703829, -13.9962835, 14.7464380
4: -5.4345155, 4.1710734, -6.7905116, 4.9988461, -10.4333611, 10.9615850

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B1_A2_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A1_B1_A2_B2_B2_B1_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9322764, upper bound: 2.9318362
time: 0.72 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_B2_B1_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9322764, upper bound: 2.9318362
time: 0.62 seconds

## BFS NS instance: NS_A2_A1_B1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -2.0262442, 2.1512952, -2.5455768, 2.5780110, -4.6042552, 4.6968713
1: -14.1832790, 4.6793637, -15.7658300, 5.7393050, -19.9225807, 20.4451942
2: -7.4420910, 4.8257933, -8.9653969, 5.7232680, -13.1653585, 13.7911901
3: -9.5031919, 3.5322607, -10.8623714, 4.1364422, -13.6396341, 14.3946323
4: -5.2661848, 4.0661726, -6.5373135, 4.8460283, -10.1122122, 10.6034861

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B1_A2_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B1_A2_B2_B2_B2_A1

### Relational analysis result of NS_A2_A1_B1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9265851, upper bound: 2.9265143
time: 0.67 seconds

## Relational analysis of NS_A2_A1_B1_A2_B2_B2_B2_A2

### Relational analysis result of NS_A2_A1_B1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9265851, upper bound: 2.9325358
time: 0.76 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -1.8301196, 1.9322033, -2.5164137, 2.6497631, -4.4798827, 4.4486165
1: -13.3010311, 4.2809772, -17.3001595, 5.8685355, -19.1695652, 21.5811367
2: -6.9386649, 4.4182096, -9.2901440, 5.9534764, -12.8921394, 13.7083530
3: -8.9451132, 3.2408326, -11.8059387, 4.3869157, -13.3320284, 15.0467701
4: -4.8173528, 3.6605444, -6.5406036, 4.9815555, -9.7989082, 10.2011480

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_B1_A1_A1_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9971101, upper bound: 2.9910866
time: 0.62 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_A1_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9971101, upper bound: 2.9910866
time: 0.51 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -1.6834698, 1.8195522, -2.2049921, 2.4152424, -4.0987115, 4.0245438
1: -12.6472445, 3.9598475, -15.9296331, 5.1677732, -17.8150177, 19.8894730
2: -6.4216123, 4.1334224, -8.1454906, 5.3603663, -11.7819777, 12.2789135
3: -8.4141006, 3.0395024, -10.6620464, 3.9686313, -12.3827324, 13.7015486
4: -4.4288602, 3.4578896, -5.7138391, 4.5442834, -8.9731436, 9.1717281

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_B1_A1_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9971101, upper bound: 2.9910866
time: 0.53 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9971101, upper bound: 2.9910866
time: 0.56 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -2.0948675, 2.1960993, -2.6157134, 2.7339842, -4.8288517, 4.8118124
1: -14.7493467, 4.8687658, -17.7920570, 6.0875754, -20.8369198, 22.6608238
2: -7.8158031, 5.0041780, -9.6429987, 6.1528578, -13.9686584, 14.6471767
3: -10.0087337, 3.6592517, -12.1856079, 4.5275812, -14.5363150, 15.8448563
4: -5.4995861, 4.1508007, -6.8029108, 5.1402907, -10.6398773, 10.9537115

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_B1_A1_A2_A1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0185832, upper bound: 3.0064937
time: 0.57 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_A2_A1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9908536, upper bound: 2.9840015
time: 0.56 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -2.0099099, 2.1475215, -2.5335557, 2.6650832, -4.6749930, 4.6810770
1: -14.6645365, 4.6874971, -17.4186687, 5.9114251, -20.5759583, 22.1061630
2: -7.5885382, 4.8814821, -9.3725796, 5.9986329, -13.5871687, 14.2540588
3: -9.8664408, 3.5615032, -11.9013901, 4.4152603, -14.2816992, 15.4628935
4: -5.3043137, 4.0446177, -6.5973592, 5.0116982, -10.3160114, 10.6419754

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_B1_A1_A2_A2_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0096368, upper bound: 3.0000059
time: 0.61 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_A2_A2_A2

### Relational analysis result of NS_A2_A1_B2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9752437, upper bound: 2.9731232
time: 0.68 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -2.5251026, 2.6602459, -2.5343070, 2.6688688, -5.1939707, 5.1945529
1: -17.3686161, 5.8911114, -17.4156818, 5.9115872, -23.2802029, 23.3067913
2: -9.3235998, 5.9766178, -9.3557100, 5.9955416, -15.3191414, 15.3323278
3: -11.8521128, 4.4046597, -11.8874426, 4.4184709, -16.2705841, 16.2921009
4: -6.5626884, 5.0010352, -6.5863404, 5.0170679, -11.5797558, 11.5873756

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B2_B1_A2_A1_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0083986, upper bound: 2.9974992
time: 0.56 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_A1_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0066949, upper bound: 2.9979555
time: 0.68 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -2.5251026, 2.6602459, -2.2217183, 2.4332461, -4.9583483, 4.8819642
1: -17.3686161, 5.8911114, -16.0391426, 5.2082191, -22.5768356, 21.9302521
2: -9.3235998, 5.9766178, -8.2069283, 5.4001851, -14.7237854, 14.1835451
3: -11.8521128, 4.4046597, -10.7389174, 3.9979100, -15.8500233, 15.1435766
4: -6.5626884, 5.0010352, -5.7566371, 4.5779624, -11.1406507, 10.7576723

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B2_B1_A2_A1_B2_A1

### Relational analysis result of NS_A2_A1_B2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0083986, upper bound: 2.9974992
time: 0.65 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_A1_B2_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -3.0066949, upper bound: 2.9979555
time: 0.58 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -2.2124963, 2.4248025, -2.5343070, 2.6688688, -4.8813643, 4.9591093
1: -15.9951668, 5.1879196, -17.4156818, 5.9115872, -21.9067535, 22.6036015
2: -8.1758146, 5.3815579, -9.3557100, 5.9955416, -14.1713562, 14.7372684
3: -10.7052555, 3.9843285, -11.8874426, 4.4184709, -15.1237259, 15.8717670
4: -5.7329583, 4.5626316, -6.5863404, 5.0170679, -10.7500248, 11.1489716

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_B1_A2_A2_B1_B1

### Relational analysis result of NS_A2_A1_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9481674, upper bound: 2.9494729
time: 0.54 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_A2_B1_B2

### Relational analysis result of NS_A2_A1_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9558381, upper bound: 2.9558380
time: 0.57 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2.2124963, 2.4248025, -2.2217183, 2.4332461, -4.6457419, 4.6465206
1: -15.9951668, 5.1879196, -16.0391426, 5.2082191, -21.2033863, 21.2270622
2: -8.1758146, 5.3815579, -8.2069283, 5.4001851, -13.5760002, 13.5884857
3: -10.7052555, 3.9843285, -10.7389174, 3.9979100, -14.7031651, 14.7232447
4: -5.7329583, 4.5626316, -5.7566371, 4.5779624, -10.3109207, 10.3192692

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B2_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_B1_A2_A2_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9481674, upper bound: 2.9494729
time: 0.62 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_A2_B2_B2

### Relational analysis result of NS_A2_A1_B2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9558381, upper bound: 2.9558380
time: 0.68 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -1.7337152, 1.8541949, -2.1210139, 2.2948720, -4.0285864, 3.9752088
1: -12.9083843, 4.0765729, -15.2544107, 4.9785285, -17.8869114, 19.3309822
2: -6.6153851, 4.2325664, -7.9006524, 5.1104946, -11.7258797, 12.1332169
3: -8.6169758, 3.1163826, -10.2580957, 3.8003430, -12.4173183, 13.3744783
4: -4.5652676, 3.5073793, -5.4797192, 4.3294578, -8.8947248, 8.9870977

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_A1_B2_B2_A1_A1_A1_B1

### Relational analysis result of NS_A2_A1_B2_B2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9873907, upper bound: 2.9787042
time: 0.63 seconds

## Relational analysis of NS_A2_A1_B2_B2_A1_A1_A1_B2

### Relational analysis result of NS_A2_A1_B2_B2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9873907, upper bound: 2.9837378
time: 0.56 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -1.9588754, 2.0920744, -2.1060367, 2.2856088, -4.2444839, 4.1981111
1: -14.1729145, 4.5792756, -15.2180147, 4.9481421, -19.1210556, 19.7972908
2: -7.3605099, 4.7543950, -7.8541212, 5.0863681, -12.4468784, 12.6085157
3: -9.5299168, 3.4857988, -10.2209349, 3.7853048, -13.3152218, 13.7067337
4: -5.1434660, 3.9532936, -5.4424405, 4.3102579, -9.4537239, 9.3957338

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B2_B2_A1_A1_A2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9903981, upper bound: 2.9798585
time: 0.52 seconds

## Relational analysis of NS_A2_A1_B2_B2_A1_A1_A2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9818531, upper bound: 2.9726183
time: 0.58 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -2.3541477, 2.5335917, -2.1401753, 2.3133917, -4.6675391, 4.6737661
1: -16.5480022, 5.5076885, -15.3669500, 5.0218735, -21.5698757, 20.8746376
2: -8.7304564, 5.6670923, -7.9661827, 5.1530037, -13.8834562, 13.6332750
3: -11.2135544, 4.1710114, -10.3377438, 3.8326349, -15.0461893, 14.5087538
4: -6.1168227, 4.7652316, -5.5266538, 4.3651185, -10.4819412, 10.2918854

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A1_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A1_B2_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A1_B2_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A1_B2_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_B2_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A2_A1_B2_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -2.1650672, 2.3915567, -2.0608981, 2.2501605, -4.4152274, 4.4524546
1: -15.9880085, 5.1047182, -14.9786873, 4.8486614, -20.8366699, 20.0834026
2: -8.1531010, 5.3338275, -7.6971955, 4.9974480, -13.1505489, 13.0310230
3: -10.7189236, 3.9284511, -10.0443172, 3.7195945, -14.4385176, 13.9727659
4: -5.6562209, 4.4831471, -5.3253908, 4.2464032, -9.9026222, 9.8085384

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_B2_A1_A2_A2_B1

### Relational analysis result of NS_A2_A1_B2_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9652423, upper bound: 2.9612123
time: 0.55 seconds

## Relational analysis of NS_A2_A1_B2_B2_A1_A2_A2_B2

### Relational analysis result of NS_A2_A1_B2_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9770264, upper bound: 2.9680395
time: 0.71 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -2.0488951, 2.1507783, -1.9905457, 2.2004578, -4.2493529, 4.1413240
1: -14.0813513, 4.7022476, -14.6510277, 4.6862955, -18.7676468, 19.3532753
2: -7.4525952, 4.8183804, -7.4255147, 4.8557930, -12.3083878, 12.2438946
3: -9.4664726, 3.5221055, -9.7713518, 3.6187465, -13.0852175, 13.2934570
4: -5.3136907, 4.0605674, -5.1329470, 4.1496363, -9.4633274, 9.1935139

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B2_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B2_B2_A2_A1_A1_A1

### Relational analysis result of NS_A2_A1_B2_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9710853, upper bound: 2.9650820
time: 0.63 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_A1_A1_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9670817, upper bound: 2.9620508
time: 0.61 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -2.2931004, 2.3968127, -1.9751711, 2.1908038, -4.4839039, 4.3719826
1: -15.4125509, 5.2549005, -14.6120491, 4.6550341, -20.0675850, 19.8669491
2: -8.2739544, 5.3643680, -7.3777857, 4.8305569, -13.1045103, 12.7421532
3: -10.4377422, 3.9052689, -9.7327099, 3.6028829, -14.0406246, 13.6379776
4: -5.9412522, 4.5187335, -5.0944858, 4.1297517, -10.0709991, 9.6132193

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A1_B2_B2_A2_A1_A2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9752837, upper bound: 2.9676704
time: 0.70 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_A1_A2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9586747, upper bound: 2.9551227
time: 0.58 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -2.7287490, 2.8691845, -1.9477559, 2.1593900, -4.8881388, 4.8169403
1: -17.8755512, 6.2584515, -14.3964348, 4.5892782, -22.4648285, 20.6548862
2: -9.7758398, 6.3398242, -7.2784872, 4.7601805, -14.5360184, 13.6183109
3: -12.2227736, 4.6297874, -9.5907164, 3.5469470, -15.7697182, 14.2205038
4: -7.0260158, 5.3790159, -5.0271902, 4.0704126, -11.0964279, 10.4062052

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9283789, upper bound: 2.9275892
time: 0.67 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9283789, upper bound: 2.9297111
time: 0.72 seconds

## BFS NS instance: NS_A2_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -2.7171235, 2.8638470, -2.2511318, 2.4501557, -5.1672792, 5.1149788
1: -17.8687229, 6.2395544, -15.9799213, 5.2679944, -23.1367168, 22.2194710
2: -9.7477007, 6.3279395, -8.2825155, 5.4140205, -15.1617212, 14.6104527
3: -12.2079802, 4.6204848, -10.7605839, 4.0179429, -16.2259235, 15.3810692
4: -7.0006757, 5.3683085, -5.7905712, 4.6047273, -11.6054029, 11.1588774

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9317742, upper bound: 2.9293122
time: 0.60 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9317742, upper bound: 2.9392441
time: 0.58 seconds

## BFS NS instance: NS_A2_A2_B1_A1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.0051413, 2.0645487, -1.5779796, 1.7418203, -3.7469616, 3.6425283
1: -13.6780014, 4.5644751, -12.3861666, 3.7627749, -17.4407768, 16.9506416
2: -7.2571650, 4.6260414, -6.1400433, 3.9637272, -11.2208920, 10.7660847
3: -9.1862755, 3.3925014, -8.1749134, 2.9296792, -12.1159544, 11.5674143
4: -5.1546779, 3.8882031, -4.1767373, 3.3141279, -8.4688053, 8.0649405

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_A1_B1

### Relational analysis result of NS_A2_A2_B1_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9287207, upper bound: 2.9289777
time: 0.60 seconds

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_A1_B2

### Relational analysis result of NS_A2_A2_B1_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9287207, upper bound: 2.9289777
time: 0.48 seconds

## BFS NS instance: NS_A2_A2_B1_A1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.1818461, 2.2593429, -1.5641261, 1.7354897, -3.9173355, 3.8234687
1: -14.6771841, 4.9733205, -12.3665886, 3.7357216, -18.4129047, 17.3399086
2: -7.8592110, 5.0513425, -6.0980587, 3.9448581, -11.8040695, 11.1494007
3: -9.9102631, 3.6970835, -8.1482601, 2.9187593, -12.8290215, 11.8453426
4: -5.6142001, 4.2586722, -4.1431293, 3.3030984, -8.9172964, 8.4018021

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_A2_A1

### Relational analysis result of NS_A2_A2_B1_A1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9175339, upper bound: 2.9188052
time: 0.58 seconds

## Relational analysis of NS_A2_A2_B1_A1_A1_B1_A2_A2

### Relational analysis result of NS_A2_A2_B1_A1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9399503, upper bound: 2.9457855
time: 0.52 seconds

## BFS NS instance: NS_A2_A2_B1_A1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.9274709, 2.0039616, -1.8938410, 2.0296502, -3.9571204, 3.8978026
1: -13.2655182, 4.3864965, -13.5780029, 4.3829913, -17.6485100, 17.9645004
2: -6.9566979, 4.4669747, -6.9863834, 4.5517120, -11.5084095, 11.4533577
3: -8.8620253, 3.2783382, -9.0363207, 3.3391247, -12.2011499, 12.3146591
4: -4.9406891, 3.7679820, -4.9141250, 3.8432422, -8.7839308, 8.6821060

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B1_A1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_A1_A1_B2_A1_B1

### Relational analysis result of NS_A2_A2_B1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8952903, upper bound: 2.8925920
time: 0.65 seconds

## Relational analysis of NS_A2_A2_B1_A1_A1_B2_A1_B2

### Relational analysis result of NS_A2_A2_B1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8952903, upper bound: 2.8925920
time: 0.54 seconds

## BFS NS instance: NS_A2_A2_B1_A1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.1058850, 2.1987715, -1.8753629, 2.0210299, -4.1269150, 4.0741343
1: -14.2558632, 4.7981353, -13.5569286, 4.3544221, -18.6102829, 18.3550644
2: -7.5613317, 4.8944402, -6.9412141, 4.5308261, -12.0921574, 11.8356533
3: -9.5841112, 3.5832775, -9.0075846, 3.3264203, -12.9105291, 12.5908623
4: -5.4040861, 4.1391120, -4.8737545, 3.8287582, -9.2328424, 9.0128651

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_A1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_B1_A1_A1_B2_A2_A1

### Relational analysis result of NS_A2_A2_B1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8975042, upper bound: 2.8955654
time: 0.61 seconds

## Relational analysis of NS_A2_A2_B1_A1_A1_B2_A2_A2

### Relational analysis result of NS_A2_A2_B1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8979539, upper bound: 2.8954233
time: 0.55 seconds

## BFS NS instance: NS_A2_A2_B1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -2.1760516, 2.3254528, -1.5789018, 1.7396508, -3.9157023, 3.9043546
1: -15.2513132, 5.0243521, -12.3308783, 3.7569101, -19.0082226, 17.3552303
2: -7.9691935, 5.1579289, -6.1228452, 3.9564517, -11.9256449, 11.2807722
3: -10.2199030, 3.8061888, -8.1407747, 2.9237449, -13.1436462, 11.9469624
4: -5.6044588, 4.3764486, -4.1695089, 3.3104661, -8.9149246, 8.5459547

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

## BFS NS instance: NS_A2_A2_B1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -2.1760516, 2.3254528, -2.1547084, 2.3466768, -4.5227284, 4.4801612
1: -15.2513132, 5.0243521, -15.5658350, 5.0776553, -20.3289680, 20.5901852
2: -7.9691935, 5.1579289, -8.0656290, 5.2704782, -13.2396717, 13.2235565
3: -10.2199030, 3.8061888, -10.4794788, 3.8853462, -14.1052475, 14.2856674
4: -5.6044588, 4.3764486, -5.6090412, 4.4233098, -10.0277691, 9.9854841

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 42

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 42

## BFS NS instance: NS_A2_A2_B1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -2.0990887, 2.2650456, -1.8853266, 2.0197036, -4.1187925, 4.1503720
1: -14.8277655, 4.8458567, -13.5065994, 4.3609700, -19.1887360, 18.3524551
2: -7.6656880, 4.9996347, -6.9433923, 4.5299449, -12.1956329, 11.9430265
3: -9.8915491, 3.6923342, -8.9870977, 3.3210793, -13.2126284, 12.6794319
4: -5.3928647, 4.2588215, -4.8910236, 3.8243361, -9.2172003, 9.1498432

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_A2_A2_B1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -2.0990887, 2.2650456, -2.3977306, 2.5943022, -4.6933899, 4.6627760
1: -14.8277655, 4.8458567, -16.5422859, 5.5553317, -20.3830948, 21.3881416
2: -7.6656880, 4.9996347, -8.6927090, 5.7205558, -13.3862438, 13.6923437
3: -9.8915491, 3.6923342, -11.1429491, 4.1983671, -14.0899162, 14.8352833
4: -5.3928647, 4.2588215, -6.1780906, 4.8661551, -10.2590189, 10.4369097

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A2_B1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_A2_A2_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.1770220, 2.1786637, -1.6912832, 1.8313422, -4.0083632, 3.8699470
1: -13.8706551, 4.9351335, -12.8823280, 4.0142522, -17.8849068, 17.8174591
2: -7.7708907, 4.9077358, -6.5212293, 4.1857643, -11.9566536, 11.4289618
3: -9.5002851, 3.5864570, -8.5639849, 3.0904498, -12.5907345, 12.1504421
4: -5.5625801, 4.1205091, -4.4640489, 3.4800715, -9.0426502, 8.5845566

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 1

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A1_A1

### Relational analysis result of NS_A2_A2_B1_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9121780, upper bound: 2.9119275
time: 0.61 seconds

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A1_A2

### Relational analysis result of NS_A2_A2_B1_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9439905, upper bound: 2.9474780
time: 0.53 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.3612700, 2.3853002, -1.6764047, 1.8238502, -4.1851201, 4.0617046
1: -14.9909582, 5.3662701, -12.8609867, 3.9846649, -18.9756241, 18.2272568
2: -8.4143152, 5.3560238, -6.4761686, 4.1649842, -12.5792980, 11.8321924
3: -10.2919016, 3.9088285, -8.5351505, 3.0777540, -13.3696556, 12.4439783
4: -6.0447741, 4.5070686, -4.4270153, 3.4665058, -9.5112782, 8.9340830

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A2_A1

### Relational analysis result of NS_A2_A2_B1_A2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9069494, upper bound: 2.9073002
time: 0.44 seconds

## Relational analysis of NS_A2_A2_B1_A2_A1_B1_A2_A2

### Relational analysis result of NS_A2_A2_B1_A2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9502373, upper bound: 2.9585039
time: 0.52 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.0940483, 2.1140671, -2.0097783, 2.1303844, -4.2244320, 4.1238451
1: -13.4343796, 4.7424717, -14.0657778, 4.6388898, -18.0732670, 18.8082447
2: -7.4488182, 4.7369509, -7.3771262, 4.7805634, -12.2293797, 12.1140766
3: -9.1590366, 3.4628949, -9.4205618, 3.5012851, -12.6603212, 12.8834553
4: -5.3365307, 3.9900813, -5.2121191, 4.0276232, -9.3641539, 9.2021990

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_A2_A1_B2_A1_B1

### Relational analysis result of NS_A2_A2_B1_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9031308, upper bound: 2.9008826
time: 0.51 seconds

## Relational analysis of NS_A2_A2_B1_A2_A1_B2_A1_B2

### Relational analysis result of NS_A2_A2_B1_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9031308, upper bound: 2.9008826
time: 0.52 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.2778707, 2.3200653, -1.9910079, 2.1215129, -4.3993835, 4.3110728
1: -14.5499830, 5.1743460, -14.0430107, 4.6094427, -19.1594257, 19.2173557
2: -8.0910711, 5.1848326, -7.3312035, 4.7592402, -12.8503103, 12.5160351
3: -9.9476213, 3.7853339, -9.3901653, 3.4874198, -13.4350405, 13.1754990
4: -5.8192043, 4.3761516, -5.1712775, 4.0103712, -9.8295755, 9.5474281

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_B1_A2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_A2_A1_B2_A2_B1

### Relational analysis result of NS_A2_A2_B1_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9048207, upper bound: 2.9041448
time: 0.59 seconds

## Relational analysis of NS_A2_A2_B1_A2_A1_B2_A2_B2

### Relational analysis result of NS_A2_A2_B1_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9048207, upper bound: 2.9041448
time: 0.66 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -2.6183746, 2.7032361, -1.5753320, 1.7359349, -4.3543096, 4.2785678
1: -16.8617439, 6.0007520, -12.3792725, 3.7638855, -20.6256294, 18.3800240
2: -9.3581791, 5.9959054, -6.1468430, 3.9601831, -13.3183622, 12.1427479
3: -11.5602083, 4.4091063, -8.1720314, 2.9272091, -14.4874163, 12.5811377
4: -6.6527472, 5.0801163, -4.1772308, 3.2998767, -9.9526224, 9.2573471

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B1_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9467285, upper bound: 2.9540017
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9467285, upper bound: 2.9540017
time: 0.52 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -2.6001568, 2.6908453, -1.7972625, 1.9613245, -4.5614810, 4.4881077
1: -16.8170109, 5.9631066, -13.6500483, 4.2608228, -21.0778332, 19.6131535
2: -9.2997828, 5.9667664, -6.8824449, 4.4688792, -13.7686615, 12.8492107
3: -11.5158815, 4.3903050, -9.0799570, 3.2935324, -14.8094139, 13.4702625
4: -6.6082168, 5.0553923, -4.7307792, 3.7086813, -10.3168983, 9.7861700

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_A2_A2_B1_B2_B1

### Relational analysis result of NS_A2_A2_B1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9501735, upper bound: 2.9589872
time: 0.62 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2_B1_B2_B2

### Relational analysis result of NS_A2_A2_B1_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9501735, upper bound: 2.9589872
time: 0.50 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -2.5322313, 2.6360061, -1.8961667, 2.0381186, -4.5703497, 4.5321727
1: -16.4108562, 5.8010559, -13.6098967, 4.4046478, -20.8155041, 19.4109535
2: -9.0255051, 5.8196120, -7.0261021, 4.5676713, -13.5931759, 12.8457117
3: -11.2070847, 4.2818880, -9.0625877, 3.3518250, -14.5589094, 13.3444757
4: -6.4223452, 4.9448166, -4.9304967, 3.8530989, -10.2754440, 9.8753128

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 31

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_A2_A2_B2_B1_B1

### Relational analysis result of NS_A2_A2_B1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8993193, upper bound: 2.9012946
time: 0.55 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2_B2_B1_B2

### Relational analysis result of NS_A2_A2_B1_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8993193, upper bound: 2.9012946
time: 0.74 seconds

## BFS NS instance: NS_A2_A2_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -2.5144980, 2.6241026, -2.0995858, 2.2540448, -4.7685428, 4.7236872
1: -16.3662834, 5.7645202, -14.7592907, 4.8659334, -21.2322159, 20.5238113
2: -8.9681940, 5.7915473, -7.7007771, 5.0395365, -14.0077305, 13.4923248
3: -11.1634827, 4.2637615, -9.8866625, 3.6846399, -14.8481226, 14.1504230
4: -6.3787346, 4.9215736, -5.4547968, 4.2518072, -10.6305399, 10.3763704

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B1_A2_A2_B2_B2_B1

### Relational analysis result of NS_A2_A2_B1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9029690, upper bound: 2.9029989
time: 0.77 seconds

## Relational analysis of NS_A2_A2_B1_A2_A2_B2_B2_B2

### Relational analysis result of NS_A2_A2_B1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8993193, upper bound: 2.9029989
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -2.0678477, 2.1272078, -2.2509525, 2.2522371, -4.3200846, 4.3781605
1: -14.0492592, 4.7090015, -14.2948370, 5.1047606, -19.1540203, 19.0038376
2: -7.4866199, 4.7734518, -8.0405874, 5.0789738, -12.5655899, 12.8140383
3: -9.4518328, 3.5033979, -9.8095264, 3.7130065, -13.1648378, 13.3129234
4: -5.3139734, 4.0146866, -5.7511282, 4.2662520, -9.5802250, 9.7658119

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_A2_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -2.1822925, 2.3305235, -2.2509525, 2.2522371, -4.4345293, 4.5814743
1: -15.2766113, 5.0384221, -14.2948370, 5.1047606, -20.3813725, 19.3332577
2: -7.9910593, 5.1700749, -8.0405874, 5.0789738, -13.0700312, 13.2106628
3: -10.2413445, 3.8145323, -9.8095264, 3.7130065, -13.9543495, 13.6240578
4: -5.6203971, 4.3862247, -5.7511282, 4.2662520, -9.8866491, 10.1373529

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B2_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_A2_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -2.0678477, 2.1272078, -2.6526113, 2.7354038, -4.8032513, 4.7798190
1: -14.0492592, 4.7090015, -17.0463524, 6.0773921, -20.1266479, 21.7553539
2: -7.4866199, 4.7734518, -9.4769955, 6.0689521, -13.5555696, 14.2504463
3: -9.4518328, 3.5033979, -11.6956482, 4.4625092, -13.9143419, 15.1990461
4: -5.3139734, 4.0146866, -6.7376823, 5.1412826, -10.4552536, 10.7523680

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -2.1822925, 2.3305235, -2.6526113, 2.7354038, -4.9176960, 4.9831338
1: -15.2766113, 5.0384221, -17.0463524, 6.0773921, -21.3540020, 22.0847721
2: -7.9910593, 5.1700749, -9.4769955, 6.0689521, -14.0600109, 14.6470699
3: -10.2413445, 3.8145323, -11.6956482, 4.4625092, -14.7038536, 15.5101805
4: -5.6203971, 4.3862247, -6.7376823, 5.1412826, -10.7616796, 11.1239071

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_A2_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## BFS NS instance: NS_A2_A2_B2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -2.2416875, 2.2451229, -2.1929636, 2.1929579, -4.4346452, 4.4380860
1: -14.2976427, 5.0906162, -13.9490595, 4.9706612, -19.2683029, 19.0396767
2: -8.0277023, 5.0668845, -7.8268633, 4.9403377, -12.9680405, 12.8937473
3: -9.8037119, 3.7052939, -9.5611944, 3.6092932, -13.4130030, 13.2664881
4: -5.7301707, 4.2532201, -5.6040964, 4.1475706, -9.8777409, 9.8573160

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8961923, upper bound: 2.8977218
time: 0.77 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8961923, upper bound: 2.8977218
time: 0.65 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -2.2177832, 2.2326524, -2.3780897, 2.4005492, -4.6183314, 4.6107421
1: -14.2576380, 5.0500817, -15.0743723, 5.4039927, -19.6616287, 20.1244526
2: -7.9667311, 5.0373015, -8.4733200, 5.3903899, -13.3571196, 13.5106201
3: -9.7594366, 3.6865578, -10.3559303, 3.9332924, -13.6927290, 14.0424871
4: -5.6777859, 4.2300882, -6.0882630, 4.5356693, -10.2134552, 10.3183517

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9022098, upper bound: 2.9021644
time: 0.71 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.9022098, upper bound: 2.9024400
time: 0.64 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -2.1156807, 2.1452503, -2.6347945, 2.7181952, -4.8338752, 4.7800446
1: -13.8069391, 4.8288932, -16.9446259, 6.0369029, -19.8438416, 21.7735195
2: -7.6348619, 4.8328614, -9.4164181, 6.0294738, -13.6643353, 14.2492790
3: -9.4122381, 3.5393569, -11.6235838, 4.4326553, -13.8448915, 15.1629410
4: -5.4181070, 4.0587258, -6.6950073, 5.1081810, -10.5262880, 10.7537327

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A2_B2_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_A2_B2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8979093, upper bound: 2.8968375
time: 0.58 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_A2_B2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2.8979093, upper bound: 2.8974457
time: 0.60 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.25 + 417.75 = 421.01 seconds
