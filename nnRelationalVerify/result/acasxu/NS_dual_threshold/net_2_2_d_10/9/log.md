## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 9)
Time budget: 420 seconds
Split limit: 100
Threshold: 198.13671952904002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-75.5068207, 138.2489624, -75.5068207, 138.2489624, -213.7557831, 213.7557831)
1: (-67.3784637, 126.3198547, -67.3784637, 126.3198547, -193.6983032, 193.6983032)
2: (-58.9700966, 130.5795898, -58.9700966, 130.5795898, -189.5496826, 189.5496826)
3: (-92.7449951, 129.7379456, -92.7449951, 129.7379456, -222.4829407, 222.4829407)
4: (-72.3862000, 139.0076752, -72.3862000, 139.0076752, -211.3938751, 211.3938751)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.48 + 1.88 = 3.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -198.1763548, upper bound: 198.1763548

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1400700, upper bound: 198.1546311
time: 0.81 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.75 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.71 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.71
Output dim: 0, lower bound: -198.1400700, upper bound: 198.1546311
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.71
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -73.8932495, 135.3371582, -75.5068207, 138.2489624, -212.1422119, 210.8439636
1: -65.9493561, 123.6492538, -67.3784637, 126.3198547, -192.2691650, 191.0277100
2: -57.7101402, 127.8353882, -58.9700966, 130.5795898, -188.2897034, 186.8054810
3: -90.7982178, 126.9703827, -92.7449951, 129.7379456, -220.5361633, 219.7153473
4: -70.8404388, 136.0842285, -72.3862000, 139.0076752, -209.8481140, 208.4704285

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.79 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.58 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -100.2753067, 184.2676697, -71.7902069, 131.6054993, -231.8807983, 256.0578613
1: -90.0813370, 169.2561340, -64.0946198, 120.2725677, -210.3538818, 233.3507385
2: -78.5410080, 174.6634674, -56.0710869, 124.3399887, -202.8809814, 230.7345581
3: -123.8081818, 173.3339539, -88.2992783, 123.4007797, -247.2089539, 261.6332092
4: -96.0769653, 185.9647369, -68.8296967, 132.3437958, -228.4207611, 254.7944336

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.83 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.98 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 0, lower bound: -198.1393660, upper bound: 198.1393660

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -73.8932495, 135.3371582, -73.8932495, 135.3371582, -209.2304077, 209.2304077
1: -65.9493561, 123.6492538, -65.9493561, 123.6492538, -189.5986023, 189.5986023
2: -57.7101402, 127.8353882, -57.7101402, 127.8353882, -185.5455170, 185.5455170
3: -90.7982178, 126.9703827, -90.7982178, 126.9703827, -217.7686005, 217.7686005
4: -70.8404388, 136.0842285, -70.8404388, 136.0842285, -206.9246674, 206.9246674

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1383619, upper bound: 198.1515884
time: 0.53 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1396325, upper bound: 198.1542815
time: 0.62 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -73.8932495, 135.3371582, -100.2753067, 184.2676697, -258.1608276, 235.6124573
1: -65.9493561, 123.6492538, -90.0813370, 169.2561340, -235.2054749, 213.7305756
2: -57.7101402, 127.8353882, -78.5410080, 174.6634674, -232.3736115, 206.3763885
3: -90.7982178, 126.9703827, -123.8081818, 173.3339539, -264.1321716, 250.7785492
4: -70.8404388, 136.0842285, -96.0769653, 185.9647369, -256.8051453, 232.1611938

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1314898, upper bound: 198.1462241
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1400700, upper bound: 198.1544491
time: 0.63 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -100.2753067, 184.2676697, -73.7518311, 135.0833740, -235.3586731, 258.0194702
1: -90.0813370, 169.2561340, -65.8236465, 123.4182816, -213.4995880, 235.0797729
2: -78.5410080, 174.6634674, -57.5982323, 127.5984879, -206.1394958, 232.2617035
3: -123.8081818, 173.3339539, -90.6277237, 126.7278290, -250.5360107, 263.9616699
4: -96.0769653, 185.9647369, -70.7023621, 135.8310852, -231.9080505, 256.6670837

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1375423, upper bound: 198.1342960
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1375423, upper bound: 198.1378750
time: 0.63 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -100.2753067, 184.2676697, -100.2753067, 184.2676697, -284.5429688, 284.5429688
1: -90.0813370, 169.2561340, -90.0813370, 169.2561340, -259.3374329, 259.3374634
2: -78.5410080, 174.6634674, -78.5410080, 174.6634674, -253.2044678, 253.2044678
3: -123.8081818, 173.3339539, -123.8081818, 173.3339539, -297.1421204, 297.1421204
4: -96.0769653, 185.9647369, -96.0769653, 185.9647369, -282.0416565, 282.0416565

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1375423, upper bound: 198.1342960
time: 0.56 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1375423, upper bound: 198.1378750
time: 0.58 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.52 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -198.1383619, upper bound: 198.1515884
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -198.1396325, upper bound: 198.1542815
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -198.1314898, upper bound: 198.1462241
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -198.1400700, upper bound: 198.1544491
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -198.1375423, upper bound: 198.1342960
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -198.1375423, upper bound: 198.1378750
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -198.1375423, upper bound: 198.1342960
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.52
Output dim: 0, lower bound: -198.1375423, upper bound: 198.1378750

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -71.7323227, 131.3703918, -73.8928528, 135.3364258, -207.0687561, 205.2632446
1: -64.0240555, 120.0572815, -65.9490128, 123.6485901, -187.6726379, 186.0062866
2: -56.0017052, 124.1496277, -57.7098465, 127.8347092, -183.8364105, 181.8594666
3: -88.2226944, 123.2192612, -90.7977448, 126.9697037, -215.1923981, 214.0169983
4: -68.7247696, 132.1839752, -70.8400650, 136.0835266, -204.8082733, 203.0240479

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1757534, upper bound: 198.1757534
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1757534, upper bound: 198.1758232
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -76.4018631, 139.7591095, -73.8287888, 135.2197571, -211.6216125, 213.5878906
1: -68.1784134, 127.7053299, -65.8921509, 123.5425034, -191.7208710, 193.5974731
2: -59.6843987, 132.0273438, -57.6596336, 127.7258224, -187.4102173, 189.6869812
3: -93.8784409, 131.2083588, -90.7212219, 126.8594513, -220.7378845, 221.9295654
4: -73.2558670, 140.6001587, -70.7780838, 135.9680481, -209.2239075, 211.3782349

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1758232, upper bound: 198.1760024
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1758232, upper bound: 198.1762223
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -67.0086594, 122.8483200, -99.4451752, 182.7548981, -249.7635040, 222.2934875
1: -59.8643456, 112.1512146, -89.3491287, 167.8705444, -227.7348938, 201.5003357
2: -52.3211174, 116.0761490, -77.8942490, 173.2454376, -225.5665436, 193.9703522
3: -82.5262222, 115.0166855, -122.8225937, 171.8946991, -254.4208984, 237.8392639
4: -64.1688080, 123.5948715, -95.2742233, 184.4654541, -248.6342621, 218.8690948

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 2

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -75.4596481, 138.1517944, -98.9066010, 181.7753143, -257.2349548, 237.0583954
1: -67.3544464, 126.0165710, -88.8536987, 166.9465332, -234.3009796, 214.8702698
2: -58.9000587, 130.4310303, -77.4583740, 172.3070526, -231.2071075, 207.8894043
3: -92.7729645, 129.5375519, -122.1375732, 170.9605103, -263.7334290, 251.6750946
4: -72.2363968, 138.9253235, -94.7378311, 183.4536285, -255.6900330, 233.6631470

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1385376, upper bound: 198.1526980
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1383619, upper bound: 198.1514387
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1396325, upper bound: 198.1539258
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -89.7139740, 165.4311218, -70.0654297, 128.4956818, -218.2096558, 235.4965515
1: -80.6646500, 152.0952911, -62.5611115, 117.4516983, -198.1163483, 214.6563873
2: -70.2311172, 157.0376587, -54.7142830, 121.4664001, -191.6975098, 211.7519379
3: -111.0168304, 155.4375763, -86.2261658, 120.4584579, -231.4752808, 241.6637268
4: -85.8804855, 167.0632324, -67.1387787, 129.2890930, -215.1695862, 234.2019958

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1508985, upper bound: 198.1350205
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1508985, upper bound: 198.1350383
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -97.0768967, 178.7394562, -70.5789642, 129.3165588, -226.3934479, 249.3183746
1: -87.1710892, 164.3710175, -62.9861069, 118.1981583, -205.3692474, 227.3571014
2: -75.9999466, 169.5014648, -55.1021233, 122.2218475, -198.2217712, 224.6035919
3: -119.7879181, 168.0469055, -86.8118210, 121.2797394, -241.0676422, 254.8587036
4: -92.9816666, 180.3123169, -67.6287918, 130.1087799, -223.0904236, 247.9411011

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1526980, upper bound: 198.1385376
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1526980, upper bound: 198.1386230
time: 1.07 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -89.7139740, 165.4311218, -96.5787430, 177.7145386, -267.4284973, 262.0098267
1: -80.6646500, 152.0952911, -86.7882233, 163.2918549, -243.9564972, 238.8835144
2: -70.2311172, 157.0376587, -75.6355591, 168.5328674, -238.7639771, 232.6732178
3: -111.0168304, 155.4375763, -119.3433533, 167.1166840, -278.1334839, 274.7808838
4: -85.8804855, 167.0632324, -92.5018082, 179.3859558, -265.2664490, 259.5649719

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -198.1341039, upper bound: 198.1341039
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -198.1341039, upper bound: 198.1342960
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -97.0768967, 178.7394562, -97.3873749, 179.0367737, -276.1136780, 276.1268311
1: -87.1710892, 164.3710175, -87.4799271, 164.4980927, -251.6691895, 251.8509521
2: -75.9999466, 169.5014648, -76.2599716, 169.7621460, -245.7620392, 245.7614288
3: -119.7879181, 168.0469055, -120.2710495, 168.4095917, -288.1975098, 288.3179626
4: -92.9816666, 180.3123169, -93.2749176, 180.7218018, -273.7034607, 273.5872192

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1342960, upper bound: 198.1375423
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1342960, upper bound: 198.1378750
time: 0.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.03 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -198.1757534, upper bound: 198.1757534
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -198.1757534, upper bound: 198.1758232
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -198.1758232, upper bound: 198.1760024
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -198.1758232, upper bound: 198.1762223
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -198.1383619, upper bound: 198.1514387
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -198.1396325, upper bound: 198.1539258
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -198.1508985, upper bound: 198.1350205
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -198.1508985, upper bound: 198.1350383
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -198.1526980, upper bound: 198.1385376
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -198.1526980, upper bound: 198.1386230
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.03
Output dim: 0, lower bound: -198.1341039, upper bound: 198.1341039
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.03
Output dim: 0, lower bound: -198.1341039, upper bound: 198.1342960
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -198.1342960, upper bound: 198.1375423
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.03
Output dim: 0, lower bound: -198.1342960, upper bound: 198.1378750

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -71.7323227, 131.3703918, -71.7323227, 131.3703918, -203.1027222, 203.1027222
1: -64.0240555, 120.0572815, -64.0240555, 120.0572815, -184.0813293, 184.0813293
2: -56.0017052, 124.1496277, -56.0017052, 124.1496277, -180.1513214, 180.1513214
3: -88.2226944, 123.2192612, -88.2226944, 123.2192612, -211.4419556, 211.4419556
4: -68.7247696, 132.1839752, -68.7247696, 132.1839752, -200.9087524, 200.9087524

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1752164, upper bound: 198.1744461
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1754323, upper bound: 198.1754323
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -71.7323227, 131.3703918, -76.4018631, 139.7591095, -211.4914246, 207.7722473
1: -64.0240555, 120.0572815, -68.1784134, 127.7053299, -191.7293854, 188.2356720
2: -56.0017052, 124.1496277, -59.6843987, 132.0273438, -188.0290527, 183.8340302
3: -88.2226944, 123.2192612, -93.8784409, 131.2083588, -219.4310608, 217.0977020
4: -68.7247696, 132.1839752, -73.2558670, 140.6001587, -209.3249207, 205.4398499

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1744461, upper bound: 198.1754656
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1754323, upper bound: 198.1755029
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -76.4018631, 139.7591095, -71.7323227, 131.3703918, -207.7722473, 211.4914246
1: -68.1784134, 127.7053299, -64.0240555, 120.0572815, -188.2356720, 191.7293854
2: -59.6843987, 132.0273438, -56.0017052, 124.1496277, -183.8340302, 188.0290527
3: -93.8784409, 131.2083588, -88.2226944, 123.2192612, -217.0977020, 219.4310608
4: -73.2558670, 140.6001587, -68.7247696, 132.1839752, -205.4398499, 209.3249207

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1754656, upper bound: 198.1752415
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1755029, upper bound: 198.1756953
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -76.4018631, 139.7591095, -76.4018631, 139.7591095, -216.1609802, 216.1609802
1: -68.1784134, 127.7053299, -68.1784134, 127.7053299, -195.8837433, 195.8837433
2: -59.6843987, 132.0273438, -59.6843987, 132.0273438, -191.7117462, 191.7117462
3: -93.8784409, 131.2083588, -93.8784409, 131.2083588, -225.0867920, 225.0867920
4: -73.2558670, 140.6001587, -73.2558670, 140.6001587, -213.8560181, 213.8560181

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1757585, upper bound: 198.1753987
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1753000, upper bound: 198.1753534
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -73.2202988, 134.0480499, -98.9061966, 181.7745667, -254.9948273, 232.9542542
1: -65.3604202, 122.3066254, -88.8533401, 166.9458466, -232.3062744, 211.1599731
2: -57.1332550, 126.6202850, -77.4580612, 172.3063812, -229.4396362, 204.0783081
3: -90.1140518, 125.6618958, -122.1370773, 170.9598389, -261.0738525, 247.7989807
4: -70.0512238, 134.8936462, -94.7374344, 183.4528809, -253.5041046, 229.6310730

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1346626, upper bound: 198.1497232
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1346626, upper bound: 198.1514387
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -77.4162979, 141.5731049, -98.8481445, 181.6690521, -259.0853271, 240.4212494
1: -69.1059418, 129.1897278, -88.8015671, 166.8496094, -235.9555511, 217.9912720
2: -60.4538040, 133.6945038, -77.4123077, 172.2077026, -232.6614990, 211.1068115
3: -95.1927185, 132.8453522, -122.0669022, 170.8604279, -266.0531616, 254.9122467
4: -74.1398239, 142.4512177, -94.6809616, 183.3477783, -257.4875793, 237.1321716

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1296802, upper bound: 198.1407992
time: 0.73 seconds

## Relational analysis of NS_A1_B2_A2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1395958, upper bound: 198.1539047
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -89.7139740, 165.4311218, -62.8091240, 115.5251770, -205.2391510, 228.2402496
1: -80.6646500, 152.0952911, -56.1326523, 105.6843643, -186.3490143, 208.2279205
2: -70.2311172, 157.0376587, -49.0318260, 109.3671722, -179.5982971, 206.0694733
3: -111.0168304, 155.4375763, -77.5500031, 108.0773239, -219.0941467, 232.9875336
4: -85.8804855, 167.0632324, -60.0997124, 116.3916931, -202.2721863, 227.1629486

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1459467, upper bound: 198.1332704
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1506516, upper bound: 198.1345695
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -89.7139740, 165.4311218, -69.0083542, 127.2241211, -216.9380951, 234.4394684
1: -80.6646500, 152.0952911, -61.6511650, 116.4282837, -197.0929260, 213.7464600
2: -70.2311172, 157.0376587, -53.9276848, 120.2318268, -190.4629517, 210.9653473
3: -111.0168304, 155.4375763, -84.8656769, 118.9597015, -229.9765320, 240.3032074
4: -85.8804855, 167.0632324, -66.2087097, 127.7416382, -213.6221313, 233.2719421

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1459467, upper bound: 198.1335205
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1459467, upper bound: 198.1345859
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -97.0768967, 178.7394562, -62.8091240, 115.5251770, -212.6020813, 241.5485840
1: -87.1710892, 164.3710175, -56.1326523, 105.6843643, -192.8554535, 220.5036469
2: -75.9999466, 169.5014648, -49.0318260, 109.3671722, -185.3670959, 218.5332794
3: -119.7879181, 168.0469055, -77.5500031, 108.0773239, -227.8652191, 245.5968628
4: -92.9816666, 180.3123169, -60.0997124, 116.3916931, -209.3733521, 240.4120331

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1468243, upper bound: 198.1356682
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1524789, upper bound: 198.1381125
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -97.0768967, 178.7394562, -69.0276108, 127.2570877, -224.3339691, 247.7670593
1: -87.1710892, 164.3710175, -61.6679955, 116.4586792, -203.6297607, 226.0390015
2: -75.9999466, 169.5014648, -53.9421806, 120.2624664, -196.2623749, 223.4436493
3: -119.7879181, 168.0469055, -84.8881531, 118.9919510, -238.7798309, 252.9350281
4: -92.9816666, 180.3123169, -66.2265930, 127.7745514, -220.7562256, 246.5388947

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1468243, upper bound: 198.1365733
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1524789, upper bound: 198.1381944
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -97.0768967, 178.7394562, -89.7139740, 165.4311218, -262.5079956, 268.4534302
1: -87.1710892, 164.3710175, -80.6646500, 152.0952911, -239.2663879, 245.0356598
2: -75.9999466, 169.5014648, -70.2311172, 157.0376587, -233.0375671, 239.7325745
3: -119.7879181, 168.0469055, -111.0168304, 155.4375763, -275.2254944, 279.0637207
4: -92.9816666, 180.3123169, -85.8804855, 167.0632324, -260.0448303, 266.1928101

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -198.1332202, upper bound: 198.1325523
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1332202, upper bound: 198.1371027
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -97.0768967, 178.7394562, -97.0768967, 178.7394562, -275.8163452, 275.8163452
1: -87.1710892, 164.3710175, -87.1710892, 164.3710175, -251.5421143, 251.5421143
2: -75.9999466, 169.5014648, -75.9999466, 169.5014648, -245.5013733, 245.5013733
3: -119.7879181, 168.0469055, -119.7879181, 168.0469055, -287.8348389, 287.8348389
4: -92.9816666, 180.3123169, -92.9816666, 180.3123169, -273.2939758, 273.2939758

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -198.1330475, upper bound: 198.1329309
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1338254, upper bound: 198.1374314
time: 0.91 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.84 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1752164, upper bound: 198.1744461
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1754323, upper bound: 198.1754323
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1744461, upper bound: 198.1754656
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1754323, upper bound: 198.1755029
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1754656, upper bound: 198.1752415
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1755029, upper bound: 198.1756953
NS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1757585, upper bound: 198.1753987
NS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1753000, upper bound: 198.1753534
NS_A1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1346626, upper bound: 198.1497232
NS_A1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1346626, upper bound: 198.1514387
NS_A1_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1296802, upper bound: 198.1407992
NS_A1_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1395958, upper bound: 198.1539047
NS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1459467, upper bound: 198.1332704
NS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1506516, upper bound: 198.1345695
NS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1459467, upper bound: 198.1335205
NS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1459467, upper bound: 198.1345859
NS_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1468243, upper bound: 198.1356682
NS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1524789, upper bound: 198.1381125
NS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1468243, upper bound: 198.1365733
NS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1524789, upper bound: 198.1381944
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1332202, upper bound: 198.1325523
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1332202, upper bound: 198.1371027
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1330475, upper bound: 198.1329309
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.84
Output dim: 0, lower bound: -198.1338254, upper bound: 198.1374314

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -60.7263298, 111.6992645, -68.0726624, 124.8268661, -185.5531769, 179.7719116
1: -54.2753906, 102.2124329, -60.7866364, 114.1278152, -168.4031982, 162.9990692
2: -47.3915520, 105.8053741, -53.1387863, 118.0545349, -165.4460907, 158.9441528
3: -75.0700607, 104.4469604, -83.8519516, 116.9861832, -192.0562439, 188.2988892
4: -58.0564804, 112.6244354, -65.1854630, 125.6828232, -183.7393036, 177.8098907

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1744435, upper bound: 198.1744435
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1744435, upper bound: 198.1744461
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -67.2720490, 124.0373383, -68.7675858, 125.9882355, -193.2602386, 192.8049316
1: -60.1041641, 113.5252609, -61.3726540, 115.1808167, -175.2849426, 174.8979187
2: -52.5611534, 117.2556915, -53.6707840, 119.1257172, -171.6868744, 170.9264679
3: -82.7915649, 115.9270477, -84.6495514, 118.1303635, -200.9219360, 200.5765991
4: -64.5064316, 124.5961914, -65.8554001, 126.8339844, -191.3404236, 190.4515991

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1744461, upper bound: 198.1752164
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1744461, upper bound: 198.1754323
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -68.0726624, 124.8268661, -64.9003983, 119.2250748, -187.2977295, 189.7272644
1: -60.7866364, 114.1278152, -58.0025406, 109.0720825, -169.8587036, 172.1303558
2: -53.1387863, 118.0545349, -50.6844482, 112.8709641, -166.0097198, 168.7389832
3: -83.8519516, 116.9861832, -80.1294022, 111.6289215, -195.4808655, 197.1155853
4: -65.1854630, 125.6828232, -62.1333542, 120.1638412, -185.3493042, 187.8161774

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1744435, upper bound: 198.1745080
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1751096, upper bound: 198.1754656
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -68.7675858, 125.9882355, -69.3731003, 127.7007446, -196.4683075, 195.3613129
1: -61.3726540, 115.1808167, -61.9543877, 116.8179703, -178.1906281, 177.1351471
2: -53.6707840, 119.1257172, -54.1951065, 120.7011871, -174.3719788, 173.3208313
3: -84.6495514, 118.1303635, -85.3355026, 119.4388046, -204.0883331, 203.4658661
4: -65.8554001, 126.8339844, -66.5297165, 128.3148193, -194.1702271, 193.3637085

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1755295, upper bound: 198.1745080
time: 1.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1755295, upper bound: 198.1755029
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -64.9003983, 119.2250748, -68.0726624, 124.8268661, -189.7272644, 187.2977295
1: -58.0025406, 109.0720825, -60.7866364, 114.1278152, -172.1303558, 169.8587036
2: -50.6844482, 112.8709641, -53.1387863, 118.0545349, -168.7389832, 166.0097198
3: -80.1294022, 111.6289215, -83.8519516, 116.9861832, -197.1155853, 195.4808655
4: -62.1333542, 120.1638412, -65.1854630, 125.6828232, -187.8161774, 185.3493042

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1745080, upper bound: 198.1751096
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1745080, upper bound: 198.1752415
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -69.3731003, 127.7007446, -68.7675858, 125.9882355, -195.3613129, 196.4683075
1: -61.9543877, 116.8179703, -61.3726540, 115.1808167, -177.1351471, 178.1906281
2: -54.1951065, 120.7011871, -53.6707840, 119.1257172, -173.3208313, 174.3719788
3: -85.3355026, 119.4388046, -84.6495514, 118.1303635, -203.4658661, 204.0883331
4: -66.5297165, 128.3148193, -65.8554001, 126.8339844, -193.3637085, 194.1702271

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1745080, upper bound: 198.1755295
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1745080, upper bound: 198.1756953
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -75.4220581, 137.9786072, -69.3130264, 126.9194717, -202.3415222, 207.2916260
1: -67.3118362, 126.0610046, -61.9138870, 115.8751678, -183.1869812, 187.9748840
2: -58.9169006, 130.3464050, -54.1315231, 119.9258575, -178.8427582, 184.4779053
3: -92.7006836, 129.5054321, -85.3620605, 118.9121475, -211.6128235, 214.8674774
4: -72.3053360, 138.8167877, -66.3831482, 127.7434845, -200.0488129, 205.1999207

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1753548, upper bound: 198.1753534
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1753548, upper bound: 198.1753534
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -74.9219360, 137.0415344, -77.4162979, 141.5731049, -216.4950409, 214.4578247
1: -66.8591537, 125.2085876, -69.1059418, 129.1897278, -196.0488739, 194.3145294
2: -58.5162659, 129.4824066, -60.4538040, 133.6945038, -192.2107697, 189.9362030
3: -92.1235199, 128.6247253, -95.1927185, 132.8453522, -224.9688568, 223.8174286
4: -71.8109512, 137.9114227, -74.1398239, 142.4512177, -214.2621613, 212.0512390

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1753548, upper bound: 198.1753534
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1753548, upper bound: 198.1753534
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -73.2202988, 134.0480499, -96.7323456, 177.8336029, -251.0538940, 230.7803955
1: -65.3604202, 122.3066254, -86.9098663, 163.3676605, -228.7280884, 209.2164917
2: -57.1332550, 126.6202850, -75.7419739, 168.6255646, -225.7588196, 202.3622437
3: -90.1140518, 125.6618958, -119.5076904, 167.2345886, -257.3485718, 245.1695862
4: -70.0512238, 134.8936462, -92.6160812, 179.5352020, -249.5864258, 227.5097351

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -73.2202988, 134.0480499, -101.1052856, 185.7148743, -258.9351196, 235.1533356
1: -65.3604202, 122.3066254, -90.7817688, 170.5618744, -235.9223022, 213.0883789
2: -57.1332550, 126.6202850, -79.1638031, 176.0216980, -233.1549530, 205.7840424
3: -90.1140518, 125.6618958, -124.7375793, 174.7095032, -264.8235474, 250.3994751
4: -70.0512238, 134.8936462, -96.8456497, 187.3998108, -257.4510498, 231.7392883

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -70.5096359, 129.3531647, -93.6525421, 172.4155273, -242.9251404, 223.0057068
1: -62.9827766, 118.1037064, -84.1816711, 158.4298401, -221.4125824, 202.2853546
2: -55.0660858, 122.3288651, -73.3362427, 163.5752869, -218.6413727, 195.6651001
3: -87.0344391, 121.2641296, -115.8478851, 162.0958405, -249.1302795, 237.1120148
4: -67.4557343, 130.3237915, -89.6595993, 174.1218872, -241.5776215, 219.9833832

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## BFS NS instance: NS_A1_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -73.9360886, 135.0950165, -97.9564667, 179.9969330, -253.9330139, 233.0514832
1: -65.9984970, 123.3587723, -88.0053101, 165.3378448, -231.3363342, 211.3640747
2: -57.7254639, 127.6470337, -76.7115936, 170.6276855, -228.3531494, 204.3586273
3: -90.9812241, 126.8377609, -120.9720535, 169.3176727, -260.2988892, 247.8098145
4: -70.7962418, 136.0540466, -93.8258820, 181.6760101, -252.4722595, 229.8799286

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1347520, upper bound: 198.1510330
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1347520, upper bound: 198.1539047
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -89.7135239, 165.4303131, -60.6556625, 111.5707245, -201.2842407, 226.0859680
1: -80.6642685, 152.0945282, -54.2129211, 102.0959702, -182.7602234, 206.3074493
2: -70.2307587, 157.0369110, -47.3361053, 105.6864548, -175.9172058, 204.3730164
3: -111.0163193, 155.4368134, -74.9860916, 104.3239899, -215.3403015, 230.4228821
4: -85.8800507, 167.0624390, -57.9867744, 112.4983292, -198.3783875, 225.0491943

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1264365
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1332704
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -89.6713638, 165.3533173, -64.8144684, 119.0687866, -208.7401276, 230.1677856
1: -80.6267548, 152.0242004, -57.9266968, 108.9306641, -189.5573883, 209.9508972
2: -70.1974945, 156.9648438, -50.6168175, 112.7266388, -182.9241333, 207.5816650
3: -110.9649887, 155.3642120, -80.0274353, 111.4796524, -222.4446259, 235.3916473
4: -85.8391571, 166.9855957, -62.0486107, 120.0108566, -205.8500061, 229.0342102

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1465860, upper bound: 198.1266538
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1465860, upper bound: 198.1345705
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -89.7135239, 165.4303131, -67.1973801, 123.9003677, -213.6138916, 232.6276855
1: -80.6642685, 152.0945282, -60.0388756, 113.3990707, -194.0633392, 212.1333923
2: -70.2307587, 157.0369110, -52.5032768, 117.1275482, -187.3583069, 209.5401917
3: -111.0163193, 155.4368134, -82.7038040, 115.7961426, -226.8124695, 238.1405792
4: -85.8800507, 167.0624390, -64.4340057, 124.4612274, -210.3412781, 231.4964447

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1264980
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1335205
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -89.6713638, 165.3533173, -69.1258545, 127.2678833, -216.9392395, 234.4791718
1: -80.6267548, 152.0242004, -61.7384262, 116.4186554, -197.0453796, 213.7626343
2: -70.1974945, 156.9648438, -54.0048790, 120.2963791, -190.4938660, 210.9697113
3: -110.9649887, 155.3642120, -85.0449295, 119.0194321, -229.9844055, 240.4091187
4: -85.8391571, 166.9855957, -66.2964706, 127.8818817, -213.7210083, 233.2820587

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1461661, upper bound: 198.1266457
time: 0.93 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1461661, upper bound: 198.1345859
time: 0.57 seconds

## BFS NS instance: NS_A2_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -97.0765686, 178.7388611, -60.6556625, 111.5707245, -208.6472931, 239.3945312
1: -87.1708069, 164.3704834, -54.2129211, 102.0959702, -189.2667847, 218.5834045
2: -75.9996796, 169.5008850, -47.3361053, 105.6864548, -181.6861115, 216.8369904
3: -119.7875290, 168.0463257, -74.9860916, 104.3239899, -224.1114502, 243.0324097
4: -92.9813538, 180.3117371, -57.9867744, 112.4983292, -205.4796753, 238.2984924

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1453740, upper bound: 198.1324837
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1453740, upper bound: 198.1356682
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -96.9971619, 178.5940552, -64.8144684, 119.0687866, -216.0659180, 243.4085236
1: -87.0999527, 164.2385864, -57.9266968, 108.9306641, -196.0305939, 222.1652832
2: -75.9370575, 169.3654480, -50.6168175, 112.7266388, -188.6636658, 219.9822693
3: -119.6907883, 167.9101562, -80.0274353, 111.4796524, -231.1704407, 247.9375916
4: -92.9044647, 180.1673584, -62.0486107, 120.0108566, -212.9153137, 242.2159729

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1465860, upper bound: 198.1335094
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1465860, upper bound: 198.1381125
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -97.0765686, 178.7388611, -67.1973801, 123.9003677, -220.9769287, 245.9362335
1: -87.1708069, 164.3704834, -60.0388756, 113.3990707, -200.5698853, 224.4093628
2: -75.9996796, 169.5008850, -52.5032768, 117.1275482, -193.1272125, 222.0041656
3: -119.7875290, 168.0463257, -82.7038040, 115.7961426, -235.5836334, 250.7501221
4: -92.9813538, 180.3117371, -64.4340057, 124.4612274, -217.4425812, 244.7457428

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1490727, upper bound: 198.1334611
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1490727, upper bound: 198.1365733
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -96.9971619, 178.5940552, -69.2396469, 127.4618378, -224.4589691, 247.8336945
1: -87.0999527, 164.2385864, -61.8369408, 116.5977936, -203.6977539, 226.0755157
2: -75.9370575, 169.3654480, -54.0907745, 120.4764404, -196.4134521, 223.4562225
3: -119.6907883, 167.9101562, -85.1761932, 119.2094421, -238.9002228, 253.0863342
4: -92.9044647, 180.1673584, -66.4014816, 128.0751495, -220.9796143, 246.5688477

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1502503, upper bound: 198.1335995
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1502503, upper bound: 198.1381944
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -97.0379410, 178.5511017, -89.6713638, 165.3533173, -262.3912659, 268.2224731
1: -87.1212387, 164.1400452, -80.6267548, 152.0242004, -239.1454315, 244.7667847
2: -75.9528656, 169.3253326, -70.1974945, 156.9648438, -232.9177094, 239.5228271
3: -119.7224197, 167.8793488, -110.9649887, 155.3642120, -275.0866394, 278.8442993
4: -92.9256363, 180.1524506, -85.8391571, 166.9855957, -259.9112244, 265.9915466

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -198.1259325, upper bound: 198.1351559
time: 0.80 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1259325, upper bound: 198.1371027
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -97.0379410, 178.5511017, -96.9971619, 178.5940552, -275.6319580, 275.5482788
1: -87.1212387, 164.1400452, -87.0999527, 164.2385864, -251.3598175, 251.2399902
2: -75.9528656, 169.3253326, -75.9370575, 169.3654480, -245.3183136, 245.2623596
3: -119.7224197, 167.8793488, -119.6907883, 167.9101562, -287.6325684, 287.5700378
4: -92.9256363, 180.1524506, -92.9044647, 180.1673584, -273.0929871, 273.0569153

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -198.1326707, upper bound: 198.1365690
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1326707, upper bound: 198.1374314
time: 1.01 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.30 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1744435, upper bound: 198.1744435
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1744435, upper bound: 198.1744461
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1744461, upper bound: 198.1752164
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1744461, upper bound: 198.1754323
NS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1744435, upper bound: 198.1745080
NS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1751096, upper bound: 198.1754656
NS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1755295, upper bound: 198.1745080
NS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1755295, upper bound: 198.1755029
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1745080, upper bound: 198.1751096
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1745080, upper bound: 198.1752415
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1745080, upper bound: 198.1755295
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1745080, upper bound: 198.1756953
NS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1753548, upper bound: 198.1753534
NS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1753548, upper bound: 198.1753534
NS_A1_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1753548, upper bound: 198.1753534
NS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1753548, upper bound: 198.1753534
NS_A1_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1347520, upper bound: 198.1510330
NS_A1_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1347520, upper bound: 198.1539047
NS_A2_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1264365
NS_A2_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1332704
NS_A2_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1465860, upper bound: 198.1266538
NS_A2_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1465860, upper bound: 198.1345705
NS_A2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1264980
NS_A2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1441503, upper bound: 198.1335205
NS_A2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1461661, upper bound: 198.1266457
NS_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1461661, upper bound: 198.1345859
NS_A2_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1453740, upper bound: 198.1324837
NS_A2_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1453740, upper bound: 198.1356682
NS_A2_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1465860, upper bound: 198.1335094
NS_A2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1465860, upper bound: 198.1381125
NS_A2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1490727, upper bound: 198.1334611
NS_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1490727, upper bound: 198.1365733
NS_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1502503, upper bound: 198.1335995
NS_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1502503, upper bound: 198.1381944
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1259325, upper bound: 198.1351559
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1259325, upper bound: 198.1371027
NS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1326707, upper bound: 198.1365690
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.30
Output dim: 0, lower bound: -198.1326707, upper bound: 198.1374314

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -60.7263298, 111.6992645, -60.7263298, 111.6992645, -172.4255676, 172.4255676
1: -54.2753906, 102.2124329, -54.2753906, 102.2124329, -156.4878235, 156.4878235
2: -47.3915520, 105.8053741, -47.3915520, 105.8053741, -153.1969299, 153.1969299
3: -75.0700607, 104.4469604, -75.0700607, 104.4469604, -179.5170288, 179.5170288
4: -58.0564804, 112.6244354, -58.0564804, 112.6244354, -170.6809082, 170.6809082

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1739073, upper bound: 198.1661032
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1656348, upper bound: 198.1656348
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -60.7263298, 111.6992645, -67.2720490, 124.0373383, -184.7636719, 178.9712982
1: -54.2753906, 102.2124329, -60.1041641, 113.5252609, -167.8006592, 162.3165894
2: -47.3915520, 105.8053741, -52.5611534, 117.2556915, -164.6472321, 158.3665161
3: -75.0700607, 104.4469604, -82.7915649, 115.9270477, -190.9971008, 187.2385101
4: -58.0564804, 112.6244354, -64.5064316, 124.5961914, -182.6526794, 177.1308594

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1661032, upper bound: 198.1743258
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1656348, upper bound: 198.1660978
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -67.2720490, 124.0373383, -60.7263298, 111.6992645, -178.9712982, 184.7636719
1: -60.1041641, 113.5252609, -54.2753906, 102.2124329, -162.3165894, 167.8006592
2: -52.5611534, 117.2556915, -47.3915520, 105.8053741, -158.3665161, 164.6472321
3: -82.7915649, 115.9270477, -75.0700607, 104.4469604, -187.2385101, 190.9971008
4: -64.5064316, 124.5961914, -58.0564804, 112.6244354, -177.1308594, 182.6526794

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1743258, upper bound: 198.1749139
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1660978, upper bound: 198.1743843
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -67.2720490, 124.0373383, -67.2720490, 124.0373383, -191.3093872, 191.3093872
1: -60.1041641, 113.5252609, -60.1041641, 113.5252609, -173.6294250, 173.6294250
2: -52.5611534, 117.2556915, -52.5611534, 117.2556915, -169.8168335, 169.8168335
3: -82.7915649, 115.9270477, -82.7915649, 115.9270477, -198.7186127, 198.7186127
4: -64.5064316, 124.5961914, -64.5064316, 124.5961914, -189.1026306, 189.1026306

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1661129, upper bound: 198.1753908
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1660978, upper bound: 198.1752389
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -60.7263298, 111.6992645, -64.9003983, 119.2250748, -179.9514008, 176.5996552
1: -54.2753906, 102.2124329, -58.0025406, 109.0720825, -163.3474274, 160.2149658
2: -47.3915520, 105.8053741, -50.6844482, 112.8709641, -160.2624817, 156.4898224
3: -75.0700607, 104.4469604, -80.1294022, 111.6289215, -186.6989746, 184.5763550
4: -58.0564804, 112.6244354, -62.1333542, 120.1638412, -178.2203217, 174.7577820

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1747356, upper bound: 198.1661961
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1710679, upper bound: 198.1660540
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -67.2720490, 124.0373383, -64.9003983, 119.2250748, -186.4971313, 188.9377441
1: -60.1041641, 113.5252609, -58.0025406, 109.0720825, -169.1762085, 171.5278015
2: -52.5611534, 117.2556915, -50.6844482, 112.8709641, -165.4320831, 167.9401245
3: -82.7915649, 115.9270477, -80.1294022, 111.6289215, -194.4204865, 196.0564575
4: -64.5064316, 124.5961914, -62.1333542, 120.1638412, -184.6702728, 186.7295532

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1747356, upper bound: 198.1661961
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1710679, upper bound: 198.1748954
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -60.7263298, 111.6992645, -69.3731003, 127.7007446, -188.4270630, 181.0723572
1: -54.2753906, 102.2124329, -61.9543877, 116.8179703, -171.0933533, 164.1668243
2: -47.3915520, 105.8053741, -54.1951065, 120.7011871, -168.0927429, 160.0004730
3: -75.0700607, 104.4469604, -85.3355026, 119.4388046, -194.5088654, 189.7824707
4: -58.0564804, 112.6244354, -66.5297165, 128.3148193, -186.3713074, 179.1541443

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1707830, upper bound: 198.1743354
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1710679, upper bound: 198.1661602
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -67.2720490, 124.0373383, -69.3731003, 127.7007446, -194.9727936, 193.4104309
1: -60.1041641, 113.5252609, -61.9543877, 116.8179703, -176.9221344, 175.4796448
2: -52.5611534, 117.2556915, -54.1951065, 120.7011871, -173.2623444, 171.4507904
3: -82.7915649, 115.9270477, -85.3355026, 119.4388046, -202.2303467, 201.2625427
4: -64.5064316, 124.5961914, -66.5297165, 128.3148193, -192.8212585, 191.1259155

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1707830, upper bound: 198.1754459
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1710679, upper bound: 198.1753000
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -64.9003983, 119.2250748, -60.7263298, 111.6992645, -176.5996552, 179.9514008
1: -58.0025406, 109.0720825, -54.2753906, 102.2124329, -160.2149658, 163.3474274
2: -50.6844482, 112.8709641, -47.3915520, 105.8053741, -156.4898224, 160.2624817
3: -80.1294022, 111.6289215, -75.0700607, 104.4469604, -184.5763550, 186.6989746
4: -62.1333542, 120.1638412, -58.0564804, 112.6244354, -174.7577820, 178.2203217

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1661961, upper bound: 198.1747356
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1660540, upper bound: 198.1710679
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -64.9003983, 119.2250748, -67.2720490, 124.0373383, -188.9377441, 186.4971313
1: -58.0025406, 109.0720825, -60.1041641, 113.5252609, -171.5278015, 169.1762085
2: -50.6844482, 112.8709641, -52.5611534, 117.2556915, -167.9401245, 165.4320831
3: -80.1294022, 111.6289215, -82.7915649, 115.9270477, -196.0564575, 194.4204865
4: -62.1333542, 120.1638412, -64.5064316, 124.5961914, -186.7295532, 184.6702728

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1661961, upper bound: 198.1752415
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1660540, upper bound: 198.1724062
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -69.3731003, 127.7007446, -60.7263298, 111.6992645, -181.0723572, 188.4270630
1: -61.9543877, 116.8179703, -54.2753906, 102.2124329, -164.1668243, 171.0933533
2: -54.1951065, 120.7011871, -47.3915520, 105.8053741, -160.0004730, 168.0927429
3: -85.3355026, 119.4388046, -75.0700607, 104.4469604, -189.7824707, 194.5088654
4: -66.5297165, 128.3148193, -58.0564804, 112.6244354, -179.1541443, 186.3713074

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1743354, upper bound: 198.1749119
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1661602, upper bound: 198.1744498
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -69.3731003, 127.7007446, -67.2720490, 124.0373383, -193.4104309, 194.9727936
1: -61.9543877, 116.8179703, -60.1041641, 113.5252609, -175.4796448, 176.9221344
2: -54.1951065, 120.7011871, -52.5611534, 117.2556915, -171.4507904, 173.2623444
3: -85.3355026, 119.4388046, -82.7915649, 115.9270477, -201.2625427, 202.2303467
4: -66.5297165, 128.3148193, -64.5064316, 124.5961914, -191.1259155, 192.8212585

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1743258, upper bound: 198.1752950
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1661602, upper bound: 198.1752444
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -69.3130264, 126.9194717, -69.3130264, 126.9194717, -196.2324982, 196.2324982
1: -61.9138870, 115.8751678, -61.9138870, 115.8751678, -177.7890472, 177.7890472
2: -54.1315231, 119.9258575, -54.1315231, 119.9258575, -174.0573578, 174.0573578
3: -85.3620605, 118.9121475, -85.3620605, 118.9121475, -204.2742004, 204.2742004
4: -66.3831482, 127.7434845, -66.3831482, 127.7434845, -194.1265869, 194.1265869

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1756327, upper bound: 198.1753685
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1756327, upper bound: 198.1753971
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -77.4162979, 141.5731049, -69.3130264, 126.9194717, -204.3357544, 210.8861389
1: -69.1059418, 129.1897278, -61.9138870, 115.8751678, -184.9811096, 191.1036072
2: -60.4538040, 133.6945038, -54.1315231, 119.9258575, -180.3796692, 187.8259888
3: -95.1927185, 132.8453522, -85.3620605, 118.9121475, -214.1048584, 218.2073975
4: -74.1398239, 142.4512177, -66.3831482, 127.7434845, -201.8832855, 208.8343506

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1744821, upper bound: 198.1741432
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1758324, upper bound: 198.1753971
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -69.3130264, 126.9194717, -77.4162979, 141.5731049, -210.8861389, 204.3357544
1: -61.9138870, 115.8751678, -69.1059418, 129.1897278, -191.1036072, 184.9811096
2: -54.1315231, 119.9258575, -60.4538040, 133.6945038, -187.8259888, 180.3796692
3: -85.3620605, 118.9121475, -95.1927185, 132.8453522, -218.2073975, 214.1048584
4: -66.3831482, 127.7434845, -74.1398239, 142.4512177, -208.8343506, 201.8832855

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1741319, upper bound: 198.1752118
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1753526, upper bound: 198.1753516
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -77.4162979, 141.5731049, -77.4162979, 141.5731049, -218.9894104, 218.9894104
1: -69.1059418, 129.1897278, -69.1059418, 129.1897278, -198.2956696, 198.2956696
2: -60.4538040, 133.6945038, -60.4538040, 133.6945038, -194.1483002, 194.1483002
3: -95.1927185, 132.8453522, -95.1927185, 132.8453522, -228.0380707, 228.0380707
4: -74.1398239, 142.4512177, -74.1398239, 142.4512177, -216.5910339, 216.5910339

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1741319, upper bound: 198.1752118
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -198.1753526, upper bound: 198.1753516
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -73.9360886, 135.0950165, -95.8470306, 176.1741333, -250.1102295, 230.9420471
1: -65.9984970, 123.3587723, -86.1194992, 161.8654785, -227.8639832, 209.4782715
2: -57.7254639, 127.6470337, -75.0460815, 167.0563202, -224.7817841, 202.6931152
3: -90.9812241, 126.8377609, -118.4201660, 165.7026978, -256.6839294, 245.2579346
4: -70.7962418, 136.0540466, -91.7669144, 177.8738708, -248.6701050, 227.8209534

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -73.9360886, 135.0950165, -100.2224884, 184.0552216, -257.9913025, 235.3174896
1: -65.9984970, 123.3587723, -89.9931641, 169.0628052, -235.0612946, 213.3519287
2: -57.7254639, 127.6470337, -78.4707184, 174.4548950, -232.1803436, 206.1177521
3: -90.9812241, 126.8377609, -123.6586533, 173.1781158, -264.1593323, 250.4964142
4: -70.7962418, 136.0540466, -95.9979248, 185.7450104, -256.5412598, 232.0519562

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A2_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -87.4276657, 161.2543640, -60.6556625, 111.5707245, -198.9983826, 221.9100189
1: -78.6254196, 148.2930603, -54.2129211, 102.0959702, -180.7213898, 202.5059814
2: -68.4286423, 153.1336823, -47.3361053, 105.6864548, -174.1150970, 200.4697723
3: -108.2343979, 151.4825134, -74.9860916, 104.3239899, -212.5583649, 226.4685974
4: -83.6595383, 162.9042053, -57.9867744, 112.4983292, -196.1578674, 220.8909760

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -91.5342484, 168.7243958, -60.6556625, 111.5707245, -203.1049805, 229.3800354
1: -82.2651749, 155.1216583, -54.2129211, 102.0959702, -184.3611145, 209.3345795
2: -71.6478653, 160.1437988, -47.3361053, 105.6864548, -177.3343201, 207.4799042
3: -113.1501312, 158.5738373, -74.9860916, 104.3239899, -217.4740906, 233.5599060
4: -87.6395721, 170.3393250, -57.9867744, 112.4983292, -200.1378632, 228.3260956

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -87.4276657, 161.2543640, -64.8144684, 119.0687866, -206.4964142, 226.0688324
1: -78.6254196, 148.2930603, -57.9266968, 108.9306641, -187.5560608, 206.2197571
2: -68.4286423, 153.1336823, -50.6168175, 112.7266388, -181.1552734, 203.7505035
3: -108.2343979, 151.4825134, -80.0274353, 111.4796524, -219.7140503, 231.5099487
4: -83.6595383, 162.9042053, -62.0486107, 120.0108566, -203.6703949, 224.9528198

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -91.5342484, 168.7243958, -64.8144684, 119.0687866, -210.6029968, 233.5388641
1: -82.2651749, 155.1216583, -57.9266968, 108.9306641, -191.1957703, 213.0483551
2: -71.6478653, 160.1437988, -50.6168175, 112.7266388, -184.3745117, 210.7606201
3: -113.1501312, 158.5738373, -80.0274353, 111.4796524, -224.6297760, 238.6012573
4: -87.6395721, 170.3393250, -62.0486107, 120.0108566, -207.6504059, 232.3879395

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -87.4276657, 161.2543640, -67.1973801, 123.9003677, -211.3280182, 228.4517365
1: -78.6254196, 148.2930603, -60.0388756, 113.3990707, -192.0244904, 208.3319397
2: -68.4286423, 153.1336823, -52.5032768, 117.1275482, -185.5561676, 205.6369476
3: -108.2343979, 151.4825134, -82.7038040, 115.7961426, -224.0305481, 234.1862946
4: -83.6595383, 162.9042053, -64.4340057, 124.4612274, -208.1207581, 227.3382111

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -91.5342484, 168.7243958, -67.1973801, 123.9003677, -215.4346161, 235.9217529
1: -82.2651749, 155.1216583, -60.0388756, 113.3990707, -195.6642151, 215.1605225
2: -71.6478653, 160.1437988, -52.5032768, 117.1275482, -188.7754211, 212.6470795
3: -113.1501312, 158.5738373, -82.7038040, 115.7961426, -228.9462738, 241.2776031
4: -87.6395721, 170.3393250, -64.4340057, 124.4612274, -212.1007843, 234.7733307

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -87.4276657, 161.2543640, -69.1258545, 127.2678833, -214.6955566, 230.3802185
1: -78.6254196, 148.2930603, -61.7384262, 116.4186554, -195.0440521, 210.0314941
2: -68.4286423, 153.1336823, -54.0048790, 120.2963791, -188.7250061, 207.1385498
3: -108.2343979, 151.4825134, -85.0449295, 119.0194321, -227.2538147, 236.5274200
4: -83.6595383, 162.9042053, -66.2964706, 127.8818817, -211.5414124, 229.2006836

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -91.5342484, 168.7243958, -69.1258545, 127.2678833, -218.8021240, 237.8502350
1: -82.2651749, 155.1216583, -61.7384262, 116.4186554, -198.6837616, 216.8600769
2: -71.6478653, 160.1437988, -54.0048790, 120.2963791, -191.9442444, 214.1486816
3: -113.1501312, 158.5738373, -85.0449295, 119.0194321, -232.1695557, 243.6187286
4: -87.6395721, 170.3393250, -66.2964706, 127.8818817, -215.5214081, 236.6358032

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -95.1811142, 175.2666473, -60.6556625, 111.5707245, -206.7518311, 235.9223022
1: -85.4842682, 161.2029877, -54.2129211, 102.0959702, -187.5802307, 215.4159088
2: -74.5097961, 166.2485199, -47.3361053, 105.6864548, -180.1962585, 213.5846252
3: -117.4889450, 164.7580414, -74.9860916, 104.3239899, -221.8129272, 239.7441254
4: -91.1446838, 176.8563690, -57.9867744, 112.4983292, -203.6429901, 234.8431091

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -97.0379410, 178.5511017, -60.6556625, 111.5707245, -208.6086426, 239.2067566
1: -87.1212387, 164.1400452, -54.2129211, 102.0959702, -189.2172089, 218.3529663
2: -75.9528656, 169.3253326, -47.3361053, 105.6864548, -181.6393127, 216.6614380
3: -119.7224197, 167.8793488, -74.9860916, 104.3239899, -224.0463867, 242.8654327
4: -92.9256363, 180.1524506, -57.9867744, 112.4983292, -205.4239655, 238.1392059

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -95.1811142, 175.2666473, -64.8144684, 119.0687866, -214.2498779, 240.0811157
1: -85.4842682, 161.2029877, -57.9266968, 108.9306641, -194.4149170, 219.1296844
2: -74.5097961, 166.2485199, -50.6168175, 112.7266388, -187.2364349, 216.8653412
3: -117.4889450, 164.7580414, -80.0274353, 111.4796524, -228.9685974, 244.7854767
4: -91.1446838, 176.8563690, -62.0486107, 120.0108566, -211.1555328, 238.9049835

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -97.0379410, 178.5511017, -64.8144684, 119.0687866, -216.1066895, 243.3655701
1: -87.1212387, 164.1400452, -57.9266968, 108.9306641, -196.0518799, 222.0667419
2: -75.9528656, 169.3253326, -50.6168175, 112.7266388, -188.6795044, 219.9421539
3: -119.7224197, 167.8793488, -80.0274353, 111.4796524, -231.2020721, 247.9067688
4: -92.9256363, 180.1524506, -62.0486107, 120.0108566, -212.9364929, 242.2010651

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -95.1811142, 175.2666473, -67.1973801, 123.9003677, -219.0814819, 242.4640198
1: -85.4842682, 161.2029877, -60.0388756, 113.3990707, -198.8833313, 221.2418671
2: -74.5097961, 166.2485199, -52.5032768, 117.1275482, -191.6373444, 218.7518005
3: -117.4889450, 164.7580414, -82.7038040, 115.7961426, -233.2850952, 247.4618225
4: -91.1446838, 176.8563690, -64.4340057, 124.4612274, -215.6059113, 241.2903748

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -97.0379410, 178.5511017, -67.1973801, 123.9003677, -220.9382935, 245.7484741
1: -87.1212387, 164.1400452, -60.0388756, 113.3990707, -200.5203094, 224.1788940
2: -75.9528656, 169.3253326, -52.5032768, 117.1275482, -193.0804138, 221.8286133
3: -119.7224197, 167.8793488, -82.7038040, 115.7961426, -235.5185547, 250.5831146
4: -92.9256363, 180.1524506, -64.4340057, 124.4612274, -217.3868713, 244.5864563

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -95.1811142, 175.2666473, -69.2396469, 127.4618378, -222.6429443, 244.5062866
1: -85.4842682, 161.2029877, -61.8369408, 116.5977936, -202.0820618, 223.0399323
2: -74.5097961, 166.2485199, -54.0907745, 120.4764404, -194.9862366, 220.3392944
3: -117.4889450, 164.7580414, -85.1761932, 119.2094421, -236.6983795, 249.9342041
4: -91.1446838, 176.8563690, -66.4014816, 128.0751495, -219.2198334, 243.2578430

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -97.0379410, 178.5511017, -69.2396469, 127.4618378, -224.4997559, 247.7907410
1: -87.1212387, 164.1400452, -61.8369408, 116.5977936, -203.7190247, 225.9769592
2: -75.9528656, 169.3253326, -54.0907745, 120.4764404, -196.4293060, 223.4161072
3: -119.7224197, 167.8793488, -85.1761932, 119.2094421, -238.9318542, 253.0555115
4: -92.9256363, 180.1524506, -66.4014816, 128.0751495, -221.0007935, 246.5539246

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 17

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -97.0379410, 178.5511017, -91.5342484, 168.7243958, -265.7622681, 270.0853577
1: -87.1212387, 164.1400452, -82.2651749, 155.1216583, -242.2428894, 246.4051666
2: -75.9528656, 169.3253326, -71.6478653, 160.1437988, -236.0966644, 240.9732056
3: -119.7224197, 167.8793488, -113.1501312, 158.5738373, -278.2962341, 281.0294800
4: -92.9256363, 180.1524506, -87.6395721, 170.3393250, -263.2649536, 267.7919922

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 19

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -97.0379410, 178.5511017, -97.0379410, 178.5511017, -275.5890503, 275.5890503
1: -87.1212387, 164.1400452, -87.1212387, 164.1400452, -251.2612610, 251.2612610
2: -75.9528656, 169.3253326, -75.9528656, 169.3253326, -245.2781982, 245.2781982
3: -119.7224197, 167.8793488, -119.7224197, 167.8793488, -287.6017456, 287.6017456
4: -92.9256363, 180.1524506, -92.9256363, 180.1524506, -273.0780945, 273.0780945

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 21

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.36 + 261.90 = 265.26 seconds
