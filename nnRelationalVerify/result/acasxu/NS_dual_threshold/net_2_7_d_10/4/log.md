## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_7.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 4)
Time budget: 420 seconds
Split limit: 100
Threshold: 65.1166706475


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-362.6475830, 502.5717163, -362.6475830, 502.5717163, -865.2191772, 865.2191772)
1: (-47.5883713, 41.3081436, -47.5883713, 41.3081436, -88.8964996, 88.8964996)
2: (-25.9062366, 47.7711182, -25.9062366, 47.7711182, -73.6773529, 73.6773529)
3: (-20.5445766, 48.1182556, -20.5445766, 48.1182556, -68.6628342, 68.6628342)
4: (-31.2747822, 40.6671028, -31.2747822, 40.6671028, -71.9418869, 71.9418869)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.51 + 1.69 = 4.20 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -65.1818525, upper bound: 65.1818525

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1730331, upper bound: 65.1766414
time: 0.58 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1727647, upper bound: 65.1769633
time: 0.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.40 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 4, lower bound: -65.1730331, upper bound: 65.1766414
NS_B2, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 4, lower bound: -65.1727647, upper bound: 65.1769633

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -358.3995056, 495.6178894, -266.5393982, 341.3458252, -699.7452393, 762.1572266
1: -46.9103851, 40.7917747, -31.8348675, 29.4663811, -76.3767700, 72.6266403
2: -25.5727177, 47.1101761, -18.3058014, 32.4970741, -58.0697899, 65.4159775
3: -20.2989616, 47.4560127, -15.0378723, 32.8243256, -53.1232872, 62.4938850
4: -30.8767471, 40.1130981, -22.0716648, 27.9611950, -58.8379288, 62.1847610

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1724066, upper bound: 65.1724066
time: 0.58 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1724066, upper bound: 65.1766051
time: 0.57 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -354.7922668, 492.4851074, -583.4630737, 744.9866943, -1099.7789307, 1075.9481201
1: -46.6498642, 40.4537964, -68.8743057, 64.9022522, -111.5521088, 109.3280945
2: -25.3681698, 46.7910461, -40.3570328, 70.6871948, -96.0553665, 87.1480637
3: -20.0866470, 47.1575775, -33.4847565, 71.3168869, -91.4035339, 80.6423340
4: -30.6123447, 39.8508224, -48.2948952, 61.0561142, -91.6684418, 88.1457062

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1766051, upper bound: 65.1727647
time: 0.57 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1766051, upper bound: 65.1769633
time: 0.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.72 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 4, lower bound: -65.1724066, upper bound: 65.1724066
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 4, lower bound: -65.1724066, upper bound: 65.1766051
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 4, lower bound: -65.1766051, upper bound: 65.1727647
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 3.72
Output dim: 4, lower bound: -65.1766051, upper bound: 65.1769633

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -266.5393982, 341.3458252, -266.5393982, 341.3458252, -607.8851929, 607.8851929
1: -31.8348675, 29.4663811, -31.8348675, 29.4663811, -61.3012428, 61.3012466
2: -18.3058014, 32.4970741, -18.3058014, 32.4970741, -50.8028755, 50.8028755
3: -15.0378723, 32.8243256, -15.0378723, 32.8243256, -47.8621979, 47.8621979
4: -22.0716648, 27.9611950, -22.0716648, 27.9611950, -50.0328560, 50.0328560

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1719784, upper bound: 65.1722003
time: 0.53 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1717722, upper bound: 65.1717722
time: 0.54 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -583.4630737, 744.9866943, -266.5393982, 341.3458252, -924.8088989, 1011.5261230
1: -68.8743057, 64.9022522, -31.8348675, 29.4663811, -98.3406830, 96.7371063
2: -40.3570328, 70.6871948, -18.3058014, 32.4970741, -72.8540955, 88.9929962
3: -33.4847565, 71.3168869, -15.0378723, 32.8243256, -66.3090668, 86.3547592
4: -48.2948952, 61.0561142, -22.0716648, 27.9611950, -76.2560806, 83.1277695

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1695340, upper bound: 65.1756423
time: 0.57 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1695340, upper bound: 65.1766414
time: 0.61 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -266.5393982, 341.3458252, -583.4630737, 744.9866943, -1011.5261230, 924.8088379
1: -31.8348675, 29.4663811, -68.8743057, 64.9022522, -96.7371063, 98.3406830
2: -18.3058014, 32.4970741, -40.3570328, 70.6871948, -88.9929962, 72.8540955
3: -15.0378723, 32.8243256, -33.4847565, 71.3168869, -86.3547592, 66.3090668
4: -22.0716648, 27.9611950, -48.2948952, 61.0561142, -83.1277695, 76.2560806

Time for backsubstitution: 2.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1719533, upper bound: 65.1718108
time: 0.61 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1724066, upper bound: 65.1727647
time: 0.59 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -583.4630737, 744.9866943, -583.4630737, 744.9866943, -1328.4497070, 1328.4497070
1: -68.8743057, 64.9022522, -68.8743057, 64.9022522, -133.7765503, 133.7765503
2: -40.3570328, 70.6871948, -40.3570328, 70.6871948, -111.0442123, 111.0442200
3: -33.4847565, 71.3168869, -33.4847565, 71.3168869, -104.8016434, 104.8016434
4: -48.2948952, 61.0561142, -48.2948952, 61.0561142, -109.3509903, 109.3509903

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_A2_A1

### Relational analysis result of NS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1719784, upper bound: 65.1738040
time: 0.61 seconds

## Relational analysis of NS_B2_A2_A2

### Relational analysis result of NS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1717722, upper bound: 65.1769590
time: 0.71 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.35 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 4, lower bound: -65.1719784, upper bound: 65.1722003
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 4, lower bound: -65.1717722, upper bound: 65.1717722
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 4, lower bound: -65.1695340, upper bound: 65.1756423
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 4, lower bound: -65.1695340, upper bound: 65.1766414
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 4, lower bound: -65.1719533, upper bound: 65.1718108
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 4, lower bound: -65.1724066, upper bound: 65.1727647
NS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 4, lower bound: -65.1719784, upper bound: 65.1738040
NS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.35
Output dim: 4, lower bound: -65.1717722, upper bound: 65.1769590

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -263.2796936, 337.7713928, -266.5393982, 341.3458252, -604.6254883, 604.3107300
1: -31.5349445, 29.0909214, -31.8348675, 29.4663811, -61.0013199, 60.9257736
2: -18.1074886, 32.1444550, -18.3058014, 32.4970741, -50.6045609, 50.4502563
3: -14.8415689, 32.4973145, -15.0378723, 32.8243256, -47.6658897, 47.5351868
4: -21.8209801, 27.6549778, -22.0716648, 27.9611950, -49.7821732, 49.7266426

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_A1

### Relational analysis result of NS_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1712490, upper bound: 65.1693239
time: 0.60 seconds

## Relational analysis of NS_B1_A1_A1_A2

### Relational analysis result of NS_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1720147, upper bound: 65.1721576
time: 0.63 seconds

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -262.7756653, 336.9244995, -266.5393982, 341.3458252, -604.1214600, 603.4637451
1: -31.4369431, 29.0561047, -31.8348675, 29.4663811, -60.9033165, 60.8909645
2: -18.0524635, 32.0731583, -18.3058014, 32.4970741, -50.5495300, 50.3789597
3: -14.8204651, 32.3979568, -15.0378723, 32.8243256, -47.6447868, 47.4358292
4: -21.7730389, 27.5877228, -22.0716648, 27.9611950, -49.7342262, 49.6593857

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B1_A1_A2_A1

### Relational analysis result of NS_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1238582, upper bound: 65.1398197
time: 0.65 seconds

## Relational analysis of NS_B1_A1_A2_A2

### Relational analysis result of NS_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1126392, upper bound: 65.1126392
time: 0.51 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -569.1228027, 720.5650024, -94.2483063, 113.2373810, -682.3601074, 814.8132324
1: -66.5356598, 63.0975189, -10.7956877, 9.8754654, -76.4111252, 73.8931885
2: -39.2302284, 68.3910370, -6.1857562, 10.7456675, -49.9758949, 74.5767899
3: -32.6565666, 69.0560684, -5.0331597, 11.2536554, -43.9102135, 74.0892258
4: -46.9061546, 59.1472321, -7.3785291, 9.3780966, -56.2842522, 66.5257568

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B1_B1

### Relational analysis result of NS_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1696539, upper bound: 65.1726629
time: 0.67 seconds

## Relational analysis of NS_B1_A2_B1_B2

### Relational analysis result of NS_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1692957, upper bound: 65.1734434
time: 0.66 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -581.7056885, 742.9239502, -265.6603394, 340.1935730, -921.8991699, 1008.5842896
1: -68.6827316, 64.7128677, -31.7329540, 29.3631477, -98.0458832, 96.4458084
2: -40.2370834, 70.4836807, -18.2432175, 32.3864441, -72.6235275, 88.7268982
3: -33.3851509, 71.1173935, -14.9854231, 32.7193451, -66.1044922, 86.1027985
4: -48.1552124, 60.8823013, -21.9946251, 27.8653603, -76.0205688, 82.8769226

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A2_B2_B1

### Relational analysis result of NS_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1718079, upper bound: 65.1737195
time: 0.62 seconds

## Relational analysis of NS_B1_A2_B2_B2

### Relational analysis result of NS_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1692957, upper bound: 65.1742090
time: 0.70 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -256.8340759, 330.5543213, -583.4630737, 744.9866943, -1001.8208008, 914.0173340
1: -30.8947983, 28.4257393, -68.8743057, 64.9022522, -95.7970505, 97.3000336
2: -17.6641197, 31.4720097, -40.3570328, 70.6871948, -88.3513184, 71.8290253
3: -14.4615583, 31.7990417, -33.4847565, 71.3168869, -85.7784424, 65.2837830
4: -21.3087902, 27.0503941, -48.2948952, 61.0561142, -82.3648834, 75.3452759

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1237561, upper bound: 65.1022933
time: 0.55 seconds

## Relational analysis of NS_B2_A1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1729969, upper bound: 65.1718108
time: 0.56 seconds

## Relational analysis of NS_B2_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1737195, upper bound: 65.1718065
time: 0.59 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -268.0832520, 351.1907959, -583.4630737, 744.9866943, -1013.0699463, 934.6538086
1: -33.0530777, 29.8238754, -68.8743057, 64.9022522, -97.9553223, 98.6981812
2: -18.5632839, 33.5292130, -40.3570328, 70.6871948, -89.2504730, 73.8862381
3: -15.0968866, 33.7410965, -33.4847565, 71.3168869, -86.4137726, 67.2258453
4: -22.4540520, 28.7375927, -48.2948952, 61.0561142, -83.5101242, 77.0324860

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B2_A1_A2_A1

### Relational analysis result of NS_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1742090, upper bound: 65.1729952
time: 0.61 seconds

## Relational analysis of NS_B2_A1_A2_A2

### Relational analysis result of NS_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1737195, upper bound: 65.1723322
time: 0.62 seconds

## BFS NS instance: NS_B2_A2_A1

### Backsubstitution after applying NS history:
0: -580.9688110, 744.5696411, -583.4630737, 744.9866943, -1325.9554443, 1328.0325928
1: -68.8410721, 64.7268372, -68.8743057, 64.9022522, -133.7433167, 133.6011353
2: -40.2585335, 70.5923157, -40.3570328, 70.6871948, -110.9457169, 110.9493484
3: -33.3403168, 71.2683945, -33.4847565, 71.3168869, -104.6572037, 104.7531433
4: -48.1804390, 60.9837036, -48.2948952, 61.0561142, -109.2365417, 109.2785797

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A1_A1

### Relational analysis result of NS_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1599436, upper bound: 65.1572559
time: 0.59 seconds

## Relational analysis of NS_B2_A2_A1_A2

### Relational analysis result of NS_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1610740, upper bound: 65.1637488
time: 0.75 seconds

## BFS NS instance: NS_B2_A2_A2

### Backsubstitution after applying NS history:
0: -580.4371948, 741.4429321, -583.4630737, 744.9866943, -1325.4238281, 1324.9060059
1: -68.5514374, 64.5757523, -68.8743057, 64.9022522, -133.4536896, 133.4500580
2: -40.1522827, 70.3439865, -40.3570328, 70.6871948, -110.8394699, 110.7009964
3: -33.3106575, 70.9771729, -33.4847565, 71.3168869, -104.6275482, 104.4619217
4: -48.0544090, 60.7574081, -48.2948952, 61.0561142, -109.1105194, 109.0522995

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A2_A1

### Relational analysis result of NS_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1594886, upper bound: 65.1568945
time: 0.65 seconds

## Relational analysis of NS_B2_A2_A2_A2

### Relational analysis result of NS_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1654079, upper bound: 65.1676774
time: 0.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.40 seconds
NS_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1712490, upper bound: 65.1693239
NS_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1720147, upper bound: 65.1721576
NS_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1238582, upper bound: 65.1398197
NS_B1_A1_A2_A2, status: Status.VERIFIED, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1126392, upper bound: 65.1126392
NS_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1696539, upper bound: 65.1726629
NS_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1692957, upper bound: 65.1734434
NS_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1718079, upper bound: 65.1737195
NS_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1692957, upper bound: 65.1742090
NS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1729969, upper bound: 65.1718108
NS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1737195, upper bound: 65.1718065
NS_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1742090, upper bound: 65.1729952
NS_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1737195, upper bound: 65.1723322
NS_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1599436, upper bound: 65.1572559
NS_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1610740, upper bound: 65.1637488
NS_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1594886, upper bound: 65.1568945
NS_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 4.40
Output dim: 4, lower bound: -65.1654079, upper bound: 65.1676774

## BFS NS instance: NS_B1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -96.2884369, 115.5135956, -249.8812714, 315.2765198, -411.5649414, 365.3948669
1: -11.0079184, 10.0834484, -29.3269882, 27.4781342, -38.4860458, 39.4104385
2: -6.3257790, 10.9778528, -17.0512066, 29.9773540, -36.3031311, 28.0290604
3: -5.1302724, 11.5335751, -14.0760155, 30.3738117, -35.5040855, 25.6095867
4: -7.5142975, 9.5992689, -20.5297527, 25.8643742, -33.3786697, 30.1290188

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_A1_A1

### Relational analysis result of NS_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1670028, upper bound: 65.1579123
time: 0.56 seconds

## Relational analysis of NS_B1_A1_A1_A1_A2

### Relational analysis result of NS_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1670028, upper bound: 65.1693239
time: 0.61 seconds

## BFS NS instance: NS_B1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -262.3829956, 336.6220093, -266.2365112, 340.9533691, -603.3363647, 602.8585205
1: -31.4326057, 28.9870243, -31.8001957, 29.4300060, -60.8626099, 60.7872086
2: -18.0440483, 32.0328064, -18.2838287, 32.4590836, -50.5031319, 50.3166275
3: -14.7885447, 32.3920135, -15.0198545, 32.7886124, -47.5771523, 47.4118690
4: -21.7434425, 27.5584126, -22.0449448, 27.9282665, -49.6717072, 49.6033554

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1695702, upper bound: 65.1711926
time: 0.61 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1695702, upper bound: 65.1721576
time: 0.60 seconds

## BFS NS instance: NS_B1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -251.9737091, 323.3518677, -266.5393982, 341.3458252, -593.3195190, 589.8912354
1: -30.2202110, 27.8650455, -31.8348675, 29.4663811, -59.6865921, 59.6999092
2: -17.2938766, 30.8006954, -18.3058014, 32.4970741, -49.7909508, 49.1064987
3: -14.1899519, 31.1154060, -15.0378723, 32.8243256, -47.0142708, 46.1532784
4: -20.8754559, 26.4752045, -22.0716648, 27.9611950, -48.8366432, 48.5468674

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A2_A1_B1

### Relational analysis result of NS_B1_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1117550, upper bound: 65.1104567
time: 0.70 seconds

## Relational analysis of NS_B1_A1_A2_A1_B2

### Relational analysis result of NS_B1_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1096228, upper bound: 65.1097869
time: 0.60 seconds

## BFS NS instance: NS_B1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -569.1228027, 720.5650024, -90.1080246, 108.1620255, -677.2847900, 810.6730347
1: -66.5356598, 63.0975189, -10.3416023, 9.4061069, -75.9417648, 73.4391098
2: -39.2302284, 68.3910370, -5.8995852, 10.2631893, -49.4934120, 74.2906189
3: -32.6565666, 69.0560684, -4.7839789, 10.7745600, -43.4311180, 73.8400497
4: -46.9061546, 59.1472321, -7.0255108, 8.9616709, -55.8678246, 66.1727448

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_A2_B1_B1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1696539, upper bound: 65.1719403
time: 0.58 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1696495, upper bound: 65.1726629
time: 0.58 seconds

## BFS NS instance: NS_B1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -569.1228027, 720.5650024, -105.3710403, 135.6639709, -704.7866821, 825.9360352
1: -66.5356598, 63.0975189, -13.0996990, 11.3691397, -77.9048004, 76.1971970
2: -39.2302284, 68.3910370, -7.1339488, 12.9377890, -52.1680183, 75.5249863
3: -32.6565666, 69.0560684, -5.6908026, 13.3522415, -46.0088043, 74.7468643
4: -46.9061546, 59.1472321, -8.5788450, 11.1250610, -58.0312157, 67.7260742

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_A2_B1_B2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1696539, upper bound: 65.1722790
time: 0.59 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1693277, upper bound: 65.1730016
time: 0.60 seconds

## BFS NS instance: NS_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -581.7056885, 742.9239502, -256.0330811, 329.5161133, -911.2218018, 998.9570312
1: -68.6827316, 64.7128677, -30.8023796, 28.3328476, -97.0155716, 95.5152206
2: -40.2370834, 70.4836807, -17.6074467, 31.3720188, -71.6090927, 88.0911255
3: -33.3851509, 71.1173935, -14.4139414, 31.7036228, -65.0887680, 85.5313187
4: -48.1552124, 60.8823013, -21.2392769, 26.9638958, -75.1191025, 82.1215668

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1022933, upper bound: 65.1237561
time: 0.61 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_A2_B2_B1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1696538, upper bound: 65.1729969
time: 0.65 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1696495, upper bound: 65.1737195
time: 0.66 seconds

## BFS NS instance: NS_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -581.7056885, 742.9239502, -267.4748840, 350.4080200, -932.1137085, 1010.3988037
1: -68.6827316, 64.7128677, -32.9838295, 29.7537632, -98.4364929, 97.6966858
2: -40.2370834, 70.4836807, -18.5207596, 33.4551468, -73.6922302, 89.0044327
3: -33.3851509, 71.1173935, -15.0604372, 33.6691856, -67.0543365, 86.1778336
4: -48.1552124, 60.8823013, -22.4012814, 28.6729298, -76.8281403, 83.2835617

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_B1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_B1_A2_B2_B2_A1

### Relational analysis result of NS_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1718079, upper bound: 65.1732440
time: 0.60 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2

### Relational analysis result of NS_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1723322, upper bound: 65.1739666
time: 0.63 seconds

## BFS NS instance: NS_B2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -256.8340759, 330.5543213, -580.9688110, 744.5696411, -1001.4036255, 911.5230103
1: -30.8947983, 28.4257393, -68.8410721, 64.7268372, -95.6216354, 97.2668076
2: -17.6641197, 31.4720097, -40.2585335, 70.5923157, -88.2564392, 71.7305298
3: -14.4615583, 31.7990417, -33.3403168, 71.2683945, -85.7299347, 65.1393433
4: -21.3087902, 27.0503941, -48.1804390, 60.9837036, -82.2924728, 75.2308273

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A1_B1_B1

### Relational analysis result of NS_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1569021, upper bound: 65.1591894
time: 0.59 seconds

## Relational analysis of NS_B2_A1_A1_B1_B2

### Relational analysis result of NS_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1607925, upper bound: 65.1609905
time: 0.68 seconds

## BFS NS instance: NS_B2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -256.8340759, 330.5543213, -580.4371948, 741.4429321, -998.2769775, 910.9915161
1: -30.8947983, 28.4257393, -68.5514374, 64.5757523, -95.4705505, 96.9771729
2: -17.6641197, 31.4720097, -40.1522827, 70.3439865, -88.0081024, 71.6242905
3: -14.4615583, 31.7990417, -33.3106575, 70.9771729, -85.4387207, 65.1096878
4: -21.3087902, 27.0503941, -48.0544090, 60.7574081, -82.0661926, 75.1048050

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A1_B2_B1

### Relational analysis result of NS_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1564824, upper bound: 65.1590826
time: 0.63 seconds

## Relational analysis of NS_B2_A1_A1_B2_B2

### Relational analysis result of NS_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1640919, upper bound: 65.1609810
time: 0.59 seconds

## BFS NS instance: NS_B2_A1_A2_A1

### Backsubstitution after applying NS history:
0: -267.6126709, 355.6670837, -583.4630737, 744.9866943, -1012.5993042, 939.1301270
1: -33.6261826, 29.8906956, -68.8743057, 64.9022522, -98.5284119, 98.7649994
2: -18.6352978, 33.9454422, -40.3570328, 70.6871948, -89.3224869, 74.3024597
3: -15.0496101, 34.2653122, -33.4847565, 71.3168869, -86.3664932, 67.7500610
4: -22.5923862, 29.0224304, -48.2948952, 61.0561142, -83.6484909, 77.3173218

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A2_A1_A1

### Relational analysis result of NS_B2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1736862, upper bound: 65.1722344
time: 0.67 seconds

## Relational analysis of NS_B2_A1_A2_A1_A2

### Relational analysis result of NS_B2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1742090, upper bound: 65.1729952
time: 0.71 seconds

## BFS NS instance: NS_B2_A1_A2_A2

### Backsubstitution after applying NS history:
0: -264.7385559, 347.4119873, -583.4630737, 744.9866943, -1009.7252197, 930.8749390
1: -32.7178040, 29.4604149, -68.8743057, 64.9022522, -97.6200562, 98.3347092
2: -18.3396873, 33.1661072, -40.3570328, 70.6871948, -89.0268784, 73.5231400
3: -14.9018459, 33.3741646, -33.4847565, 71.3168869, -86.2187347, 66.8589096
4: -22.1940899, 28.4165497, -48.2948952, 61.0561142, -83.2501831, 76.7114334

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A2_A2_A1

### Relational analysis result of NS_B2_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1677441, upper bound: 65.1624231
time: 0.59 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2

### Relational analysis result of NS_B2_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1737195, upper bound: 65.1723322
time: 0.62 seconds

## BFS NS instance: NS_B2_A2_A1_A1

### Backsubstitution after applying NS history:
0: -555.0100098, 709.7383423, -583.4630737, 744.9866943, -1299.9967041, 1293.2010498
1: -65.6161804, 61.8273239, -68.8743057, 64.9022522, -130.5184326, 130.7016296
2: -38.3745918, 67.3106537, -40.3570328, 70.6871948, -109.0617752, 107.6676636
3: -31.8558216, 67.9307098, -33.4847565, 71.3168869, -103.1727066, 101.4154663
4: -45.9714317, 58.1589890, -48.2948952, 61.0561142, -107.0275269, 106.4538727

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B2_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A1_A1_B1

### Relational analysis result of NS_B2_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1557612
time: 0.58 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2

### Relational analysis result of NS_B2_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1572559
time: 0.64 seconds

## BFS NS instance: NS_B2_A2_A1_A2

### Backsubstitution after applying NS history:
0: -572.9612427, 735.6890869, -583.4630737, 744.9866943, -1317.9478760, 1319.1520996
1: -68.0334930, 63.9209595, -68.8743057, 64.9022522, -132.9357147, 132.7952576
2: -39.7246094, 69.7113266, -40.3570328, 70.6871948, -110.4118042, 110.0683594
3: -32.8924675, 70.4113235, -33.4847565, 71.3168869, -104.2093506, 103.8960800
4: -47.5731239, 60.2263641, -48.2948952, 61.0561142, -108.6292343, 108.5212479

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B2_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A1_A2_B1

### Relational analysis result of NS_B2_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1573146, upper bound: 65.1618505
time: 0.61 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2

### Relational analysis result of NS_B2_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1573146, upper bound: 65.1637488
time: 0.62 seconds

## BFS NS instance: NS_B2_A2_A2_A1

### Backsubstitution after applying NS history:
0: -557.4353638, 709.4327393, -583.4630737, 744.9866943, -1302.4219971, 1292.8957520
1: -65.5964661, 61.9446335, -68.8743057, 64.9022522, -130.4987030, 130.8189240
2: -38.4446907, 67.3669357, -40.3570328, 70.6871948, -109.1318817, 107.7239609
3: -31.9870701, 67.9271240, -33.4847565, 71.3168869, -103.3039551, 101.4118805
4: -46.0468712, 58.1938019, -48.2948952, 61.0561142, -107.1029816, 106.4886856

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B2_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B2_A2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B2_A2_A2_A1_B1

### Relational analysis result of NS_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1553951, upper bound: 65.1553951
time: 0.57 seconds

## Relational analysis of NS_B2_A2_A2_A1_B2

### Relational analysis result of NS_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1554067, upper bound: 65.1568945
time: 0.59 seconds

## BFS NS instance: NS_B2_A2_A2_A2

### Backsubstitution after applying NS history:
0: -572.2338867, 732.4370117, -583.4630737, 744.9866943, -1317.2205811, 1315.8999023
1: -67.7335968, 63.7569275, -68.8743057, 64.9022522, -132.6358337, 132.6312103
2: -39.6098976, 69.4449234, -40.3570328, 70.6871948, -110.2970734, 109.8019409
3: -32.8490639, 70.1085892, -33.4847565, 71.3168869, -104.1659546, 103.5933380
4: -47.4335594, 59.9865417, -48.2948952, 61.0561142, -108.4896622, 108.2814331

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_A2_A2_B1

### Relational analysis result of NS_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1393268, upper bound: 65.1465409
time: 0.72 seconds

## Relational analysis of NS_B2_A2_A2_A2_B2

### Relational analysis result of NS_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1250734, upper bound: 65.1252391
time: 0.87 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.25 seconds
NS_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1670028, upper bound: 65.1579123
NS_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1670028, upper bound: 65.1693239
NS_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1695702, upper bound: 65.1711926
NS_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1695702, upper bound: 65.1721576
NS_B1_A1_A2_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1117550, upper bound: 65.1104567
NS_B1_A1_A2_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1096228, upper bound: 65.1097869
NS_B1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1696539, upper bound: 65.1719403
NS_B1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1696495, upper bound: 65.1726629
NS_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1696539, upper bound: 65.1722790
NS_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1693277, upper bound: 65.1730016
NS_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1696538, upper bound: 65.1729969
NS_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1696495, upper bound: 65.1737195
NS_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1718079, upper bound: 65.1732440
NS_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1723322, upper bound: 65.1739666
NS_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1569021, upper bound: 65.1591894
NS_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1607925, upper bound: 65.1609905
NS_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1564824, upper bound: 65.1590826
NS_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1640919, upper bound: 65.1609810
NS_B2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1736862, upper bound: 65.1722344
NS_B2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1742090, upper bound: 65.1729952
NS_B2_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1677441, upper bound: 65.1624231
NS_B2_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1737195, upper bound: 65.1723322
NS_B2_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1557612
NS_B2_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1572559
NS_B2_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1573146, upper bound: 65.1618505
NS_B2_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1573146, upper bound: 65.1637488
NS_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1553951, upper bound: 65.1553951
NS_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1554067, upper bound: 65.1568945
NS_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1393268, upper bound: 65.1465409
NS_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.25
Output dim: 4, lower bound: -65.1250734, upper bound: 65.1252391

## BFS NS instance: NS_B1_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -83.5658188, 100.7713089, -244.4489746, 307.2020569, -390.7678833, 345.2202759
1: -9.5390444, 8.8237305, -28.5825882, 26.8402824, -36.3793259, 37.4063187
2: -5.5096540, 9.4843712, -16.6515522, 29.2217064, -34.7313614, 26.1359234
3: -4.5210495, 9.8829308, -13.7584305, 29.6182289, -34.1392708, 23.6413593
4: -6.6271715, 8.2658491, -20.0434551, 25.2278996, -31.8550720, 28.3093033

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A1_A1_A1_A1

### Relational analysis result of NS_B1_A1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1670028, upper bound: 65.1579123
time: 0.55 seconds

## Relational analysis of NS_B1_A1_A1_A1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_A1_A1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1670028, upper bound: 65.1579123
time: 0.58 seconds

## Relational analysis of NS_B1_A1_A1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1670028, upper bound: 65.1579123
time: 0.59 seconds

## BFS NS instance: NS_B1_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -94.9856949, 113.8814621, -249.8812714, 315.2765198, -410.2622070, 363.7627258
1: -10.8554583, 9.9324665, -29.3269882, 27.4781342, -38.3335838, 39.2594528
2: -6.2316704, 10.8189964, -17.0512066, 29.9773540, -36.2090225, 27.8701973
3: -5.0534482, 11.3767242, -14.0760155, 30.3738117, -35.4272575, 25.4527302
4: -7.4000573, 9.4728413, -20.5297527, 25.8643742, -33.2644310, 30.0025940

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A1_A1_A2_A1

### Relational analysis result of NS_B1_A1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1699743, upper bound: 65.1649763
time: 0.61 seconds

## Relational analysis of NS_B1_A1_A1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_A1_A1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1712383, upper bound: 65.1693239
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1712383, upper bound: 65.1693239
time: 0.59 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -262.3829956, 336.6220093, -94.2483063, 113.2373810, -375.6203613, 430.8703003
1: -31.4326057, 28.9870243, -10.7956877, 9.8754654, -41.3080711, 39.7827110
2: -18.0440483, 32.0328064, -6.1857562, 10.7456675, -28.7897148, 38.2185555
3: -14.7885447, 32.3920135, -5.0331597, 11.2536554, -26.0421982, 37.4251747
4: -21.7434425, 27.5584126, -7.3785291, 9.3780966, -31.1215382, 34.9369354

Time for backsubstitution: 2.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1693319, upper bound: 65.1708538
time: 0.62 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1693319, upper bound: 65.1711926
time: 0.63 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -262.3829956, 336.6220093, -265.6603394, 340.1935730, -602.5764771, 602.2823486
1: -31.4326057, 28.9870243, -31.7329540, 29.3631477, -60.7957535, 60.7199783
2: -18.0440483, 32.0328064, -18.2432175, 32.3864441, -50.4304924, 50.2760239
3: -14.7885447, 32.3920135, -14.9854231, 32.7193451, -47.5078888, 47.3774376
4: -21.7434425, 27.5584126, -21.9946251, 27.8653603, -49.6088028, 49.5530357

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A1_A1_A2_B2_A1

### Relational analysis result of NS_B1_A1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1571678, upper bound: 65.1595791
time: 0.60 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B1_A1_A1_A2_B2_B1

### Relational analysis result of NS_B1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1695663, upper bound: 65.1721467
time: 0.63 seconds

## Relational analysis of NS_B1_A1_A1_A2_B2_B2

### Relational analysis result of NS_B1_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1695663, upper bound: 65.1721467
time: 0.68 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -566.0540771, 719.7155762, -90.1080246, 108.1620255, -674.2161255, 809.8236084
1: -66.4680252, 62.8794937, -10.3416023, 9.4061069, -75.8741302, 73.2210999
2: -39.1043129, 68.2485123, -5.8995852, 10.2631893, -49.3674927, 74.1480942
3: -32.4809647, 68.9393463, -4.7839789, 10.7745600, -43.2555199, 73.7233276
4: -46.7585754, 58.9938622, -7.0255108, 8.9616709, -55.7202454, 66.0193710

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B1_A1_A1

### Relational analysis result of NS_B1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1559316, upper bound: 65.1551791
time: 0.60 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_A2

### Relational analysis result of NS_B1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1577460, upper bound: 65.1614379
time: 0.65 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -566.0786743, 717.0025635, -90.1080246, 108.1620255, -674.2407227, 807.1105957
1: -66.2111969, 62.7697601, -10.3416023, 9.4061069, -75.6173019, 73.1113586
2: -39.0243111, 68.0459290, -5.8995852, 10.2631893, -49.2874985, 73.9455032
3: -32.4816475, 68.7099991, -4.7839789, 10.7745600, -43.2562065, 73.4939728
4: -46.6643677, 58.8430481, -7.0255108, 8.9616709, -55.6260338, 65.8685532

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B1_A2_A1

### Relational analysis result of NS_B1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1558238, upper bound: 65.1547950
time: 0.58 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_A2

### Relational analysis result of NS_B1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1573211, upper bound: 65.1624082
time: 0.58 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -566.0540771, 719.7155762, -105.3710403, 135.6639709, -701.7180176, 825.0866089
1: -66.4680252, 62.8794937, -13.0996990, 11.3691397, -77.8371582, 75.9791794
2: -39.1043129, 68.2485123, -7.1339488, 12.9377890, -52.0421028, 75.3824615
3: -32.4809647, 68.9393463, -5.6908026, 13.3522415, -45.8332062, 74.6301346
4: -46.7585754, 58.9938622, -8.5788450, 11.1250610, -57.8836365, 67.5727081

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B2_A1_A1

### Relational analysis result of NS_B1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1559531, upper bound: 65.1552871
time: 0.58 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_A2

### Relational analysis result of NS_B1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1577675, upper bound: 65.1615459
time: 0.71 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -566.0786743, 717.0025635, -105.3710403, 135.6639709, -701.7426758, 822.3735962
1: -66.2111969, 62.7697601, -13.0996990, 11.3691397, -77.5803375, 75.8694382
2: -39.0243111, 68.0459290, -7.1339488, 12.9377890, -51.9621010, 75.1798706
3: -32.4816475, 68.7099991, -5.6908026, 13.3522415, -45.8338890, 74.4007797
4: -46.6643677, 58.8430481, -8.5788450, 11.1250610, -57.7894287, 67.4218826

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B2_A2_A1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1558453, upper bound: 65.1549030
time: 0.57 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1577579, upper bound: 65.1625162
time: 0.67 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -579.1906738, 742.4566650, -256.0330811, 329.5161133, -908.7067871, 998.4897461
1: -68.6442719, 64.5340805, -30.8023796, 28.3328476, -96.9771194, 95.3364334
2: -40.1361885, 70.3845291, -17.6074467, 31.3720188, -71.5082016, 87.9919739
3: -33.2396164, 71.0645676, -14.4139414, 31.7036228, -64.9432373, 85.4785080
4: -48.0380592, 60.8068962, -21.2392769, 26.9638958, -75.0019379, 82.0461578

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_B1_A1_A1

### Relational analysis result of NS_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1559316, upper bound: 65.1568628
time: 0.65 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_A2

### Relational analysis result of NS_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1577460, upper bound: 65.1614379
time: 0.61 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -578.6952515, 739.3917236, -256.0330811, 329.5161133, -908.2113647, 995.4248047
1: -68.3605576, 64.3878098, -30.8023796, 28.3328476, -96.6934052, 95.1901779
2: -40.0332336, 70.1416168, -17.6074467, 31.3720188, -71.4052505, 87.7490616
3: -33.2120857, 70.7788239, -14.4139414, 31.7036228, -64.9156952, 85.1927490
4: -47.9157257, 60.5847855, -21.2392769, 26.9638958, -74.8796234, 81.8240585

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_B1_A2_A1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1558238, upper bound: 65.1564787
time: 0.65 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1573211, upper bound: 65.1640919
time: 0.73 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -579.1906738, 742.4566650, -267.4748840, 350.4080200, -929.5986328, 1009.9315186
1: -68.6442719, 64.5340805, -32.9838295, 29.7537632, -98.3980331, 97.5178986
2: -40.1361885, 70.3845291, -18.5207596, 33.4551468, -73.5913391, 88.9052811
3: -33.2396164, 71.0645676, -15.0604372, 33.6691856, -66.9087982, 86.1250076
4: -48.0380592, 60.8068962, -22.4012814, 28.6729298, -76.7109909, 83.2081528

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_B2_A1_B1

### Relational analysis result of NS_B1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1585812, upper bound: 65.1621888
time: 0.67 seconds

## Relational analysis of NS_B1_A2_B2_B2_A1_B2

### Relational analysis result of NS_B1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1577460, upper bound: 65.1633335
time: 0.76 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -578.6952515, 739.3917236, -267.4748840, 350.4080200, -929.1031494, 1006.8665771
1: -68.3605576, 64.3878098, -32.9838295, 29.7537632, -98.1143188, 97.3716431
2: -40.0332336, 70.1416168, -18.5207596, 33.4551468, -73.4883804, 88.6623688
3: -33.2120857, 70.7788239, -15.0604372, 33.6691856, -66.8812637, 85.8392639
4: -47.9157257, 60.5847855, -22.4012814, 28.6729298, -76.5886536, 82.9860535

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 26

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_B2_A2_B1

### Relational analysis result of NS_B1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1585716, upper bound: 65.1631591
time: 0.67 seconds

## Relational analysis of NS_B1_A2_B2_B2_A2_B2

### Relational analysis result of NS_B1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1610879, upper bound: 65.1643038
time: 0.71 seconds

## BFS NS instance: NS_B2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -256.8340759, 330.5543213, -555.0100098, 709.7383423, -966.5723877, 885.5642090
1: -30.8947983, 28.4257393, -65.6161804, 61.8273239, -92.7221222, 94.0419159
2: -17.6641197, 31.4720097, -38.3745918, 67.3106537, -84.9747772, 69.8465881
3: -14.4615583, 31.7990417, -31.8558216, 67.9307098, -82.3922653, 63.6548615
4: -21.3087902, 27.0503941, -45.9714317, 58.1589890, -79.4677658, 73.0218201

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A1_B1_B1_A1

### Relational analysis result of NS_B2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1556956, upper bound: 65.1570910
time: 0.57 seconds

## Relational analysis of NS_B2_A1_A1_B1_B1_A2

### Relational analysis result of NS_B2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1556956, upper bound: 65.1591894
time: 0.56 seconds

## BFS NS instance: NS_B2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -256.8340759, 330.5543213, -572.9612427, 735.6890869, -992.5230713, 903.5155640
1: -30.8947983, 28.4257393, -68.0334930, 63.9209595, -94.8157578, 96.4592285
2: -17.6641197, 31.4720097, -39.7246094, 69.7113266, -87.3754425, 71.1966095
3: -14.4615583, 31.7990417, -32.8924675, 70.4113235, -84.8728714, 64.6915054
4: -21.3087902, 27.0503941, -47.5731239, 60.2263641, -81.5351410, 74.6235123

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_B2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A1_B1_B2_A1

### Relational analysis result of NS_B2_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1619151, upper bound: 65.1588921
time: 0.62 seconds

## Relational analysis of NS_B2_A1_A1_B1_B2_A2

### Relational analysis result of NS_B2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1619151, upper bound: 65.1609905
time: 0.64 seconds

## BFS NS instance: NS_B2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -256.8340759, 330.5543213, -557.4353638, 709.4327393, -966.2667847, 887.9896240
1: -30.8947983, 28.4257393, -65.5964661, 61.9446335, -92.8394318, 94.0222015
2: -17.6641197, 31.4720097, -38.4446907, 67.3669357, -85.0310516, 69.9167023
3: -14.4615583, 31.7990417, -31.9870701, 67.9271240, -82.3886795, 63.7861099
4: -21.3087902, 27.0503941, -46.0468712, 58.1938019, -79.5025787, 73.0972672

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_A1_B2_B1_B1

### Relational analysis result of NS_B2_A1_A1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1000553, upper bound: 65.1144115
time: 0.52 seconds

## Relational analysis of NS_B2_A1_A1_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A1_B2_B1_A1

### Relational analysis result of NS_B2_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1552759, upper bound: 65.1569842
time: 0.58 seconds

## Relational analysis of NS_B2_A1_A1_B2_B1_A2

### Relational analysis result of NS_B2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1552759, upper bound: 65.1590826
time: 0.59 seconds

## BFS NS instance: NS_B2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -256.8340759, 330.5543213, -572.2338867, 732.4370117, -989.2710571, 902.7882080
1: -30.8947983, 28.4257393, -67.7335968, 63.7569275, -94.6517181, 96.1593323
2: -17.6641197, 31.4720097, -39.6098976, 69.4449234, -87.1090393, 71.0818863
3: -14.4615583, 31.7990417, -32.8490639, 70.1085892, -84.5701370, 64.6481018
4: -21.3087902, 27.0503941, -47.4335594, 59.9865417, -81.2953339, 74.4839401

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B2_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_B2_A1_A1_B2_B2_B1

### Relational analysis result of NS_B2_A1_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1047723, upper bound: 65.1164314
time: 0.62 seconds

## Relational analysis of NS_B2_A1_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A1_B2_B2_A1

### Relational analysis result of NS_B2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1594231, upper bound: 65.1588825
time: 0.65 seconds

## Relational analysis of NS_B2_A1_A1_B2_B2_A2

### Relational analysis result of NS_B2_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1628854, upper bound: 65.1609810
time: 0.57 seconds

## BFS NS instance: NS_B2_A1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -249.7648621, 328.5451965, -583.3702393, 744.8399048, -994.6047363, 911.9154053
1: -31.0120773, 27.7626171, -68.8603516, 64.8909912, -95.9030533, 96.6229706
2: -17.3082466, 31.3351669, -40.3499985, 70.6733246, -87.9815598, 71.6851654
3: -14.0226698, 31.7465363, -33.4793968, 71.3032227, -85.3258896, 65.2259216
4: -20.9653969, 26.8219604, -48.2863388, 61.0443802, -82.0097656, 75.1082993

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A2_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A1_A2_A1_A1_B1

### Relational analysis result of NS_B2_A1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1729636, upper bound: 65.1722344
time: 0.65 seconds

## Relational analysis of NS_B2_A1_A2_A1_A1_B2

### Relational analysis result of NS_B2_A1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1729636, upper bound: 65.1722344
time: 0.63 seconds

## BFS NS instance: NS_B2_A1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -262.8413086, 350.5447998, -583.4630737, 744.9866943, -1007.8279419, 934.0078735
1: -33.1728935, 29.3903141, -68.8743057, 64.9022522, -98.0751343, 98.2646179
2: -18.3230171, 33.4525719, -40.3570328, 70.6871948, -89.0102081, 73.8095932
3: -14.7728243, 33.7671814, -33.4847565, 71.3168869, -86.0897141, 67.2519302
4: -22.2261600, 28.5834999, -48.2948952, 61.0561142, -83.2822723, 76.8783951

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A1_A2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1734864, upper bound: 65.1729952
time: 0.66 seconds

## Relational analysis of NS_B2_A1_A2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1734864, upper bound: 65.1729952
time: 0.64 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -245.9996338, 320.3171997, -583.3702393, 744.8399048, -990.8393555, 903.6873779
1: -30.1047344, 27.2969322, -68.8603516, 64.8909912, -94.9957123, 96.1572876
2: -16.9830437, 30.5275307, -40.3499985, 70.6733246, -87.6563492, 70.8775177
3: -13.8342152, 30.8469505, -33.4793968, 71.3032227, -85.1374359, 64.3263474
4: -20.5249577, 26.1940804, -48.2863388, 61.0443802, -81.5693207, 74.4804230

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 26

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A2_A2_A1_A1

### Relational analysis result of NS_B2_A1_A2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1570593, upper bound: 65.1523568
time: 0.70 seconds

## Relational analysis of NS_B2_A1_A2_A2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B2_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B2_A1_A2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A1_A2_A2_A1_B1

### Relational analysis result of NS_B2_A1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1677441, upper bound: 65.1624231
time: 0.64 seconds

## Relational analysis of NS_B2_A1_A2_A2_A1_B2

### Relational analysis result of NS_B2_A1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1677441, upper bound: 65.1624231
time: 0.62 seconds

## BFS NS instance: NS_B2_A1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -259.4486084, 341.3056946, -583.4630737, 744.9866943, -1004.4353027, 924.7686157
1: -32.1765022, 28.8813801, -68.8743057, 64.9022522, -97.0787354, 97.7556686
2: -17.9792099, 32.5832520, -40.3570328, 70.6871948, -88.6664047, 72.9402618
3: -14.5917358, 32.7860336, -33.4847565, 71.3168869, -85.9086075, 66.2707901
4: -21.7755928, 27.9024982, -48.2948952, 61.0561142, -82.8316956, 76.1973877

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_B2_A1_A2_A2_A2_A1

### Relational analysis result of NS_B2_A1_A2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1635982, upper bound: 65.1601694
time: 0.65 seconds

## Relational analysis of NS_B2_A1_A2_A2_A2_A2

### Relational analysis result of NS_B2_A1_A2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1640919, upper bound: 65.1612568
time: 0.64 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -555.0100098, 709.7383423, -560.3817749, 712.8793335, -1267.8894043, 1270.1199951
1: -65.6161804, 61.8273239, -65.9113235, 62.2621956, -127.8783722, 127.7386322
2: -38.3745918, 67.3106537, -38.6438942, 67.7012787, -106.0758591, 105.9545441
3: -31.8558216, 67.9307098, -32.1561127, 68.2580185, -100.1138382, 100.0868225
4: -45.9714317, 58.1589890, -46.2805519, 58.4848251, -104.4562531, 104.4395447

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A2_A1_A1_B1_B1

### Relational analysis result of NS_B2_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1557612
time: 0.67 seconds

## Relational analysis of NS_B2_A2_A1_A1_B1_B2

### Relational analysis result of NS_B2_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1557612
time: 0.59 seconds

## BFS NS instance: NS_B2_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -555.0100098, 709.7383423, -575.2778931, 736.0101929, -1291.0202637, 1285.0158691
1: -65.6161804, 61.8273239, -68.0594177, 64.0859375, -129.7021179, 129.8867340
2: -38.3745918, 67.3106537, -39.8161278, 69.7905807, -108.1651611, 107.1267853
3: -31.8558216, 67.9307098, -33.0241737, 70.4509583, -102.3067780, 100.9548798
4: -45.9714317, 58.1589890, -47.6757431, 60.2870026, -106.2584229, 105.8347321

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A2_A1_A1_B2_B1

### Relational analysis result of NS_B2_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1572559
time: 0.58 seconds

## Relational analysis of NS_B2_A2_A1_A1_B2_B2

### Relational analysis result of NS_B2_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1572559
time: 0.61 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -572.9612427, 735.6890869, -560.3817749, 712.8793335, -1285.8405762, 1296.0708008
1: -68.0334930, 63.9209595, -65.9113235, 62.2621956, -130.2956696, 129.8322754
2: -39.7246094, 69.7113266, -38.6438942, 67.7012787, -107.4258881, 108.3552246
3: -32.8924675, 70.4113235, -32.1561127, 68.2580185, -101.1504822, 102.5674362
4: -47.5731239, 60.2263641, -46.2805519, 58.4848251, -106.0579529, 106.5069122

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_B2_A2_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A2_A1_A2_B1_B1

### Relational analysis result of NS_B2_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1618505
time: 0.64 seconds

## Relational analysis of NS_B2_A2_A1_A2_B1_B2

### Relational analysis result of NS_B2_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1573146, upper bound: 65.1618505
time: 0.71 seconds

## BFS NS instance: NS_B2_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -572.9612427, 735.6890869, -575.2778931, 736.0101929, -1308.9714355, 1310.9669189
1: -68.0334930, 63.9209595, -68.0594177, 64.0859375, -132.1194000, 131.9803619
2: -39.7246094, 69.7113266, -39.8161278, 69.7905807, -109.5151901, 109.5274506
3: -32.8924675, 70.4113235, -33.0241737, 70.4509583, -103.3434296, 103.4354935
4: -47.5731239, 60.2263641, -47.6757431, 60.2870026, -107.8601227, 107.9021072

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_B2_A2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A2_A1_A2_B2_B1

### Relational analysis result of NS_B2_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1618505
time: 0.75 seconds

## Relational analysis of NS_B2_A2_A1_A2_B2_B2

### Relational analysis result of NS_B2_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1573146, upper bound: 65.1637488
time: 0.63 seconds

## BFS NS instance: NS_B2_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -557.4353638, 709.4327393, -560.3817749, 712.8793335, -1270.3146973, 1269.8144531
1: -65.5964661, 61.9446335, -65.9113235, 62.2621956, -127.8586578, 127.8559570
2: -38.4446907, 67.3669357, -38.6438942, 67.7012787, -106.1459656, 106.0108337
3: -31.9870701, 67.9271240, -32.1561127, 68.2580185, -100.2450867, 100.0832367
4: -46.0468712, 58.1938019, -46.2805519, 58.4848251, -104.5316925, 104.4743500

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A2_A2_A1_B1_B1

### Relational analysis result of NS_B2_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1554067, upper bound: 65.1553951
time: 0.57 seconds

## Relational analysis of NS_B2_A2_A2_A1_B1_B2

### Relational analysis result of NS_B2_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1554067, upper bound: 65.1553951
time: 0.56 seconds

## BFS NS instance: NS_B2_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -557.4353638, 709.4327393, -575.2778931, 736.0101929, -1293.4454346, 1284.7106934
1: -65.5964661, 61.9446335, -68.0594177, 64.0859375, -129.6823883, 130.0040436
2: -38.4446907, 67.3669357, -39.8161278, 69.7905807, -108.2352753, 107.1830597
3: -31.9870701, 67.9271240, -33.0241737, 70.4509583, -102.4380264, 100.9512939
4: -46.0468712, 58.1938019, -47.6757431, 60.2870026, -106.3338776, 105.8695450

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_B2_A2_A2_A1_B2_B1

### Relational analysis result of NS_B2_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1554067, upper bound: 65.1568945
time: 0.65 seconds

## Relational analysis of NS_B2_A2_A2_A1_B2_B2

### Relational analysis result of NS_B2_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1554067, upper bound: 65.1568945
time: 0.63 seconds

## BFS NS instance: NS_B2_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -572.2338867, 732.4370117, -578.0662231, 735.6224976, -1307.8564453, 1310.5031738
1: -67.7335968, 63.7569275, -67.9983139, 64.2084961, -131.9420776, 131.7552032
2: -39.6098976, 69.4449234, -39.9019623, 69.8631439, -109.4730225, 109.3468857
3: -32.8490639, 70.1085892, -33.1582413, 70.4442368, -103.2933044, 103.2668304
4: -47.4335594, 59.9865417, -47.7618790, 60.3534012, -107.7869568, 107.7484207

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B2_A2_A2_A2_B1_B1

### Relational analysis result of NS_B2_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1207266, upper bound: 65.1287849
time: 0.73 seconds

## Relational analysis of NS_B2_A2_A2_A2_B1_B2

### Relational analysis result of NS_B2_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1378554, upper bound: 65.1465409
time: 0.86 seconds

## BFS NS instance: NS_B2_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -566.9519043, 727.0499268, -593.3674316, 761.7014771, -1328.6530762, 1320.4171143
1: -67.2241592, 63.2438812, -69.9898453, 66.8516388, -134.0757599, 133.2337189
2: -39.2761116, 68.8906708, -41.2600517, 71.8805008, -111.1566162, 110.1507034
3: -32.5593109, 69.5742111, -34.3288231, 72.6880112, -105.2473221, 103.9030304
4: -47.0504112, 59.5159988, -49.5323486, 62.2144051, -109.2648163, 109.0483475

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 26

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_B2_A2_A2_A2_B2_A1

### Relational analysis result of NS_B2_A2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1250734, upper bound: 65.1252391
time: 0.83 seconds

## Relational analysis of NS_B2_A2_A2_A2_B2_A2

### Relational analysis result of NS_B2_A2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1250734, upper bound: 65.1252391
time: 0.80 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.39 seconds
NS_B1_A1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1670028, upper bound: 65.1579123
NS_B1_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1670028, upper bound: 65.1579123
NS_B1_A1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1712383, upper bound: 65.1693239
NS_B1_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1712383, upper bound: 65.1693239
NS_B1_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1693319, upper bound: 65.1708538
NS_B1_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1693319, upper bound: 65.1711926
NS_B1_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1695663, upper bound: 65.1721467
NS_B1_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1695663, upper bound: 65.1721467
NS_B1_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1559316, upper bound: 65.1551791
NS_B1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1577460, upper bound: 65.1614379
NS_B1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1558238, upper bound: 65.1547950
NS_B1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1573211, upper bound: 65.1624082
NS_B1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1559531, upper bound: 65.1552871
NS_B1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1577675, upper bound: 65.1615459
NS_B1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1558453, upper bound: 65.1549030
NS_B1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1577579, upper bound: 65.1625162
NS_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1559316, upper bound: 65.1568628
NS_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1577460, upper bound: 65.1614379
NS_B1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1558238, upper bound: 65.1564787
NS_B1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1573211, upper bound: 65.1640919
NS_B1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1585812, upper bound: 65.1621888
NS_B1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1577460, upper bound: 65.1633335
NS_B1_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1585716, upper bound: 65.1631591
NS_B1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1610879, upper bound: 65.1643038
NS_B2_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1556956, upper bound: 65.1570910
NS_B2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1556956, upper bound: 65.1591894
NS_B2_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1619151, upper bound: 65.1588921
NS_B2_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1619151, upper bound: 65.1609905
NS_B2_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1552759, upper bound: 65.1569842
NS_B2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1552759, upper bound: 65.1590826
NS_B2_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1594231, upper bound: 65.1588825
NS_B2_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1628854, upper bound: 65.1609810
NS_B2_A1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1729636, upper bound: 65.1722344
NS_B2_A1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1729636, upper bound: 65.1722344
NS_B2_A1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1734864, upper bound: 65.1729952
NS_B2_A1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1734864, upper bound: 65.1729952
NS_B2_A1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1677441, upper bound: 65.1624231
NS_B2_A1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1677441, upper bound: 65.1624231
NS_B2_A1_A2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1635982, upper bound: 65.1601694
NS_B2_A1_A2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1640919, upper bound: 65.1612568
NS_B2_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1557612
NS_B2_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1557612
NS_B2_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1572559
NS_B2_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1572559
NS_B2_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1618505
NS_B2_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1573146, upper bound: 65.1618505
NS_B2_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1555135, upper bound: 65.1618505
NS_B2_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1573146, upper bound: 65.1637488
NS_B2_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1554067, upper bound: 65.1553951
NS_B2_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1554067, upper bound: 65.1553951
NS_B2_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1554067, upper bound: 65.1568945
NS_B2_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1554067, upper bound: 65.1568945
NS_B2_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1207266, upper bound: 65.1287849
NS_B2_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1378554, upper bound: 65.1465409
NS_B2_A2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1250734, upper bound: 65.1252391
NS_B2_A2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.39
Output dim: 4, lower bound: -65.1250734, upper bound: 65.1252391

## BFS NS instance: NS_B1_A1_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -83.5658188, 100.7713089, -240.0609894, 302.6248474, -386.1906738, 340.8323059
1: -9.5390444, 8.8237305, -28.1603699, 26.3649445, -35.9039879, 36.9841003
2: -5.5096540, 9.4843712, -16.3807678, 28.7503529, -34.2600060, 25.8651390
3: -4.5210495, 9.8829308, -13.5052137, 29.1510925, -33.6721344, 23.3881397
4: -6.6271715, 8.2658491, -19.7143135, 24.8065643, -31.4337349, 27.9801636

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_A1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_A1_A1_B1_B1

### Relational analysis result of NS_B1_A1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1648826, upper bound: 65.1575639
time: 0.60 seconds

## Relational analysis of NS_B1_A1_A1_A1_A1_B1_B2

### Relational analysis result of NS_B1_A1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1648826, upper bound: 65.1579123
time: 0.58 seconds

## BFS NS instance: NS_B1_A1_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -83.5658188, 100.7713089, -240.6828918, 302.7587891, -386.3246155, 341.4541626
1: -9.5390444, 8.8237305, -28.1708527, 26.4316921, -35.9707375, 36.9945831
2: -5.5096540, 9.4843712, -16.3965397, 28.7865620, -34.2962151, 25.8809090
3: -4.5210495, 9.8829308, -13.5410500, 29.1802692, -33.7013168, 23.4239769
4: -6.6271715, 8.2658491, -19.7440071, 24.8484478, -31.4756184, 28.0098572

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_A1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_A1_A1_B2_B1

### Relational analysis result of NS_B1_A1_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1648826, upper bound: 65.1575639
time: 0.60 seconds

## Relational analysis of NS_B1_A1_A1_A1_A1_B2_B2

### Relational analysis result of NS_B1_A1_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1648826, upper bound: 65.1579123
time: 0.60 seconds

## BFS NS instance: NS_B1_A1_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -94.9856949, 113.8814621, -246.3505707, 311.5357361, -406.5214233, 360.2320251
1: -10.8554583, 9.9324665, -28.9983444, 27.0819206, -37.9373779, 38.9308090
2: -6.2316704, 10.8189964, -16.8361263, 29.6063061, -35.8379745, 27.6551189
3: -5.0534482, 11.3767242, -13.8682404, 30.0185604, -35.0720024, 25.2449608
4: -7.4000573, 9.4728413, -20.2617512, 25.5371189, -32.9371758, 29.7345924

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_A1_A1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_A1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_A1_A2_B1_B1

### Relational analysis result of NS_B1_A1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1687900, upper bound: 65.1683589
time: 0.63 seconds

## Relational analysis of NS_B1_A1_A1_A1_A2_B1_B2

### Relational analysis result of NS_B1_A1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1687900, upper bound: 65.1693239
time: 0.70 seconds

## BFS NS instance: NS_B1_A1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -94.9856949, 113.8814621, -246.1009674, 310.8190918, -405.8047791, 359.9824219
1: -10.8554583, 9.9324665, -28.9195366, 27.0668697, -37.9223289, 38.8520050
2: -6.2316704, 10.8189964, -16.7963982, 29.5477486, -35.7794189, 27.6153889
3: -5.0534482, 11.3767242, -13.8581133, 29.9367237, -34.9901695, 25.2348309
4: -7.4000573, 9.4728413, -20.2300453, 25.4878998, -32.8879585, 29.7028828

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_A1_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_B1_A1_A1_A1_A2_B2_B1

### Relational analysis result of NS_B1_A1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1687900, upper bound: 65.1683589
time: 0.65 seconds

## Relational analysis of NS_B1_A1_A1_A1_A2_B2_B2

### Relational analysis result of NS_B1_A1_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1687900, upper bound: 65.1693239
time: 0.67 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -262.3829956, 336.6220093, -90.1080246, 108.1620255, -370.5450134, 426.7300110
1: -31.4326057, 28.9870243, -10.3416023, 9.4061069, -40.8387146, 39.3286209
2: -18.0440483, 32.0328064, -5.8995852, 10.2631893, -28.3072376, 37.9323921
3: -14.7885447, 32.3920135, -4.7839789, 10.7745600, -25.5631027, 37.1759911
4: -21.7434425, 27.5584126, -7.0255108, 8.9616709, -30.7051125, 34.5839157

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_B1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1622596, upper bound: 65.1661926
time: 0.57 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B1_B2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1693319, upper bound: 65.1708538
time: 0.65 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -262.3829956, 336.6220093, -105.3710403, 135.6639709, -398.0469055, 441.9930420
1: -31.4326057, 28.9870243, -13.0996990, 11.3691397, -42.8017387, 42.0867195
2: -18.0440483, 32.0328064, -7.1339488, 12.9377890, -30.9818382, 39.1667557
3: -14.7885447, 32.3920135, -5.6908026, 13.3522415, -28.1407833, 38.0828133
4: -21.7434425, 27.5584126, -8.5788450, 11.1250610, -32.8684959, 36.1372528

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_B1

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1281338, upper bound: 65.1541813
time: 0.58 seconds

## Relational analysis of NS_B1_A1_A1_A2_B1_B2_B2

### Relational analysis result of NS_B1_A1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1695702, upper bound: 65.1711926
time: 0.73 seconds

## BFS NS instance: NS_B1_A1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -262.3829956, 336.6220093, -262.3829956, 336.6220093, -599.0050049, 599.0050049
1: -31.4326057, 28.9870243, -31.4326057, 28.9870243, -60.4196320, 60.4196320
2: -18.0440483, 32.0328064, -18.0440483, 32.0328064, -50.0768509, 50.0768547
3: -14.7885447, 32.3920135, -14.7885447, 32.3920135, -47.1805573, 47.1805573
4: -21.7434425, 27.5584126, -21.7434425, 27.5584126, -49.3018532, 49.3018532

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_A1_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_B1_A1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -262.3829956, 336.6220093, -261.9577637, 335.8591614, -598.2421265, 598.5797729
1: -31.4326057, 28.9870243, -31.3428001, 28.9604683, -60.3930740, 60.3298225
2: -18.0440483, 32.0328064, -17.9944477, 31.9712257, -50.0152664, 50.0272484
3: -14.7885447, 32.3920135, -14.7716713, 32.3005600, -47.0890999, 47.1636810
4: -21.7434425, 27.5584126, -21.7017422, 27.4991856, -49.2426262, 49.2601547

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_B1_A1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_B1_A1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_B1_A1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_B1_A1_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## BFS NS instance: NS_B1_A2_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -540.4608154, 685.1211548, -90.1080246, 108.1620255, -648.6228027, 775.2291870
1: -63.2654305, 60.0032310, -10.3416023, 9.4061069, -72.6715393, 70.3448334
2: -37.2363853, 64.9913483, -5.8995852, 10.2631893, -47.4995651, 70.8909302
3: -31.0137367, 65.6230621, -4.7839789, 10.7745600, -41.7882881, 70.4070435
4: -44.5642929, 56.2010536, -7.0255108, 8.9616709, -53.5259628, 63.2265587

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B1_A1_A1_B1

### Relational analysis result of NS_B1_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1552424, upper bound: 65.1548715
time: 0.59 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_A1_B2

### Relational analysis result of NS_B1_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1552424, upper bound: 65.1551791
time: 0.54 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -558.1775513, 710.9765015, -90.1080246, 108.1620255, -666.3395996, 801.0845337
1: -65.6738739, 62.0872612, -10.3416023, 9.4061069, -75.0799789, 72.4288635
2: -38.5794716, 67.3802490, -5.8995852, 10.2631893, -48.8426514, 73.2798309
3: -32.0401230, 68.0939941, -4.7839789, 10.7745600, -42.8146820, 72.8779755
4: -46.1598930, 58.2406311, -7.0255108, 8.9616709, -55.1215553, 65.2661438

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B1_A1_A2_B1

### Relational analysis result of NS_B1_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1567220, upper bound: 65.1611303
time: 0.59 seconds

## Relational analysis of NS_B1_A2_B1_B1_A1_A2_B2

### Relational analysis result of NS_B1_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1567220, upper bound: 65.1614379
time: 0.62 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -543.2243042, 685.2197266, -90.1080246, 108.1620255, -651.3863525, 775.3277588
1: -63.2760124, 60.1591339, -10.3416023, 9.4061069, -72.6821213, 70.5007324
2: -37.3296394, 65.0856171, -5.8995852, 10.2631893, -47.5928268, 70.9851990
3: -31.1666451, 65.7025986, -4.7839789, 10.7745600, -41.9412041, 70.4865799
4: -44.6684113, 56.3076477, -7.0255108, 8.9616709, -53.6300812, 63.3331604

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1551346, upper bound: 65.1544874
time: 0.62 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1551346, upper bound: 65.1547950
time: 0.59 seconds

## BFS NS instance: NS_B1_A2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -557.6950073, 707.9958496, -90.1080246, 108.1620255, -665.8570557, 798.1038818
1: -65.3920135, 61.9409561, -10.3416023, 9.4061069, -74.7981186, 72.2825623
2: -38.4749031, 67.1408920, -5.8995852, 10.2631893, -48.7380867, 73.0404739
3: -32.0124474, 67.8215942, -4.7839789, 10.7745600, -42.7870064, 72.6055679
4: -46.0361252, 58.0539818, -7.0255108, 8.9616709, -54.9977951, 65.0794907

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1570472, upper bound: 65.1621006
time: 0.64 seconds

## Relational analysis of NS_B1_A2_B1_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1566319, upper bound: 65.1624082
time: 0.59 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -540.4608154, 685.1211548, -105.3710403, 135.6639709, -676.1247559, 790.4921875
1: -63.2654305, 60.0032310, -13.0996990, 11.3691397, -74.6345673, 73.1029205
2: -37.2363853, 64.9913483, -7.1339488, 12.9377890, -50.1741753, 72.1252975
3: -31.0137367, 65.6230621, -5.6908026, 13.3522415, -44.3659744, 71.3138504
4: -44.5642929, 56.2010536, -8.5788450, 11.1250610, -55.6893539, 64.7798843

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B2_A1_A1_B1

### Relational analysis result of NS_B1_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1559456, upper bound: 65.1552871
time: 0.64 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_A1_B2

### Relational analysis result of NS_B1_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1559456, upper bound: 65.1552871
time: 0.64 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -558.1775513, 710.9765015, -105.3710403, 135.6639709, -693.8414917, 816.3475342
1: -65.6738739, 62.0872612, -13.0996990, 11.3691397, -77.0430145, 75.1869507
2: -38.5794716, 67.3802490, -7.1339488, 12.9377890, -51.5172615, 74.5141983
3: -32.0401230, 68.0939941, -5.6908026, 13.3522415, -45.3923645, 73.7847900
4: -46.1598930, 58.2406311, -8.5788450, 11.1250610, -57.2849541, 66.8194733

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B2_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B2_A1_A2_B1

### Relational analysis result of NS_B1_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1577600, upper bound: 65.1615459
time: 0.62 seconds

## Relational analysis of NS_B1_A2_B1_B2_A1_A2_B2

### Relational analysis result of NS_B1_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1567220, upper bound: 65.1615459
time: 0.66 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -543.2243042, 685.2197266, -105.3710403, 135.6639709, -678.8883057, 790.5907593
1: -63.2760124, 60.1591339, -13.0996990, 11.3691397, -74.6451492, 73.2588196
2: -37.3296394, 65.0856171, -7.1339488, 12.9377890, -50.2674294, 72.2195663
3: -31.1666451, 65.7025986, -5.6908026, 13.3522415, -44.5188866, 71.3933945
4: -44.6684113, 56.3076477, -8.5788450, 11.1250610, -55.7934723, 64.8864822

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B2_A2_A1_B1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1557954, upper bound: 65.1549030
time: 0.63 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A1_B2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1557954, upper bound: 65.1549030
time: 0.63 seconds

## BFS NS instance: NS_B1_A2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -557.6950073, 707.9958496, -105.3710403, 135.6639709, -693.3589478, 813.3668823
1: -65.3920135, 61.9409561, -13.0996990, 11.3691397, -76.7611542, 75.0406570
2: -38.4749031, 67.1408920, -7.1339488, 12.9377890, -51.4126930, 74.2748413
3: -32.0124474, 67.8215942, -5.6908026, 13.3522415, -45.3646889, 73.5123825
4: -46.0361252, 58.0539818, -8.5788450, 11.1250610, -57.1611862, 66.6328201

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_B1

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1577505, upper bound: 65.1625162
time: 0.71 seconds

## Relational analysis of NS_B1_A2_B1_B2_A2_A2_B2

### Relational analysis result of NS_B1_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1577505, upper bound: 65.1625162
time: 0.63 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -553.1695557, 707.5757446, -256.0330811, 329.5161133, -882.6856689, 963.6088257
1: -65.4144669, 61.6290855, -30.8023796, 28.3328476, -93.7473145, 92.4314499
2: -38.2487068, 67.0973129, -17.6074467, 31.3720188, -69.6207047, 84.7047501
3: -31.7518997, 67.7217560, -14.4139414, 31.7036228, -63.4555092, 82.1356888
4: -45.8250923, 57.9771385, -21.2392769, 26.9638958, -72.7889786, 79.2164078

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_B1_A1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_B1_A1_A1_B1

### Relational analysis result of NS_B1_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1520435, upper bound: 65.1537618
time: 0.62 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_A1_B2

### Relational analysis result of NS_B1_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1520435, upper bound: 65.1568628
time: 0.59 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -571.1734619, 733.5733643, -256.0330811, 329.5161133, -900.6895752, 989.6064453
1: -67.8365479, 63.7274971, -30.8023796, 28.3328476, -96.1693954, 94.5298386
2: -39.6020432, 69.5033264, -17.6074467, 31.3720188, -70.9740601, 87.1107712
3: -32.7914162, 70.2072906, -14.4139414, 31.7036228, -64.4950333, 84.6212234
4: -47.4302864, 60.0490608, -21.2392769, 26.9638958, -74.3941803, 81.2883148

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_B1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_B1_A1_A2_B1

### Relational analysis result of NS_B1_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1538579, upper bound: 65.1600206
time: 0.64 seconds

## Relational analysis of NS_B1_A2_B2_B1_A1_A2_B2

### Relational analysis result of NS_B1_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1538579, upper bound: 65.1631216
time: 0.67 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -555.6355591, 707.3342896, -256.0330811, 329.5161133, -885.1516724, 963.3673706
1: -65.4007034, 61.7513161, -30.8023796, 28.3328476, -93.7335510, 92.5536728
2: -38.3218765, 67.1594315, -17.6074467, 31.3720188, -69.6938858, 84.7668686
3: -31.8856926, 67.7238693, -14.4139414, 31.7036228, -63.5893097, 82.1378021
4: -45.9042969, 58.0167007, -21.2392769, 26.9638958, -72.8681870, 79.2559662

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_B1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1519357, upper bound: 65.1533777
time: 0.62 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A1_B2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1519357, upper bound: 65.1564787
time: 0.63 seconds

## BFS NS instance: NS_B1_A2_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -570.4838867, 730.3726196, -256.0330811, 329.5161133, -900.0000000, 986.4057007
1: -67.5415649, 63.5677299, -30.8023796, 28.3328476, -95.8744125, 94.3701096
2: -39.4901085, 69.2417984, -17.6074467, 31.3720188, -70.8621216, 86.8492432
3: -32.7502022, 69.9093781, -14.4139414, 31.7036228, -64.4538269, 84.3233032
4: -47.2939720, 59.8133202, -21.2392769, 26.9638958, -74.2578506, 81.0525818

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 37

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_A1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1163313, upper bound: 65.1047723
time: 0.60 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_B1

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1538483, upper bound: 65.1609909
time: 0.62 seconds

## Relational analysis of NS_B1_A2_B2_B1_A2_A2_B2

### Relational analysis result of NS_B1_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1534330, upper bound: 65.1640919
time: 0.68 seconds

## BFS NS instance: NS_B1_A2_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -579.1906738, 742.4566650, -243.4653320, 317.9923706, -897.1830444, 985.9219971
1: -68.6442719, 64.5340805, -29.9429379, 27.0541496, -95.6984253, 94.4770126
2: -40.1361885, 70.3845291, -16.7895985, 30.3581104, -70.4942932, 87.1741257
3: -33.2396164, 71.0645676, -13.6768017, 30.5357113, -63.7753258, 84.7413635
4: -48.0380592, 60.8068962, -20.3548908, 26.0388737, -74.0769196, 81.1617737

Time for backsubstitution: 2.58 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.20 + 416.83 = 421.03 seconds
