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
execution time: IAR + RelationalAnalysis = 2.61 + 1.68 = 4.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -65.1818525, upper bound: 65.1818525

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1817262, upper bound: 65.1799444
time: 0.55 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1815345, upper bound: 65.1815345
time: 0.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.33 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 4, lower bound: -65.1817262, upper bound: 65.1799444
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 4, lower bound: -65.1815345, upper bound: 65.1815345

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -191.2589111, 274.9101562, -344.2844543, 474.5872803, -665.8461914, 619.1945801
1: -26.4551468, 21.8712902, -44.8981438, 39.1306877, -65.5858307, 66.7694321
2: -13.8423862, 26.0311584, -24.5288010, 45.0728149, -58.9151993, 50.5599594
3: -10.6939602, 26.5576401, -19.4932232, 45.4991264, -56.1930847, 46.0508575
4: -16.6282597, 22.0045509, -29.6052933, 38.3870697, -55.0153275, 51.6098404

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1799444, upper bound: 65.1799444
time: 0.58 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1799444, upper bound: 65.1799444
time: 0.60 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -361.8001709, 501.6008911, -362.4209900, 502.3203430, -864.1204834, 864.0218506
1: -47.5034103, 41.2140923, -47.5665092, 41.2833595, -88.7867737, 88.7805710
2: -25.8481808, 47.6788597, -25.8910866, 47.7470360, -73.5952148, 73.5699463
3: -20.4953365, 48.0249825, -20.5314445, 48.0940323, -68.5893707, 68.5564270
4: -31.2060986, 40.5844193, -31.2567883, 40.6454163, -71.8515167, 71.8412094

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1727220, upper bound: 65.1766414
time: 0.73 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1727220, upper bound: 65.1766051
time: 0.75 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 4.12 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 4, lower bound: -65.1799444, upper bound: 65.1799444
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 4, lower bound: -65.1799444, upper bound: 65.1799444
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 4, lower bound: -65.1727220, upper bound: 65.1766414
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.12
Output dim: 4, lower bound: -65.1727220, upper bound: 65.1766051

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -191.2589111, 274.9101562, -191.2589111, 274.9101562, -466.1690674, 466.1690674
1: -26.4551468, 21.8712902, -26.4551468, 21.8712902, -48.3264275, 48.3264275
2: -13.8423862, 26.0311584, -13.8423862, 26.0311584, -39.8735390, 39.8735390
3: -10.6939602, 26.5576401, -10.6939602, 26.5576401, -37.2516022, 37.2516022
4: -16.6282597, 22.0045509, -16.6282597, 22.0045509, -38.6328087, 38.6328087

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1789980, upper bound: 65.1782557
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1801361, upper bound: 65.1799444
time: 0.59 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -191.2589111, 274.9101562, -361.2085266, 500.7966919, -692.0556030, 636.1186523
1: -26.4551468, 21.8712902, -47.4258537, 41.1484032, -67.6035461, 69.2971344
2: -13.8423862, 26.0311584, -25.8061504, 47.6020851, -61.4444656, 51.8373108
3: -10.6939602, 26.5576401, -20.4622231, 47.9464111, -58.6403694, 47.0198631
4: -16.6282597, 22.0045509, -31.1572189, 40.5191765, -57.1474266, 53.1617661

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1789980, upper bound: 65.1782557
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1801361, upper bound: 65.1799444
time: 0.59 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -357.5561218, 494.6552124, -266.2365112, 340.9533691, -698.5095215, 760.8917236
1: -46.8261185, 40.6984329, -31.8001957, 29.4300060, -76.2561188, 72.4986191
2: -25.5152016, 47.0185432, -18.2838287, 32.4590836, -57.9742851, 65.3023682
3: -20.2499790, 47.3636055, -15.0198545, 32.7886124, -53.0385857, 62.3834610
4: -30.8086281, 40.0309715, -22.0449448, 27.9282665, -58.7368927, 62.0759087

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1723638, upper bound: 65.1723638
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1723638, upper bound: 65.1766051
time: 0.60 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -353.9497375, 491.5305481, -581.7056885, 742.9239502, -1096.8736572, 1073.2360840
1: -46.5666199, 40.3606796, -68.6827316, 64.7128677, -111.2794876, 109.0434113
2: -25.3107319, 46.7003860, -40.2370834, 70.4836807, -95.7943878, 86.9374695
3: -20.0376587, 47.0659294, -33.3851509, 71.1173935, -91.1550522, 80.4510803
4: -30.5444202, 39.7694473, -48.1552124, 60.8823013, -91.4267120, 87.9246597

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1766051, upper bound: 65.1727220
time: 0.58 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1766051, upper bound: 65.1769633
time: 0.60 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.84 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 4, lower bound: -65.1789980, upper bound: 65.1782557
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 4, lower bound: -65.1801361, upper bound: 65.1799444
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 4, lower bound: -65.1789980, upper bound: 65.1782557
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 4, lower bound: -65.1801361, upper bound: 65.1799444
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 4, lower bound: -65.1723638, upper bound: 65.1723638
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 4, lower bound: -65.1723638, upper bound: 65.1766051
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 4, lower bound: -65.1766051, upper bound: 65.1727220
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.84
Output dim: 4, lower bound: -65.1766051, upper bound: 65.1769633

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -186.1605225, 268.7919617, -191.2589111, 274.9101562, -461.0706787, 460.0508118
1: -25.9075356, 21.3028069, -26.4551468, 21.8712902, -47.7788239, 47.7579498
2: -13.4903593, 25.4536266, -13.8423862, 26.0311584, -39.5215187, 39.2960052
3: -10.3921242, 25.9712334, -10.6939602, 26.5576401, -36.9497528, 36.6651802
4: -16.2008114, 21.4934731, -16.6282597, 22.0045509, -38.2053604, 38.1217232

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1773094, upper bound: 65.1773094
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1773094, upper bound: 65.1784474
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -202.1203461, 294.1962585, -191.2589111, 274.9101562, -477.0304871, 485.4551392
1: -28.5038471, 23.1495724, -26.4551468, 21.8712902, -50.3751373, 49.6047173
2: -14.6746120, 27.9998608, -13.8423862, 26.0311584, -40.7057686, 41.8422432
3: -11.2720556, 28.4441128, -10.6939602, 26.5576401, -37.8296967, 39.1380730
4: -17.6687298, 23.6376228, -16.6282597, 22.0045509, -39.6732788, 40.2658844

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1789980
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1801361
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -186.1605225, 268.7919617, -361.2085266, 500.7966919, -686.9572144, 630.0004883
1: -25.9075356, 21.3028069, -47.4258537, 41.1484032, -67.0559387, 68.7286606
2: -13.4903593, 25.4536266, -25.8061504, 47.6020851, -61.0924416, 51.2597771
3: -10.3921242, 25.9712334, -20.4622231, 47.9464111, -58.3385239, 46.4334564
4: -16.2008114, 21.4934731, -31.1572189, 40.5191765, -56.7199821, 52.6506882

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1773094, upper bound: 65.1773059
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1773094, upper bound: 65.1782557
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -202.1203461, 294.1962585, -361.2085266, 500.7966919, -702.9170532, 655.4047852
1: -28.5038471, 23.1495724, -47.4258537, 41.1484032, -69.6522446, 70.5754242
2: -14.6746120, 27.9998608, -25.8061504, 47.6020851, -62.2766953, 53.8060112
3: -11.2720556, 28.4441128, -20.4622231, 47.9464111, -59.2184639, 48.9063339
4: -17.6687298, 23.6376228, -31.1572189, 40.5191765, -58.1879044, 54.7948418

Time for backsubstitution: 2.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1789946
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1799444
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -265.6603394, 340.1935730, -266.2365112, 340.9533691, -606.6137085, 606.4299316
1: -31.7329540, 29.3631477, -31.8001957, 29.4300060, -61.1629601, 61.1633377
2: -18.2432175, 32.3864441, -18.2838287, 32.4590836, -50.7023010, 50.6702728
3: -14.9854231, 32.7193451, -15.0198545, 32.7886124, -47.7740364, 47.7392006
4: -21.9946251, 27.8653603, -22.0449448, 27.9282665, -49.9228897, 49.9103050

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1695340, upper bound: 65.1716344
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1695340, upper bound: 65.1724000
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -574.6646118, 734.5219727, -266.2365112, 340.9533691, -915.6179810, 1000.7584839
1: -67.8933105, 63.9520378, -31.8001957, 29.4300060, -97.3233185, 95.7522354
2: -39.7536736, 69.6544113, -18.2838287, 32.4590836, -72.2127533, 87.9382401
3: -32.9876480, 70.3029022, -15.0198545, 32.7886124, -65.7762604, 85.3227539
4: -47.5921249, 60.1794510, -22.0449448, 27.9282665, -75.5203934, 82.2243958

Time for backsubstitution: 2.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1695340, upper bound: 65.1756423
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1695340, upper bound: 65.1766414
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -265.6603394, 340.1935730, -581.7056885, 742.9239502, -1008.5842896, 921.8991699
1: -31.7329540, 29.3631477, -68.6827316, 64.7128677, -96.4458084, 98.0458832
2: -18.2432175, 32.3864441, -40.2370834, 70.4836807, -88.7268982, 72.6235275
3: -14.9854231, 32.7193451, -33.3851509, 71.1173935, -86.1027985, 66.1044922
4: -21.9946251, 27.8653603, -48.1552124, 60.8823013, -82.8769226, 76.0205688

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1497368, upper bound: 65.1375257
time: 0.57 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1302818, upper bound: 65.1314480
time: 0.58 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -574.6646118, 734.5219727, -581.7056885, 742.9239502, -1317.5885010, 1316.2276611
1: -67.8933105, 63.9520378, -68.6827316, 64.7128677, -132.6061707, 132.6347046
2: -39.7536736, 69.6544113, -40.2370834, 70.4836807, -110.2373505, 109.8914871
3: -32.9876480, 70.3029022, -33.3851509, 71.1173935, -104.1050415, 103.6880493
4: -47.5921249, 60.1794510, -48.1552124, 60.8823013, -108.4744263, 108.3346634

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1497368, upper bound: 65.1372997
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1302818, upper bound: 65.1457296
time: 0.56 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.90 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1773094, upper bound: 65.1773094
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1773094, upper bound: 65.1784474
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1789980
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1801361
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1773094, upper bound: 65.1773059
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1773094, upper bound: 65.1782557
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1789946
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1799444
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1695340, upper bound: 65.1716344
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1695340, upper bound: 65.1724000
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1695340, upper bound: 65.1756423
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1695340, upper bound: 65.1766414
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1497368, upper bound: 65.1375257
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1302818, upper bound: 65.1314480
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1497368, upper bound: 65.1372997
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.90
Output dim: 4, lower bound: -65.1302818, upper bound: 65.1457296

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -186.1605225, 268.7919617, -186.1605225, 268.7919617, -454.9524231, 454.9524231
1: -25.9075356, 21.3028069, -25.9075356, 21.3028069, -47.2103386, 47.2103424
2: -13.4903593, 25.4536266, -13.4903593, 25.4536266, -38.9439850, 38.9439812
3: -10.3921242, 25.9712334, -10.3921242, 25.9712334, -36.3633423, 36.3633423
4: -16.2008114, 21.4934731, -16.2008114, 21.4934731, -37.6942749, 37.6942787

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1517098, upper bound: 65.1666166
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1430378, upper bound: 65.1430378
time: 0.59 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -186.1605225, 268.7919617, -202.1203461, 294.1962585, -480.3567200, 470.9122314
1: -25.9075356, 21.3028069, -28.5038471, 23.1495724, -49.0571060, 49.8066559
2: -13.4903593, 25.4536266, -14.6746120, 27.9998608, -41.4902191, 40.1282349
3: -10.3921242, 25.9712334, -11.2720556, 28.4441128, -38.8362312, 37.2432861
4: -16.2008114, 21.4934731, -17.6687298, 23.6376228, -39.8384323, 39.1622009

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1517098, upper bound: 65.1695959
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1430378, upper bound: 65.1457196
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -202.1203461, 294.1962585, -186.1605225, 268.7919617, -470.9122314, 480.3567200
1: -28.5038471, 23.1495724, -25.9075356, 21.3028069, -49.8066559, 49.0571060
2: -14.6746120, 27.9998608, -13.4903593, 25.4536266, -40.1282349, 41.4902191
3: -11.2720556, 28.4441128, -10.3921242, 25.9712334, -37.2432861, 38.8362312
4: -17.6687298, 23.6376228, -16.2008114, 21.4934731, -39.1622009, 39.8384323

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1789980
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1776513, upper bound: 65.1780135
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -202.1203461, 294.1962585, -202.1203461, 294.1962585, -496.3165588, 496.3165588
1: -28.5038471, 23.1495724, -28.5038471, 23.1495724, -51.6534195, 51.6534195
2: -14.6746120, 27.9998608, -14.6746120, 27.9998608, -42.6744690, 42.6744728
3: -11.2720556, 28.4441128, -11.2720556, 28.4441128, -39.7161674, 39.7161674
4: -17.6687298, 23.6376228, -17.6687298, 23.6376228, -41.3063507, 41.3063507

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1793399
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1776513, upper bound: 65.1783346
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -186.1605225, 268.7919617, -352.3731384, 491.2232971, -677.3837891, 621.1650391
1: -25.9075356, 21.3028069, -46.5903053, 40.2132034, -66.1207428, 67.8931122
2: -13.4903593, 25.4536266, -25.2343578, 46.6940765, -60.1844292, 50.6879845
3: -10.3921242, 25.9712334, -19.9473095, 47.0373421, -57.4294548, 45.9185257
4: -16.2008114, 21.4934731, -30.4715214, 39.7144356, -55.9152451, 51.9649849

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1396605, upper bound: 65.1628413
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1302710, upper bound: 65.1380424
time: 0.52 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -186.1605225, 268.7919617, -363.5414124, 511.2002258, -697.3605347, 632.3333740
1: -25.9075356, 21.3028069, -48.6346436, 41.6172829, -67.5248184, 69.9374466
2: -13.4903593, 25.4536266, -26.1385975, 48.6437607, -62.1341209, 51.5922203
3: -10.3921242, 25.9712334, -20.5904675, 48.9339294, -59.3260422, 46.5616989
4: -16.2008114, 21.4934731, -31.5732975, 41.3568916, -57.5577011, 53.0667534

Time for backsubstitution: 2.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1396605, upper bound: 65.1642515
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1302710, upper bound: 65.1394550
time: 0.55 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -202.1203461, 294.1962585, -352.3731384, 491.2232971, -693.3436279, 646.5692749
1: -28.5038471, 23.1495724, -46.5903053, 40.2132034, -68.7170486, 69.7398758
2: -14.6746120, 27.9998608, -25.2343578, 46.6940765, -61.3686829, 53.2342186
3: -11.2720556, 28.4441128, -19.9473095, 47.0373421, -58.3093987, 48.3914185
4: -17.6687298, 23.6376228, -30.4715214, 39.7144356, -57.3831635, 54.1091461

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1789883
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1776513, upper bound: 65.1780038
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -202.1203461, 294.1962585, -363.5414124, 511.2002258, -713.3204346, 657.7376709
1: -28.5038471, 23.1495724, -48.6346436, 41.6172829, -70.1211166, 71.7842102
2: -14.6746120, 27.9998608, -26.1385975, 48.6437607, -63.3183670, 54.1384583
3: -11.2720556, 28.4441128, -20.5904675, 48.9339294, -60.2059822, 49.0345802
4: -17.6687298, 23.6376228, -31.5732975, 41.3568916, -59.0256195, 55.2109108

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1792957
time: 0.63 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1776513, upper bound: 65.1783112
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -265.6603394, 340.1935730, -94.2483063, 113.2373810, -378.8977051, 434.4418945
1: -31.7329540, 29.3631477, -10.7956877, 9.8754654, -41.6084213, 40.1588364
2: -18.2432175, 32.3864441, -6.1857562, 10.7456675, -28.9888840, 38.5721970
3: -14.9854231, 32.7193451, -5.0331597, 11.2536554, -26.2390785, 37.7525063
4: -21.9946251, 27.8653603, -7.3785291, 9.3780966, -31.3727207, 35.2438889

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1690807, upper bound: 65.1707203
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1680240, upper bound: 65.1716344
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -265.6603394, 340.1935730, -265.6603394, 340.1935730, -605.8538208, 605.8538208
1: -31.7329540, 29.3631477, -31.7329540, 29.3631477, -61.0960999, 61.0960999
2: -18.2432175, 32.3864441, -18.2432175, 32.3864441, -50.6296616, 50.6296616
3: -14.9854231, 32.7193451, -14.9854231, 32.7193451, -47.7047691, 47.7047691
4: -21.9946251, 27.8653603, -21.9946251, 27.8653603, -49.8599854, 49.8599854

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1690807, upper bound: 65.1714859
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1695702, upper bound: 65.1724000
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -574.6646118, 734.5219727, -94.2483063, 113.2373810, -687.9019775, 828.7702026
1: -67.8933105, 63.9520378, -10.7956877, 9.8754654, -77.7687759, 74.7477264
2: -39.7536736, 69.6544113, -6.1857562, 10.7456675, -50.4993401, 75.8401642
3: -32.9876480, 70.3029022, -5.0331597, 11.2536554, -44.2412949, 75.3360596
4: -47.5921249, 60.1794510, -7.3785291, 9.3780966, -56.9702225, 67.5579834

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1171982, upper bound: 65.1251641
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1167569, upper bound: 65.1258108
time: 0.56 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -574.6646118, 734.5219727, -265.6603394, 340.1935730, -914.8581543, 1000.1823120
1: -67.8933105, 63.9520378, -31.7329540, 29.3631477, -97.2564545, 95.6849899
2: -39.7536736, 69.6544113, -18.2432175, 32.3864441, -72.1401215, 87.8976288
3: -32.9876480, 70.3029022, -14.9854231, 32.7193451, -65.7069931, 85.2883224
4: -47.5921249, 60.1794510, -21.9946251, 27.8653603, -75.4574814, 82.1740723

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1171982, upper bound: 65.1506125
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1167569, upper bound: 65.1445634
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -265.6603394, 340.1935730, -561.8596191, 708.3527222, -974.0130615, 902.0531616
1: -31.7329540, 29.3631477, -65.4313126, 62.1607704, -93.8936920, 94.7944641
2: -18.2432175, 32.3864441, -38.6209602, 67.3633041, -85.6065216, 71.0074005
3: -14.9854231, 32.7193451, -32.2209969, 67.8655701, -82.8509903, 64.9403381
4: -21.9946251, 27.8653603, -46.1892548, 58.2638283, -80.2584534, 74.0546112

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1189666, upper bound: 65.1275267
time: 0.53 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1302818, upper bound: 65.1314480
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1302818, upper bound: 65.1314480
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -261.4108276, 335.0538025, -573.8910522, 729.0388184, -990.4496460, 908.9448242
1: -31.2639294, 28.9093933, -67.2914963, 63.8158035, -95.0797348, 96.2008896
2: -17.9550800, 31.8900661, -39.6246948, 69.1094742, -87.0645523, 71.5147552
3: -14.7468510, 32.2142754, -32.9491615, 69.7141418, -84.4609909, 65.1634293
4: -21.6558266, 27.4312859, -47.4117355, 59.7835350, -81.4393463, 74.8430176

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1302818, upper bound: 65.1314480
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1302818, upper bound: 65.1314480
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -574.6646118, 734.5219727, -561.8596191, 708.3527222, -1283.0173340, 1296.3815918
1: -67.8933105, 63.9520378, -65.4313126, 62.1607704, -130.0540771, 129.3833466
2: -39.7536736, 69.6544113, -38.6209602, 67.3633041, -107.1169739, 108.2753677
3: -32.9876480, 70.3029022, -32.2209969, 67.8655701, -100.8532181, 102.5238876
4: -47.5921249, 60.1794510, -46.1892548, 58.2638283, -105.8559570, 106.3687057

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1420610, upper bound: 65.1457296
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1420610, upper bound: 65.1457296
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -570.5820312, 729.5706787, -573.8910522, 729.0388184, -1299.6208496, 1303.4616699
1: -67.4238968, 63.5256615, -67.2914963, 63.8158035, -131.2397003, 130.8171539
2: -39.4780426, 69.1625595, -39.6246948, 69.1094742, -108.5875168, 108.7872391
3: -32.7580490, 69.8147888, -32.9491615, 69.7141418, -102.4721909, 102.7639465
4: -47.2681160, 59.7659607, -47.4117355, 59.7835350, -107.0516357, 107.1776962

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1420610, upper bound: 65.1457296
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1420610, upper bound: 65.1457296
time: 0.71 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.10 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1517098, upper bound: 65.1666166
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1430378, upper bound: 65.1430378
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1517098, upper bound: 65.1695959
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1430378, upper bound: 65.1457196
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1789980
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1776513, upper bound: 65.1780135
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1793399
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1776513, upper bound: 65.1783346
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1396605, upper bound: 65.1628413
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1302710, upper bound: 65.1380424
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1396605, upper bound: 65.1642515
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1302710, upper bound: 65.1394550
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1789883
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1776513, upper bound: 65.1780038
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1784474, upper bound: 65.1792957
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1776513, upper bound: 65.1783112
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1690807, upper bound: 65.1707203
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1680240, upper bound: 65.1716344
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1690807, upper bound: 65.1714859
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1695702, upper bound: 65.1724000
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1171982, upper bound: 65.1251641
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1167569, upper bound: 65.1258108
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1171982, upper bound: 65.1506125
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1167569, upper bound: 65.1445634
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1302818, upper bound: 65.1314480
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1302818, upper bound: 65.1314480
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1302818, upper bound: 65.1314480
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1302818, upper bound: 65.1314480
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1420610, upper bound: 65.1457296
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1420610, upper bound: 65.1457296
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1420610, upper bound: 65.1457296
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.10
Output dim: 4, lower bound: -65.1420610, upper bound: 65.1457296

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -184.2849579, 266.3168945, -186.1605225, 268.7919617, -453.0769043, 452.4773865
1: -25.6803474, 21.0845985, -25.9075356, 21.3028069, -46.9831543, 46.9921341
2: -13.3558750, 25.2196388, -13.4903593, 25.4536266, -38.8095016, 38.7099876
3: -10.2817898, 25.7319946, -10.3921242, 25.9712334, -36.2530212, 36.1241035
4: -16.0367928, 21.2913914, -16.2008114, 21.4934731, -37.5302582, 37.4922028

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1430378, upper bound: 65.1430378
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1430378, upper bound: 65.1430378
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -172.4574738, 247.2798462, -186.1605225, 268.7919617, -441.2493591, 433.4403687
1: -23.8030243, 19.6571522, -25.9075356, 21.3028069, -45.1058311, 45.5646896
2: -12.4467812, 23.4034328, -13.4903593, 25.4536266, -37.9004059, 36.8937912
3: -9.6119366, 23.8297672, -10.3921242, 25.9712334, -35.5831680, 34.2218895
4: -14.9525290, 19.7782326, -16.2008114, 21.4934731, -36.4459915, 35.9790421

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1430378, upper bound: 65.1430378
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1430378, upper bound: 65.1430378
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -184.2849579, 266.3168945, -202.1203461, 294.1962585, -478.4812012, 468.4371948
1: -25.6803474, 21.0845985, -28.5038471, 23.1495724, -48.8299179, 49.5884476
2: -13.3558750, 25.2196388, -14.6746120, 27.9998608, -41.3557358, 39.8942413
3: -10.2817898, 25.7319946, -11.2720556, 28.4441128, -38.7259026, 37.0040436
4: -16.0367928, 21.2913914, -17.6687298, 23.6376228, -39.6744156, 38.9601212

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1479832, upper bound: 65.1673635
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1584785, upper bound: 65.1689185
time: 0.62 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -172.4574738, 247.2798462, -202.1203461, 294.1962585, -466.6537170, 449.4002075
1: -23.8030243, 19.6571522, -28.5038471, 23.1495724, -46.9525986, 48.1609993
2: -12.4467812, 23.4034328, -14.6746120, 27.9998608, -40.4466400, 38.0780411
3: -9.6119366, 23.8297672, -11.2720556, 28.4441128, -38.0560493, 35.1018219
4: -14.9525290, 19.7782326, -17.6687298, 23.6376228, -38.5901527, 37.4469604

Time for backsubstitution: 2.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1383010, upper bound: 65.1414496
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1482470, upper bound: 65.1415348
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -208.8586426, 306.1687622, -186.1605225, 268.7919617, -477.6506042, 492.3292236
1: -29.7535706, 23.9384346, -25.9075356, 21.3028069, -51.0563774, 49.8459625
2: -15.1695271, 29.1884422, -13.4903593, 25.4536266, -40.6231499, 42.6787987
3: -11.6117439, 29.7047939, -10.3921242, 25.9712334, -37.5829697, 40.0969048
4: -18.2824612, 24.5876198, -16.2008114, 21.4934731, -39.7759171, 40.7884293

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1695959, upper bound: 65.1595168
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1457196, upper bound: 65.1504453
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -200.8121490, 292.5421143, -186.1605225, 268.7919617, -469.6041260, 478.7026367
1: -28.3555641, 23.0005474, -25.9075356, 21.3028069, -49.6583710, 48.9080811
2: -14.5805607, 27.8434219, -13.4903593, 25.4536266, -40.0341873, 41.3337746
3: -11.1938381, 28.2860050, -10.3921242, 25.9712334, -37.1650696, 38.6781120
4: -17.5569286, 23.5020504, -16.2008114, 21.4934731, -39.0503998, 39.7028618

Time for backsubstitution: 2.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1694609, upper bound: 65.1595168
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1455556, upper bound: 65.1504453
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -208.8586426, 306.1687622, -202.1203461, 294.1962585, -503.0549011, 508.2890930
1: -29.7535706, 23.9384346, -28.5038471, 23.1495724, -52.9031448, 52.4422722
2: -15.1695271, 29.1884422, -14.6746120, 27.9998608, -43.1693878, 43.8630524
3: -11.6117439, 29.7047939, -11.2720556, 28.4441128, -40.0558548, 40.9768486
4: -18.2824612, 24.5876198, -17.6687298, 23.6376228, -41.9200821, 42.2563477

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1622903, upper bound: 65.1714408
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1791279, upper bound: 65.1793399
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -200.8121490, 292.5421143, -202.1203461, 294.1962585, -495.0084229, 494.6624146
1: -28.3555641, 23.0005474, -28.5038471, 23.1495724, -51.5051346, 51.5043945
2: -14.5805607, 27.8434219, -14.6746120, 27.9998608, -42.5804214, 42.5180283
3: -11.1938381, 28.2860050, -11.2720556, 28.4441128, -39.6379509, 39.5580521
4: -17.5569286, 23.5020504, -17.6687298, 23.6376228, -41.1945496, 41.1707802

Time for backsubstitution: 2.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1783554, upper bound: 65.1783234
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1783554, upper bound: 65.1783346
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -184.2849579, 266.3168945, -352.3731384, 491.2232971, -675.5082397, 618.6898804
1: -25.6803474, 21.0845985, -46.5903053, 40.2132034, -65.8935471, 67.6749039
2: -13.3558750, 25.2196388, -25.2343578, 46.6940765, -60.0499496, 50.4539948
3: -10.2817898, 25.7319946, -19.9473095, 47.0373421, -57.3191299, 45.6792908
4: -16.0367928, 21.2913914, -30.4715214, 39.7144356, -55.7512245, 51.7629128

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1030942, upper bound: 65.1182255
time: 0.53 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1072949, upper bound: 65.1427821
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1328958, upper bound: 65.1592202
time: 0.57 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -172.4574738, 247.2798462, -352.3731384, 491.2232971, -663.6806641, 599.6528320
1: -23.8030243, 19.6571522, -46.5903053, 40.2132034, -64.0162201, 66.2474594
2: -12.4467812, 23.4034328, -25.2343578, 46.6940765, -59.1408539, 48.6377907
3: -9.6119366, 23.8297672, -19.9473095, 47.0373421, -56.6492767, 43.7770767
4: -14.9525290, 19.7782326, -30.4715214, 39.7144356, -54.6669617, 50.2497559

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.0984013, upper bound: 65.1177561
time: 0.60 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1211569, upper bound: 65.1286815
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -184.2849579, 266.3168945, -363.5414124, 511.2002258, -695.4851074, 629.8582764
1: -25.6803474, 21.0845985, -48.6346436, 41.6172829, -67.2976227, 69.7192307
2: -13.3558750, 25.2196388, -26.1385975, 48.6437607, -61.9996338, 51.3582306
3: -10.2817898, 25.7319946, -20.5904675, 48.9339294, -59.2157211, 46.3224640
4: -16.0367928, 21.2913914, -31.5732975, 41.3568916, -57.3936844, 52.8646774

Time for backsubstitution: 2.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1343293, upper bound: 65.1543357
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1355828, upper bound: 65.1394550
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1355828, upper bound: 65.1394550
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -172.4574738, 247.2798462, -363.5414124, 511.2002258, -683.6574707, 610.8212280
1: -23.8030243, 19.6571522, -48.6346436, 41.6172829, -65.4202957, 68.2917862
2: -12.4467812, 23.4034328, -26.1385975, 48.6437607, -61.0905418, 49.5420303
3: -9.6119366, 23.8297672, -20.5904675, 48.9339294, -58.5458641, 44.4202347
4: -14.9525290, 19.7782326, -31.5732975, 41.3568916, -56.3094215, 51.3515167

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1200708, upper bound: 65.1152673
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1355828, upper bound: 65.1394550
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1355828, upper bound: 65.1394550
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -208.8586426, 306.1687622, -352.3731384, 491.2232971, -700.0819092, 658.5417480
1: -29.7535706, 23.9384346, -46.5903053, 40.2132034, -69.9667664, 70.5287399
2: -15.1695271, 29.1884422, -25.2343578, 46.6940765, -61.8635902, 54.4227982
3: -11.6117439, 29.7047939, -19.9473095, 47.0373421, -58.6490822, 49.6520958
4: -18.2824612, 24.5876198, -30.4715214, 39.7144356, -57.9968910, 55.0591431

Time for backsubstitution: 2.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1719885, upper bound: 65.1595168
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1332432, upper bound: 65.1463435
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -200.8121490, 292.5421143, -352.3731384, 491.2232971, -692.0354004, 644.9151611
1: -28.3555641, 23.0005474, -46.5903053, 40.2132034, -68.5687714, 69.5908508
2: -14.5805607, 27.8434219, -25.2343578, 46.6940765, -61.2746315, 53.0777817
3: -11.1938381, 28.2860050, -19.9473095, 47.0373421, -58.2311783, 48.2332993
4: -17.5569286, 23.5020504, -30.4715214, 39.7144356, -57.2713623, 53.9735718

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1552584, upper bound: 65.1561636
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1809300, upper bound: 65.1780038
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -208.8586426, 306.1687622, -363.5414124, 511.2002258, -720.0588379, 669.7102051
1: -29.7535706, 23.9384346, -48.6346436, 41.6172829, -71.3708420, 72.5730667
2: -15.1695271, 29.1884422, -26.1385975, 48.6437607, -63.8132782, 55.3270378
3: -11.6117439, 29.7047939, -20.5904675, 48.9339294, -60.5456657, 50.2952614
4: -18.2824612, 24.5876198, -31.5732975, 41.3568916, -59.6393471, 56.1609077

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1808055, upper bound: 65.1783112
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1722824, upper bound: 65.1598230
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1385546, upper bound: 65.1477537
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -200.8121490, 292.5421143, -363.5414124, 511.2002258, -712.0122681, 656.0834961
1: -28.3555641, 23.0005474, -48.6346436, 41.6172829, -69.9728317, 71.6351929
2: -14.5805607, 27.8434219, -26.1385975, 48.6437607, -63.2243195, 53.9820175
3: -11.1938381, 28.2860050, -20.5904675, 48.9339294, -60.1277695, 48.8764725
4: -17.5569286, 23.5020504, -31.5732975, 41.3568916, -58.9138184, 55.0753403

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1808055, upper bound: 65.1783112
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1682086, upper bound: 65.1681135
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1675839, upper bound: 65.1681254
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -256.0330811, 329.5161133, -94.2483063, 113.2373810, -369.2704468, 423.7644043
1: -30.8023796, 28.3328476, -10.7956877, 9.8754654, -40.6778450, 39.1285362
2: -17.6074467, 31.3720188, -6.1857562, 10.7456675, -28.3531132, 37.5577736
3: -14.4139414, 31.7036228, -5.0331597, 11.2536554, -25.6675968, 36.7367706
4: -21.2392769, 26.9638958, -7.3785291, 9.3780966, -30.6173744, 34.3424149

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1688424, upper bound: 65.1699398
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1688424, upper bound: 65.1707203
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -267.4748840, 350.4080200, -94.2483063, 113.2373810, -380.7122803, 444.6563110
1: -32.9838295, 29.7537632, -10.7956877, 9.8754654, -42.8592949, 40.5494499
2: -18.5207596, 33.4551468, -6.1857562, 10.7456675, -29.2664261, 39.6408958
3: -15.0604372, 33.6691856, -5.0331597, 11.2536554, -26.3140926, 38.7023392
4: -22.4012814, 28.6729298, -7.3785291, 9.3780966, -31.7793770, 36.0514565

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1693319, upper bound: 65.1708538
time: 0.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1693319, upper bound: 65.1716344
time: 0.61 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -256.0330811, 329.5161133, -265.6603394, 340.1935730, -596.2265625, 595.1764526
1: -30.8023796, 28.3328476, -31.7329540, 29.3631477, -60.1655273, 60.0657997
2: -17.6074467, 31.3720188, -18.2432175, 32.3864441, -49.9938889, 49.6152344
3: -14.4139414, 31.7036228, -14.9854231, 32.7193451, -47.1332855, 46.6890450
4: -21.2392769, 26.9638958, -21.9946251, 27.8653603, -49.1046371, 48.9585114

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1677857, upper bound: 65.1709964
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1688424, upper bound: 65.1714859
time: 0.60 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -267.4748840, 350.4080200, -265.6603394, 340.1935730, -607.6683960, 616.0682983
1: -32.9838295, 29.7537632, -31.7329540, 29.3631477, -62.3469772, 61.4867134
2: -18.5207596, 33.4551468, -18.2432175, 32.3864441, -50.9071999, 51.6983643
3: -15.0604372, 33.6691856, -14.9854231, 32.7193451, -47.7797813, 48.6546097
4: -22.4012814, 28.6729298, -21.9946251, 27.8653603, -50.2666397, 50.6675529

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1713745, upper bound: 65.1718805
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1713745, upper bound: 65.1724000
time: 0.62 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -554.5723877, 699.5662231, -94.2483063, 113.2373810, -667.8097534, 793.8145142
1: -64.6112289, 61.3695488, -10.7956877, 9.8754654, -74.4866867, 72.1652374
2: -38.1156044, 66.4977341, -6.1857562, 10.7456675, -48.8612709, 72.6834869
3: -31.8074665, 66.9943390, -5.0331597, 11.2536554, -43.0611191, 72.0274963
4: -45.6021271, 57.5163383, -7.3785291, 9.3780966, -54.9802246, 64.8948669

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1167569, upper bound: 65.1251464
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1167569, upper bound: 65.1251464
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -567.0339966, 720.9310303, -92.5215149, 110.2408905, -677.2749023, 813.4525146
1: -66.5289001, 63.0782661, -10.5133400, 9.6477785, -76.1766739, 73.5915985
2: -39.1545448, 68.3070602, -6.0437269, 10.4688826, -49.6234283, 74.3507843
3: -32.5626411, 68.9275818, -4.9293847, 10.9717360, -43.5343704, 73.8569489
4: -46.8643570, 59.1056252, -7.2021489, 9.1566401, -56.0209961, 66.3077774

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 19

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.1064501, upper bound: 65.1079854
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1167569, upper bound: 65.1258108
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1167569, upper bound: 65.1258108
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -554.5723877, 699.5662231, -265.6603394, 340.1935730, -894.7659302, 965.2265625
1: -64.6112289, 61.3695488, -31.7329540, 29.3631477, -93.9743805, 93.1024933
2: -38.1156044, 66.4977341, -18.2432175, 32.3864441, -70.5020447, 84.7409515
3: -31.8074665, 66.9943390, -14.9854231, 32.7193451, -64.5268097, 81.9797516
4: -45.6021271, 57.5163383, -21.9946251, 27.8653603, -73.4674835, 79.5109634

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1258718, upper bound: 65.1147447
time: 0.59 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1157630, upper bound: 65.1445634
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1157630, upper bound: 65.1445634
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -567.0339966, 720.9310303, -261.4108276, 335.0538025, -902.0877686, 982.3418579
1: -66.5289001, 63.0782661, -31.2639294, 28.9093933, -95.4382935, 94.3421860
2: -39.1545448, 68.3070602, -17.9550800, 31.8900661, -71.0446014, 86.2621384
3: -32.5626411, 68.9275818, -14.7468510, 32.2142754, -64.7769012, 83.6744232
4: -46.8643570, 59.1056252, -21.6558266, 27.4312859, -74.2956314, 80.7614441

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1314480, upper bound: 65.1445634
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1314480, upper bound: 65.1445634
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -243.1574707, 303.4013672, -561.8596191, 708.3527222, -951.5101929, 865.2608643
1: -28.2721748, 26.6008034, -65.4313126, 62.1607704, -90.4329300, 92.0321198
2: -16.4910641, 28.9951496, -38.6209602, 67.3633041, -83.8543701, 67.6161041
3: -13.6620102, 29.2383442, -32.2209969, 67.8655701, -81.5275803, 61.4593391
4: -19.8546906, 25.0293312, -46.1892548, 58.2638283, -78.1185074, 71.2185822

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.0781292, upper bound: 65.1169068
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.0780033, upper bound: 65.1375257
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -255.8569489, 324.3203735, -561.8596191, 708.3527222, -964.2096558, 886.1798706
1: -30.2260838, 28.2482014, -65.4313126, 62.1607704, -92.3868332, 93.6795120
2: -17.4973831, 30.8774490, -38.6209602, 67.3633041, -84.8606873, 69.4984131
3: -14.4521065, 31.1347561, -32.2209969, 67.8655701, -82.3176727, 63.3557510
4: -21.1019039, 26.5995960, -46.1892548, 58.2638283, -79.3657303, 72.7888489

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.0875904, upper bound: 65.1169068
time: 0.55 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1497368, upper bound: 65.1375257
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -243.1574707, 303.4013672, -573.8910522, 729.0388184, -972.1962891, 877.2922974
1: -28.2721748, 26.6008034, -67.2914963, 63.8158035, -92.0879822, 93.8922882
2: -16.4910641, 28.9951496, -39.6246948, 69.1094742, -85.6005325, 68.6198349
3: -13.6620102, 29.2383442, -32.9491615, 69.7141418, -83.3761520, 62.1875038
4: -19.8546906, 25.0293312, -47.4117355, 59.7835350, -79.6382065, 72.4410706

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.0875904, upper bound: 65.1297628
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1208207, upper bound: 65.1314480
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -255.8569489, 324.3203735, -573.8910522, 729.0388184, -984.8957520, 898.2113037
1: -30.2260838, 28.2482014, -67.2914963, 63.8158035, -94.0418854, 95.5396957
2: -17.4973831, 30.8774490, -39.6246948, 69.1094742, -86.6068573, 70.5021439
3: -14.4521065, 31.1347561, -32.9491615, 69.7141418, -84.1662445, 64.0839157
4: -21.1019039, 26.5995960, -47.4117355, 59.7835350, -80.8854370, 74.0113297

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.0874645, upper bound: 65.1243729
time: 0.69 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1249909, upper bound: 65.1314480
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -554.5723877, 699.5662231, -561.8596191, 708.3527222, -1262.9250488, 1261.4257812
1: -64.6112289, 61.3695488, -65.4313126, 62.1607704, -126.7719727, 126.8008575
2: -38.1156044, 66.4977341, -38.6209602, 67.3633041, -105.4789124, 105.1186905
3: -31.8074665, 66.9943390, -32.2209969, 67.8655701, -99.6730347, 99.2153168
4: -45.6021271, 57.5163383, -46.1892548, 58.2638283, -103.8659515, 103.7055893

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.0887566, upper bound: 65.1310940
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1576760, upper bound: 65.1548179
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -567.0339966, 720.9310303, -561.8596191, 708.3527222, -1275.3864746, 1282.7906494
1: -66.5289001, 63.0782661, -65.4313126, 62.1607704, -128.6896515, 128.5095825
2: -39.1545448, 68.3070602, -38.6209602, 67.3633041, -106.5178528, 106.9280243
3: -32.5626411, 68.9275818, -32.2209969, 67.8655701, -100.4282074, 101.1485596
4: -46.8643570, 59.1056252, -46.1892548, 58.2638283, -105.1281586, 105.2948685

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.0887566, upper bound: 65.1310940
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1576760, upper bound: 65.1548179
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -554.5723877, 699.5662231, -573.8910522, 729.0388184, -1283.6112061, 1273.4570312
1: -64.6112289, 61.3695488, -67.2914963, 63.8158035, -128.4270325, 128.6610413
2: -38.1156044, 66.4977341, -39.6246948, 69.1094742, -107.2250824, 106.1224213
3: -31.8074665, 66.9943390, -32.9491615, 69.7141418, -101.5215988, 99.9434967
4: -45.6021271, 57.5163383, -47.4117355, 59.7835350, -105.3856506, 104.9280701

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.0874645, upper bound: 65.1286520
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1415046, upper bound: 65.1457296
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -567.0339966, 720.9310303, -573.8910522, 729.0388184, -1296.0726318, 1294.8220215
1: -66.5289001, 63.0782661, -67.2914963, 63.8158035, -130.3446960, 130.3697662
2: -39.1545448, 68.3070602, -39.6246948, 69.1094742, -108.2640228, 107.9317398
3: -32.5626411, 68.9275818, -32.9491615, 69.7141418, -102.2767792, 101.8767395
4: -46.8643570, 59.1056252, -47.4117355, 59.7835350, -106.6478577, 106.5173645

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.0874645, upper bound: 65.1395247
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.0884985, upper bound: 65.1313106
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1360005, upper bound: 65.1394957
time: 0.76 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 7.49 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1430378, upper bound: 65.1430378
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1430378, upper bound: 65.1430378
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1430378, upper bound: 65.1430378
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1430378, upper bound: 65.1430378
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1479832, upper bound: 65.1673635
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1584785, upper bound: 65.1689185
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1383010, upper bound: 65.1414496
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1482470, upper bound: 65.1415348
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1695959, upper bound: 65.1595168
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1457196, upper bound: 65.1504453
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1694609, upper bound: 65.1595168
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1455556, upper bound: 65.1504453
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1622903, upper bound: 65.1714408
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1791279, upper bound: 65.1793399
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1783554, upper bound: 65.1783234
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1783554, upper bound: 65.1783346
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1072949, upper bound: 65.1427821
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1328958, upper bound: 65.1592202
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.0984013, upper bound: 65.1177561
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1211569, upper bound: 65.1286815
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1355828, upper bound: 65.1394550
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1355828, upper bound: 65.1394550
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1355828, upper bound: 65.1394550
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1355828, upper bound: 65.1394550
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1719885, upper bound: 65.1595168
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1332432, upper bound: 65.1463435
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1552584, upper bound: 65.1561636
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1809300, upper bound: 65.1780038
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1722824, upper bound: 65.1598230
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1385546, upper bound: 65.1477537
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1682086, upper bound: 65.1681135
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1675839, upper bound: 65.1681254
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1688424, upper bound: 65.1699398
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1688424, upper bound: 65.1707203
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1693319, upper bound: 65.1708538
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1693319, upper bound: 65.1716344
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1677857, upper bound: 65.1709964
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1688424, upper bound: 65.1714859
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1713745, upper bound: 65.1718805
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1713745, upper bound: 65.1724000
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1167569, upper bound: 65.1251464
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1167569, upper bound: 65.1251464
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1167569, upper bound: 65.1258108
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1167569, upper bound: 65.1258108
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1157630, upper bound: 65.1445634
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1157630, upper bound: 65.1445634
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1314480, upper bound: 65.1445634
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1314480, upper bound: 65.1445634
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.0781292, upper bound: 65.1169068
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.0780033, upper bound: 65.1375257
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.0875904, upper bound: 65.1169068
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1497368, upper bound: 65.1375257
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.0875904, upper bound: 65.1297628
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1208207, upper bound: 65.1314480
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.0874645, upper bound: 65.1243729
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1249909, upper bound: 65.1314480
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.0887566, upper bound: 65.1310940
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1576760, upper bound: 65.1548179
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.0887566, upper bound: 65.1310940
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1576760, upper bound: 65.1548179
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.0874645, upper bound: 65.1286520
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1415046, upper bound: 65.1457296
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.0884985, upper bound: 65.1313106
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.49
Output dim: 4, lower bound: -65.1360005, upper bound: 65.1394957

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -184.2849579, 266.3168945, -184.2849579, 266.3168945, -450.6018677, 450.6018677
1: -25.6803474, 21.0845985, -25.6803474, 21.0845985, -46.7649460, 46.7649460
2: -13.3558750, 25.2196388, -13.3558750, 25.2196388, -38.5755081, 38.5755081
3: -10.2817898, 25.7319946, -10.2817898, 25.7319946, -36.0137825, 36.0137825
4: -16.0367928, 21.2913914, -16.0367928, 21.2913914, -37.3281860, 37.3281860

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1406327, upper bound: 65.1450958
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1485026, upper bound: 65.1655103
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -184.2849579, 266.3168945, -172.4574738, 247.2798462, -431.5648193, 438.7743225
1: -25.6803474, 21.0845985, -23.8030243, 19.6571522, -45.3375015, 44.8876228
2: -13.3558750, 25.2196388, -12.4467812, 23.4034328, -36.7593079, 37.6664085
3: -10.2817898, 25.7319946, -9.6119366, 23.8297672, -34.1115570, 35.3439255
4: -16.0367928, 21.2913914, -14.9525290, 19.7782326, -35.8150253, 36.2439194

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1406327, upper bound: 65.1450958
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1485026, upper bound: 65.1655103
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -172.4574738, 247.2798462, -184.2849579, 266.3168945, -438.7743225, 431.5648193
1: -23.8030243, 19.6571522, -25.6803474, 21.0845985, -44.8876228, 45.3375015
2: -12.4467812, 23.4034328, -13.3558750, 25.2196388, -37.6664124, 36.7593079
3: -9.6119366, 23.8297672, -10.2817898, 25.7319946, -35.3439217, 34.1115570
4: -14.9525290, 19.7782326, -16.0367928, 21.2913914, -36.2439194, 35.8150253

Time for backsubstitution: 2.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1385940, upper bound: 65.1383010
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1383444, upper bound: 65.1383444
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -172.4574738, 247.2798462, -172.4574738, 247.2798462, -419.7373047, 419.7373047
1: -23.8030243, 19.6571522, -23.8030243, 19.6571522, -43.4601746, 43.4601746
2: -12.4467812, 23.4034328, -12.4467812, 23.4034328, -35.8502121, 35.8502121
3: -9.6119366, 23.8297672, -9.6119366, 23.8297672, -33.4417038, 33.4417038
4: -14.9525290, 19.7782326, -14.9525290, 19.7782326, -34.7307549, 34.7307587

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1385940, upper bound: 65.1383010
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1383444, upper bound: 65.1383444
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -183.9679108, 265.8276367, -178.3206177, 255.1539764, -439.1218262, 444.1481323
1: -25.6342239, 21.0460358, -24.8361530, 20.1406612, -45.7748871, 45.8821869
2: -13.3317776, 25.1745758, -12.7909937, 24.5071220, -37.8388939, 37.9655685
3: -10.2629509, 25.6849785, -9.8329096, 24.6745224, -34.9374733, 35.5178833
4: -16.0072956, 21.2530937, -15.3634348, 20.6624031, -36.6697006, 36.6165276

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1490316, upper bound: 65.1482010
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1404799, upper bound: 65.1673635
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -184.2849579, 266.3168945, -200.7303467, 292.3340759, -476.6190186, 467.0472107
1: -25.6803474, 21.0845985, -28.3354492, 22.9867592, -48.6670990, 49.4200478
2: -13.3558750, 25.2196388, -14.5732021, 27.8242264, -41.1800995, 39.7928314
3: -10.2817898, 25.7319946, -11.1888981, 28.2665634, -38.5483551, 36.9208832
4: -16.0367928, 21.2913914, -17.5462837, 23.4860039, -39.5227966, 38.8376770

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1406327, upper bound: 65.1482862
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1505353, upper bound: 65.1689185
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -172.1476288, 246.7915802, -178.3206177, 255.1539764, -427.3016052, 425.1121216
1: -23.7569008, 19.6188850, -24.8361530, 20.1406612, -43.8975601, 44.4550400
2: -12.4229984, 23.3586178, -12.7909937, 24.5071220, -36.9301109, 36.1496086
3: -9.5933924, 23.7829762, -9.8329096, 24.6745224, -34.2679138, 33.6158867
4: -14.9233456, 19.7401524, -15.3634348, 20.6624031, -35.5857468, 35.1035881

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1381887, upper bound: 65.1412968
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1467404, upper bound: 65.1414496
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -172.4574738, 247.2798462, -200.7303467, 292.3340759, -464.7915649, 448.0101929
1: -23.8030243, 19.6571522, -28.3354492, 22.9867592, -46.7897644, 47.9925995
2: -12.4467812, 23.4034328, -14.5732021, 27.8242264, -40.2710075, 37.9766312
3: -9.6119366, 23.8297672, -11.1888981, 28.2665634, -37.8785019, 35.0186653
4: -14.9525290, 19.7782326, -17.5462837, 23.4860039, -38.4385338, 37.3245163

Time for backsubstitution: 2.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1383415, upper bound: 65.1413820
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1383415, upper bound: 65.1415348
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -208.8586426, 306.1687622, -184.2849579, 266.3168945, -475.1755371, 490.4537354
1: -29.7535706, 23.9384346, -25.6803474, 21.0845985, -50.8381691, 49.6187706
2: -15.1695271, 29.1884422, -13.3558750, 25.2196388, -40.3891563, 42.5443192
3: -11.6117439, 29.7047939, -10.2817898, 25.7319946, -37.3437309, 39.9865837
4: -18.2824612, 24.5876198, -16.0367928, 21.2913914, -39.5738487, 40.6244125

Time for backsubstitution: 2.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1457196, upper bound: 65.1504453
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1457196, upper bound: 65.1504453
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -208.8586426, 306.1687622, -172.4574738, 247.2798462, -456.1384888, 478.6261902
1: -29.7535706, 23.9384346, -23.8030243, 19.6571522, -49.4107208, 47.7414513
2: -15.1695271, 29.1884422, -12.4467812, 23.4034328, -38.5729523, 41.6352196
3: -11.6117439, 29.7047939, -9.6119366, 23.8297672, -35.4415131, 39.3167305
4: -18.2824612, 24.5876198, -14.9525290, 19.7782326, -38.0606880, 39.5401497

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1414496, upper bound: 65.1467433
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1457196, upper bound: 65.1504453
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1457196, upper bound: 65.1504453
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -200.8121490, 292.5421143, -184.2849579, 266.3168945, -467.1290283, 476.8270874
1: -28.3555641, 23.0005474, -25.6803474, 21.0845985, -49.4401627, 48.6808929
2: -14.5805607, 27.8434219, -13.3558750, 25.2196388, -39.8001938, 41.1992950
3: -11.1938381, 28.2860050, -10.2817898, 25.7319946, -36.9258270, 38.5677948
4: -17.5569286, 23.5020504, -16.0367928, 21.2913914, -38.8483200, 39.5388412

Time for backsubstitution: 2.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1169762, upper bound: 65.0954864
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1664022, upper bound: 65.1546033
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1687088, upper bound: 65.1583481
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -200.8121490, 292.5421143, -172.4574738, 247.2798462, -448.0919800, 464.9995728
1: -28.3555641, 23.0005474, -23.8030243, 19.6571522, -48.0127182, 46.8035622
2: -14.5805607, 27.8434219, -12.4467812, 23.4034328, -37.9839935, 40.2901993
3: -11.1938381, 28.2860050, -9.6119366, 23.8297672, -35.0236053, 37.8979340
4: -17.5569286, 23.5020504, -14.9525290, 19.7782326, -37.3351593, 38.4545784

Time for backsubstitution: 2.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1405845, upper bound: 65.1447838
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1413250, upper bound: 65.1481166
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -208.5480652, 305.6879272, -178.3206177, 255.1539764, -463.7020264, 484.0085144
1: -29.7081451, 23.9006443, -24.8361530, 20.1406612, -49.8487968, 48.7367935
2: -15.1458874, 29.1442070, -12.7909937, 24.5071220, -39.6529999, 41.9351997
3: -11.5933313, 29.6585884, -9.8329096, 24.6745224, -36.2678528, 39.4914894
4: -18.2535648, 24.5500717, -15.3634348, 20.6624031, -38.9159698, 39.9135056

Time for backsubstitution: 2.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1537503, upper bound: 65.1570469
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1537503, upper bound: 65.1714408
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -208.8586426, 306.1687622, -200.7303467, 292.3340759, -501.1927185, 506.8990479
1: -29.7535706, 23.9384346, -28.3354492, 22.9867592, -52.7403221, 52.2738762
2: -15.1695271, 29.1884422, -14.5732021, 27.8242264, -42.9937439, 43.7616386
3: -11.6117439, 29.7047939, -11.1888981, 28.2665634, -39.8783073, 40.8936844
4: -18.2824612, 24.5876198, -17.5462837, 23.4860039, -41.7684631, 42.1339035

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1710577, upper bound: 65.1633708
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1710577, upper bound: 65.1793399
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -200.8121490, 292.5421143, -208.8586426, 306.1687622, -506.9808960, 501.4007568
1: -28.3555641, 23.0005474, -29.7535706, 23.9384346, -52.2939949, 52.7541161
2: -14.5805607, 27.8434219, -15.1695271, 29.1884422, -43.7690010, 43.0129395
3: -11.1938381, 28.2860050, -11.6117439, 29.7047939, -40.8986282, 39.8977356
4: -17.5569286, 23.5020504, -18.2824612, 24.5876198, -42.1445465, 41.7845078

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1565244, upper bound: 65.1641358
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1529631, upper bound: 65.1529334
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -200.8121490, 292.5421143, -200.8121490, 292.5421143, -493.3542480, 493.3542480
1: -28.3555641, 23.0005474, -28.3555641, 23.0005474, -51.3561096, 51.3561096
2: -14.5805607, 27.8434219, -14.5805607, 27.8434219, -42.4239769, 42.4239807
3: -11.1938381, 28.2860050, -11.1938381, 28.2860050, -39.4798393, 39.4798393
4: -17.5569286, 23.5020504, -17.5569286, 23.5020504, -41.0589790, 41.0589790

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1695646, upper bound: 65.1612275
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1783554, upper bound: 65.1783234
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -175.7236328, 252.8533325, -294.8692627, 410.0855408, -585.8092041, 547.7225342
1: -24.4144211, 20.0151806, -39.0235329, 33.4909554, -57.9053726, 59.0387115
2: -12.6873636, 23.9839725, -20.9863758, 39.1346283, -51.8219910, 44.9703484
3: -9.7608805, 24.4249020, -16.5540371, 39.2997131, -49.0605927, 40.9789391
4: -15.2187557, 20.2497063, -25.4112682, 33.2894211, -48.5081787, 45.6609573

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.0958685, upper bound: 65.1314382
time: 0.57 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.0958613, upper bound: 65.1330556
time: 0.60 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -184.2849579, 266.3168945, -347.4291077, 484.7604370, -669.0454102, 613.7459717
1: -25.6803474, 21.0845985, -46.0040436, 39.6612968, -65.3416367, 67.0886383
2: -13.3558750, 25.2196388, -24.8853512, 46.0735588, -59.4294357, 50.1049881
3: -10.2817898, 25.7319946, -19.6621456, 46.4303780, -56.7121658, 45.3941307
4: -16.0367928, 21.2913914, -30.0555954, 39.1768684, -55.2136574, 51.3469849

Time for backsubstitution: 2.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1139468, upper bound: 65.1453310
time: 0.55 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1139468, upper bound: 65.1460193
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -164.4706268, 234.0105133, -294.8692627, 410.0855408, -574.5561523, 528.8797607
1: -22.5604820, 18.6227417, -39.0235329, 33.4909554, -56.0514336, 57.6462708
2: -11.8040943, 22.1992359, -20.9863758, 39.1346283, -50.9387207, 43.1856117
3: -9.1159630, 22.5503044, -16.5540371, 39.2997131, -48.4156761, 39.1043396
4: -14.1580973, 18.7692890, -25.4112682, 33.2894211, -47.4475136, 44.1805573

Time for backsubstitution: 2.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0910309, upper bound: 65.0966241
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0964692, upper bound: 65.1131193
time: 0.58 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -172.4574738, 247.2798462, -347.4291077, 484.7604370, -657.2178345, 594.7089844
1: -23.8030243, 19.6571522, -46.0040436, 39.6612968, -63.4643173, 65.6611938
2: -12.4467812, 23.4034328, -24.8853512, 46.0735588, -58.5203362, 48.2887840
3: -9.6119366, 23.8297672, -19.6621456, 46.4303780, -56.0423126, 43.4919128
4: -14.9525290, 19.7782326, -30.0555954, 39.1768684, -54.1293945, 49.8338280

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 4, lower bound: -65.0910309, upper bound: 65.1081277
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -65.1194996, upper bound: 65.1245593
time: 0.59 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -184.2849579, 266.3168945, -358.1842651, 505.2524414, -689.5374146, 624.5011597
1: -25.6803474, 21.0845985, -48.1113739, 41.0600700, -66.7404175, 69.1959686
2: -13.3558750, 25.2196388, -25.7924500, 48.0750923, -61.4309692, 51.0120888
3: -10.2817898, 25.7319946, -20.2875805, 48.3676529, -58.6494446, 46.0195694
4: -16.0367928, 21.2913914, -31.1569042, 40.8619728, -56.8987656, 52.4482880

Time for backsubstitution: 2.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 38

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 20

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -184.2849579, 266.3168945, -348.6427612, 488.5848389, -672.8698120, 614.9595947
1: -25.6803474, 21.0845985, -46.4399033, 39.8489876, -65.5293350, 67.5245056
2: -13.3558750, 25.2196388, -25.0090580, 46.4942055, -59.8500748, 50.2286949
3: -10.2817898, 25.7319946, -19.7362804, 46.6899185, -56.9717102, 45.4682655
4: -16.0367928, 21.2913914, -30.2359753, 39.5413208, -55.5781136, 51.5273666

Time for backsubstitution: 2.67 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.29 + 416.75 = 421.05 seconds
