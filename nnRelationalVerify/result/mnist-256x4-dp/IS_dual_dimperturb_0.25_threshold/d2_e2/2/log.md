## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00149824981


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0000273, 0.0013243, 0.0000273, 0.0013243, -0.0011206, 0.0011206)
1: (0.9930748, 0.9958215, 0.9930748, 0.9958215, -0.0024096, 0.0024096)
2: (-0.0079923, -0.0067018, -0.0079923, -0.0067018, -0.0011062, 0.0011062)
3: (0.0027222, 0.0043448, 0.0027222, 0.0043448, -0.0014283, 0.0014283)
4: (0.0025461, 0.0046611, 0.0025461, 0.0046611, -0.0021150, 0.0021150)
5: (0.0034269, 0.0065004, 0.0034269, 0.0065004, -0.0025546, 0.0025546)
6: (-0.0021839, 0.0006582, -0.0021839, 0.0006582, -0.0026159, 0.0026159)
7: (-0.0080826, -0.0067680, -0.0080826, -0.0067680, -0.0010542, 0.0010542)
8: (0.0070569, 0.0081870, 0.0070569, 0.0081870, -0.0010781, 0.0010781)
9: (-0.0039772, -0.0020999, -0.0039772, -0.0020999, -0.0016124, 0.0016124)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.32 + 1.61 = 2.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0016339, upper bound: 0.0016339

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015574, upper bound: 0.0015935
time: 0.66 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015935, upper bound: 0.0015935
time: 0.63 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.42 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.42
Output dim: 1, lower bound: -0.0015574, upper bound: 0.0015935
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.42
Output dim: 1, lower bound: -0.0015935, upper bound: 0.0015935

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0000303, 0.0012853, 0.0000273, 0.0013243, -0.0010844, 0.0010770
1: 0.9931573, 0.9958150, 0.9930748, 0.9958215, -0.0023171, 0.0023239
2: -0.0079907, -0.0068363, -0.0079923, -0.0067018, -0.0011016, 0.0009691
3: 0.0027260, 0.0042961, 0.0027222, 0.0043448, -0.0013765, 0.0013737
4: 0.0025511, 0.0045976, 0.0025461, 0.0046611, -0.0020971, 0.0020515
5: 0.0034341, 0.0064081, 0.0034269, 0.0065004, -0.0024911, 0.0024511
6: -0.0020985, 0.0006516, -0.0021839, 0.0006582, -0.0025202, 0.0024975
7: -0.0080432, -0.0067710, -0.0080826, -0.0067680, -0.0010099, 0.0010367
8: 0.0072337, 0.0081866, 0.0070569, 0.0081870, -0.0009002, 0.0010769
9: -0.0039209, -0.0021043, -0.0039772, -0.0020999, -0.0015492, 0.0015623

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015574, upper bound: 0.0015574
time: 0.68 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015574, upper bound: 0.0015935
time: 0.72 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0000119, 0.0012929, 0.0000275, 0.0013195, -0.0011094, 0.0010986
1: 0.9931413, 0.9959044, 0.9930849, 0.9958210, -0.0023617, 0.0023737
2: -0.0080119, -0.0068101, -0.0079921, -0.0067182, -0.0011072, 0.0010028
3: 0.0026731, 0.0043056, 0.0027225, 0.0043389, -0.0014055, 0.0013998
4: 0.0024822, 0.0046099, 0.0025465, 0.0046533, -0.0021305, 0.0020634
5: 0.0033340, 0.0064261, 0.0034275, 0.0064891, -0.0025610, 0.0025050
6: -0.0021152, 0.0007441, -0.0021734, 0.0006577, -0.0025626, 0.0025415
7: -0.0080509, -0.0067282, -0.0080778, -0.0067682, -0.0010341, 0.0010700
8: 0.0071992, 0.0081922, 0.0070785, 0.0081869, -0.0009378, 0.0010610
9: -0.0039319, -0.0020431, -0.0039704, -0.0021002, -0.0015809, 0.0015994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015935, upper bound: 0.0015574
time: 0.70 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015935, upper bound: 0.0015935
time: 0.66 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.64 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 1, lower bound: -0.0015574, upper bound: 0.0015574
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 1, lower bound: -0.0015574, upper bound: 0.0015935
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 1, lower bound: -0.0015935, upper bound: 0.0015574
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.64
Output dim: 1, lower bound: -0.0015935, upper bound: 0.0015935

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0000303, 0.0012853, 0.0000303, 0.0012853, -0.0010408, 0.0010408
1: 0.9931573, 0.9958150, 0.9931573, 0.9958150, -0.0022314, 0.0022314
2: -0.0079907, -0.0068363, -0.0079907, -0.0068363, -0.0009646, 0.0009646
3: 0.0027260, 0.0042961, 0.0027260, 0.0042961, -0.0013219, 0.0013219
4: 0.0025511, 0.0045976, 0.0025511, 0.0045976, -0.0020259, 0.0020259
5: 0.0034341, 0.0064081, 0.0034341, 0.0064081, -0.0023877, 0.0023877
6: -0.0020985, 0.0006516, -0.0020985, 0.0006516, -0.0024018, 0.0024018
7: -0.0080432, -0.0067710, -0.0080432, -0.0067710, -0.0009924, 0.0009924
8: 0.0072337, 0.0081866, 0.0072337, 0.0081866, -0.0008990, 0.0008990
9: -0.0039209, -0.0021043, -0.0039209, -0.0021043, -0.0014991, 0.0014991

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015255, upper bound: 0.0014791
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015118, upper bound: 0.0015163
time: 0.57 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0000303, 0.0012853, -0.0000119, 0.0012929, -0.0010505, 0.0010704
1: 0.9931573, 0.9958150, 0.9931413, 0.9959044, -0.0022912, 0.0022521
2: -0.0079907, -0.0068363, -0.0080119, -0.0068101, -0.0010028, 0.0009852
3: 0.0027260, 0.0042961, 0.0026731, 0.0043056, -0.0013340, 0.0013568
4: 0.0025511, 0.0045976, 0.0024822, 0.0046099, -0.0020417, 0.0020670
5: 0.0034341, 0.0064081, 0.0033340, 0.0064261, -0.0024107, 0.0024688
6: -0.0020985, 0.0006516, -0.0021152, 0.0007441, -0.0024562, 0.0024232
7: -0.0080432, -0.0067710, -0.0080509, -0.0067282, -0.0010305, 0.0010023
8: 0.0072337, 0.0081866, 0.0071992, 0.0081922, -0.0009045, 0.0009374
9: -0.0039209, -0.0021043, -0.0039319, -0.0020431, -0.0015431, 0.0015132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014645, upper bound: 0.0015611
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015118, upper bound: 0.0015539
time: 0.62 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0000119, 0.0012929, 0.0000303, 0.0012853, -0.0010704, 0.0010505
1: 0.9931413, 0.9959044, 0.9931573, 0.9958150, -0.0022521, 0.0022912
2: -0.0080119, -0.0068101, -0.0079907, -0.0068363, -0.0009852, 0.0010028
3: 0.0026731, 0.0043056, 0.0027260, 0.0042961, -0.0013568, 0.0013340
4: 0.0024822, 0.0046099, 0.0025511, 0.0045976, -0.0020670, 0.0020417
5: 0.0033340, 0.0064261, 0.0034341, 0.0064081, -0.0024688, 0.0024107
6: -0.0021152, 0.0007441, -0.0020985, 0.0006516, -0.0024232, 0.0024562
7: -0.0080509, -0.0067282, -0.0080432, -0.0067710, -0.0010023, 0.0010305
8: 0.0071992, 0.0081922, 0.0072337, 0.0081866, -0.0009374, 0.0009045
9: -0.0039319, -0.0020431, -0.0039209, -0.0021043, -0.0015132, 0.0015431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015611, upper bound: 0.0014645
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015539, upper bound: 0.0015118
time: 0.67 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0000119, 0.0012929, -0.0000119, 0.0012929, -0.0010747, 0.0010747
1: 0.9931413, 0.9959044, 0.9931413, 0.9959044, -0.0023059, 0.0023059
2: -0.0080119, -0.0068101, -0.0080119, -0.0068101, -0.0009982, 0.0009982
3: 0.0026731, 0.0043056, 0.0026731, 0.0043056, -0.0013661, 0.0013661
4: 0.0024822, 0.0046099, 0.0024822, 0.0046099, -0.0021014, 0.0021014
5: 0.0033340, 0.0064261, 0.0033340, 0.0064261, -0.0024604, 0.0024604
6: -0.0021152, 0.0007441, -0.0021152, 0.0007441, -0.0024839, 0.0024839
7: -0.0080509, -0.0067282, -0.0080509, -0.0067282, -0.0010207, 0.0010207
8: 0.0071992, 0.0081922, 0.0071992, 0.0081922, -0.0009365, 0.0009365
9: -0.0039319, -0.0020431, -0.0039319, -0.0020431, -0.0015476, 0.0015476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015611, upper bound: 0.0014666
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015539, upper bound: 0.0015140
time: 0.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.65 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 1, lower bound: -0.0015255, upper bound: 0.0014791
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 1, lower bound: -0.0015118, upper bound: 0.0015163
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 1, lower bound: -0.0014645, upper bound: 0.0015611
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 1, lower bound: -0.0015118, upper bound: 0.0015539
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 1, lower bound: -0.0015611, upper bound: 0.0014645
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 1, lower bound: -0.0015539, upper bound: 0.0015118
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 1, lower bound: -0.0015611, upper bound: 0.0014666
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.65
Output dim: 1, lower bound: -0.0015539, upper bound: 0.0015140

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0000510, 0.0012832, 0.0000303, 0.0012853, -0.0010112, 0.0010335
1: 0.9931619, 0.9957713, 0.9931573, 0.9958150, -0.0022161, 0.0021687
2: -0.0079804, -0.0068437, -0.0079907, -0.0068363, -0.0009506, 0.0009566
3: 0.0027518, 0.0042934, 0.0027260, 0.0042961, -0.0012848, 0.0013128
4: 0.0025848, 0.0045940, 0.0025511, 0.0045976, -0.0019800, 0.0020141
5: 0.0034831, 0.0064030, 0.0034341, 0.0064081, -0.0023185, 0.0023705
6: -0.0020938, 0.0006062, -0.0020985, 0.0006516, -0.0023860, 0.0023350
7: -0.0080410, -0.0067920, -0.0080432, -0.0067710, -0.0009851, 0.0009634
8: 0.0072434, 0.0081838, 0.0072337, 0.0081866, -0.0008891, 0.0008953
9: -0.0039177, -0.0021342, -0.0039209, -0.0021043, -0.0014887, 0.0014565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0014791, upper bound: 0.0014791
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0014791, upper bound: 0.0014791
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0000452, 0.0013226, 0.0000326, 0.0012850, -0.0010634, 0.0010609
1: 0.9930783, 0.9957834, 0.9931580, 0.9958102, -0.0022737, 0.0022874
2: -0.0079833, -0.0067074, -0.0079896, -0.0068374, -0.0009642, 0.0010855
3: 0.0027447, 0.0043428, 0.0027288, 0.0042957, -0.0013561, 0.0013467
4: 0.0025754, 0.0046584, 0.0025548, 0.0045970, -0.0020216, 0.0020558
5: 0.0034695, 0.0064966, 0.0034395, 0.0064073, -0.0024216, 0.0024360
6: -0.0021803, 0.0006188, -0.0020978, 0.0006465, -0.0024453, 0.0024973
7: -0.0080810, -0.0067862, -0.0080428, -0.0067733, -0.0010134, 0.0009988
8: 0.0070643, 0.0081846, 0.0072351, 0.0081863, -0.0010669, 0.0008977
9: -0.0039749, -0.0021259, -0.0039204, -0.0021076, -0.0015283, 0.0015300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014791, upper bound: 0.0015163
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014791, upper bound: 0.0015163
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0000303, 0.0012853, 0.0000065, 0.0012909, -0.0010429, 0.0010424
1: 0.9931573, 0.9958150, 0.9931456, 0.9958655, -0.0022314, 0.0022359
2: -0.0079907, -0.0068363, -0.0080026, -0.0068171, -0.0009954, 0.0009721
3: 0.0027260, 0.0042961, 0.0026963, 0.0043031, -0.0013245, 0.0013215
4: 0.0025511, 0.0045976, 0.0025123, 0.0046066, -0.0020293, 0.0020236
5: 0.0034341, 0.0064081, 0.0033778, 0.0064213, -0.0023927, 0.0024038
6: -0.0020985, 0.0006516, -0.0021107, 0.0007036, -0.0023918, 0.0024065
7: -0.0080432, -0.0067710, -0.0080488, -0.0067469, -0.0010040, 0.0009946
8: 0.0072337, 0.0081866, 0.0072084, 0.0081897, -0.0009010, 0.0009281
9: -0.0039209, -0.0021043, -0.0039289, -0.0020699, -0.0015026, 0.0015022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014645, upper bound: 0.0015309
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014645, upper bound: 0.0015539
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0000326, 0.0012850, 0.0000057, 0.0013247, -0.0010661, 0.0010905
1: 0.9931580, 0.9958102, 0.9930739, 0.9958671, -0.0023411, 0.0022848
2: -0.0079896, -0.0068374, -0.0080030, -0.0067003, -0.0011029, 0.0009866
3: 0.0027288, 0.0042957, 0.0026953, 0.0043454, -0.0013533, 0.0013872
4: 0.0025548, 0.0045970, 0.0025110, 0.0046618, -0.0020643, 0.0020860
5: 0.0034395, 0.0064073, 0.0033759, 0.0065014, -0.0024485, 0.0025000
6: -0.0020978, 0.0006465, -0.0021848, 0.0007054, -0.0025406, 0.0024568
7: -0.0080428, -0.0067733, -0.0080831, -0.0067461, -0.0010390, 0.0010187
8: 0.0072351, 0.0081863, 0.0070549, 0.0081898, -0.0009037, 0.0010789
9: -0.0039204, -0.0021076, -0.0039779, -0.0020687, -0.0015703, 0.0015359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015118, upper bound: 0.0015309
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015118, upper bound: 0.0015539
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0000065, 0.0012909, 0.0000303, 0.0012853, -0.0010424, 0.0010429
1: 0.9931456, 0.9958655, 0.9931573, 0.9958150, -0.0022359, 0.0022314
2: -0.0080026, -0.0068171, -0.0079907, -0.0068363, -0.0009721, 0.0009954
3: 0.0026963, 0.0043031, 0.0027260, 0.0042961, -0.0013215, 0.0013245
4: 0.0025123, 0.0046066, 0.0025511, 0.0045976, -0.0020236, 0.0020293
5: 0.0033778, 0.0064213, 0.0034341, 0.0064081, -0.0024038, 0.0023927
6: -0.0021107, 0.0007036, -0.0020985, 0.0006516, -0.0024065, 0.0023918
7: -0.0080488, -0.0067469, -0.0080432, -0.0067710, -0.0009946, 0.0010040
8: 0.0072084, 0.0081897, 0.0072337, 0.0081866, -0.0009281, 0.0009010
9: -0.0039289, -0.0020699, -0.0039209, -0.0021043, -0.0015022, 0.0015026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0014645
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0014645
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0000057, 0.0013247, 0.0000326, 0.0012850, -0.0010905, 0.0010661
1: 0.9930739, 0.9958671, 0.9931580, 0.9958102, -0.0022848, 0.0023411
2: -0.0080030, -0.0067003, -0.0079896, -0.0068374, -0.0009866, 0.0011029
3: 0.0026953, 0.0043454, 0.0027288, 0.0042957, -0.0013872, 0.0013533
4: 0.0025110, 0.0046618, 0.0025548, 0.0045970, -0.0020860, 0.0020643
5: 0.0033759, 0.0065014, 0.0034395, 0.0064073, -0.0025000, 0.0024485
6: -0.0021848, 0.0007054, -0.0020978, 0.0006465, -0.0024568, 0.0025406
7: -0.0080831, -0.0067461, -0.0080428, -0.0067733, -0.0010187, 0.0010390
8: 0.0070549, 0.0081898, 0.0072351, 0.0081863, -0.0010789, 0.0009037
9: -0.0039779, -0.0020687, -0.0039204, -0.0021076, -0.0015359, 0.0015703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0015118
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0015118
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0000065, 0.0012909, -0.0000119, 0.0012929, -0.0010443, 0.0010677
1: 0.9931456, 0.9958655, 0.9931413, 0.9959044, -0.0022909, 0.0022412
2: -0.0080026, -0.0068171, -0.0080119, -0.0068101, -0.0009839, 0.0009907
3: 0.0026963, 0.0043031, 0.0026731, 0.0043056, -0.0013279, 0.0013573
4: 0.0025123, 0.0046066, 0.0024822, 0.0046099, -0.0020546, 0.0020898
5: 0.0033778, 0.0064213, 0.0033340, 0.0064261, -0.0023901, 0.0024436
6: -0.0021107, 0.0007036, -0.0021152, 0.0007441, -0.0024684, 0.0024160
7: -0.0080488, -0.0067469, -0.0080509, -0.0067282, -0.0010135, 0.0009907
8: 0.0072084, 0.0081897, 0.0071992, 0.0081922, -0.0009272, 0.0009327
9: -0.0039289, -0.0020699, -0.0039319, -0.0020431, -0.0015374, 0.0015037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0014666
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0014666
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000057, 0.0013247, -0.0000092, 0.0012926, -0.0010894, 0.0010942
1: 0.9930739, 0.9958671, 0.9931419, 0.9958988, -0.0023471, 0.0023425
2: -0.0080030, -0.0067003, -0.0080105, -0.0068112, -0.0009956, 0.0011011
3: 0.0026953, 0.0043454, 0.0026766, 0.0043052, -0.0013885, 0.0013905
4: 0.0025110, 0.0046618, 0.0024866, 0.0046094, -0.0020984, 0.0021304
5: 0.0033759, 0.0065014, 0.0033405, 0.0064253, -0.0024795, 0.0025073
6: -0.0021848, 0.0007054, -0.0021144, 0.0007381, -0.0025261, 0.0025539
7: -0.0080831, -0.0067461, -0.0080505, -0.0067310, -0.0010411, 0.0010219
8: 0.0070549, 0.0081898, 0.0072007, 0.0081918, -0.0010788, 0.0009346
9: -0.0039779, -0.0020687, -0.0039314, -0.0020471, -0.0015759, 0.0015675

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0015140
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0015140
time: 0.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.76 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0014791, upper bound: 0.0014791
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0014791, upper bound: 0.0014791
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0014791, upper bound: 0.0015163
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0014791, upper bound: 0.0015163
IS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0014645, upper bound: 0.0015309
IS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0014645, upper bound: 0.0015539
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0015118, upper bound: 0.0015309
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0015118, upper bound: 0.0015539
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0014645
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0014645
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0015118
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0015118
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0014666
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0014666
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0015140
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.76
Output dim: 1, lower bound: -0.0015309, upper bound: 0.0015140

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0000452, 0.0013226, 0.0000510, 0.0012832, -0.0010209, 0.0010363
1: 0.9930783, 0.9957834, 0.9931619, 0.9957713, -0.0022220, 0.0021962
2: -0.0079833, -0.0067074, -0.0079804, -0.0068437, -0.0009416, 0.0010737
3: 0.0027447, 0.0043428, 0.0027518, 0.0042934, -0.0013019, 0.0013163
4: 0.0025754, 0.0046584, 0.0025848, 0.0045940, -0.0020186, 0.0020210
5: 0.0034695, 0.0064966, 0.0034831, 0.0064030, -0.0023242, 0.0023781
6: -0.0021803, 0.0006188, -0.0020938, 0.0006062, -0.0023901, 0.0023889
7: -0.0080810, -0.0067862, -0.0080410, -0.0067920, -0.0009889, 0.0009597
8: 0.0070643, 0.0081846, 0.0072434, 0.0081838, -0.0010638, 0.0008851
9: -0.0039749, -0.0021259, -0.0039177, -0.0021342, -0.0014929, 0.0014689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011706, upper bound: 0.0014339
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014690, upper bound: 0.0015060
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000452, 0.0013226, 0.0000452, 0.0013226, -0.0010528, 0.0010528
1: 0.9930783, 0.9957834, 0.9930783, 0.9957834, -0.0022650, 0.0022650
2: -0.0079833, -0.0067074, -0.0079833, -0.0067074, -0.0010661, 0.0010661
3: 0.0027447, 0.0043428, 0.0027447, 0.0043428, -0.0013429, 0.0013429
4: 0.0025754, 0.0046584, 0.0025754, 0.0046584, -0.0020830, 0.0020830
5: 0.0034695, 0.0064966, 0.0034695, 0.0064966, -0.0023965, 0.0023965
6: -0.0021803, 0.0006188, -0.0021803, 0.0006188, -0.0024741, 0.0024741
7: -0.0080810, -0.0067862, -0.0080810, -0.0067862, -0.0009881, 0.0009881
8: 0.0070643, 0.0081846, 0.0070643, 0.0081846, -0.0010615, 0.0010615
9: -0.0039749, -0.0021259, -0.0039749, -0.0021259, -0.0015147, 0.0015147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011706, upper bound: 0.0014339
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014690, upper bound: 0.0015060
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0000510, 0.0012832, 0.0000065, 0.0012909, -0.0010133, 0.0010352
1: 0.9931619, 0.9957713, 0.9931456, 0.9958655, -0.0022161, 0.0021732
2: -0.0079804, -0.0068437, -0.0080026, -0.0068171, -0.0009814, 0.0009642
3: 0.0027518, 0.0042934, 0.0026963, 0.0043031, -0.0012875, 0.0013124
4: 0.0025848, 0.0045940, 0.0025123, 0.0046066, -0.0019835, 0.0020118
5: 0.0034831, 0.0064030, 0.0033778, 0.0064213, -0.0023236, 0.0023867
6: -0.0020938, 0.0006062, -0.0021107, 0.0007036, -0.0023760, 0.0023397
7: -0.0080410, -0.0067920, -0.0080488, -0.0067469, -0.0009967, 0.0009656
8: 0.0072434, 0.0081838, 0.0072084, 0.0081897, -0.0008911, 0.0009244
9: -0.0039177, -0.0021342, -0.0039289, -0.0020699, -0.0014922, 0.0014595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011719, upper bound: 0.0015168
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014540, upper bound: 0.0015502
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0000452, 0.0013226, 0.0000065, 0.0012909, -0.0010302, 0.0010676
1: 0.9930783, 0.9957834, 0.9931456, 0.9958655, -0.0022847, 0.0022160
2: -0.0079833, -0.0067074, -0.0080026, -0.0068171, -0.0009803, 0.0010953
3: 0.0027447, 0.0043428, 0.0026963, 0.0043031, -0.0013136, 0.0013529
4: 0.0025754, 0.0046584, 0.0025123, 0.0046066, -0.0020312, 0.0020646
5: 0.0034695, 0.0064966, 0.0033778, 0.0064213, -0.0023463, 0.0024634
6: -0.0021803, 0.0006188, -0.0021107, 0.0007036, -0.0024469, 0.0024094
7: -0.0080810, -0.0067862, -0.0080488, -0.0067469, -0.0010295, 0.0009692
8: 0.0070643, 0.0081846, 0.0072084, 0.0081897, -0.0010695, 0.0009241
9: -0.0039749, -0.0021259, -0.0039289, -0.0020699, -0.0015390, 0.0014825

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011719, upper bound: 0.0015168
time: 0.55 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014540, upper bound: 0.0015509
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0000510, 0.0012832, 0.0000057, 0.0013247, -0.0010416, 0.0010458
1: 0.9931619, 0.9957713, 0.9930739, 0.9958671, -0.0022446, 0.0022331
2: -0.0079804, -0.0068437, -0.0080030, -0.0067003, -0.0010911, 0.0009621
3: 0.0027518, 0.0042934, 0.0026953, 0.0043454, -0.0013228, 0.0013300
4: 0.0025848, 0.0045940, 0.0025110, 0.0046618, -0.0020295, 0.0020830
5: 0.0034831, 0.0064030, 0.0033759, 0.0065014, -0.0023905, 0.0023980
6: -0.0020938, 0.0006062, -0.0021848, 0.0007054, -0.0024269, 0.0024016
7: -0.0080410, -0.0067920, -0.0080831, -0.0067461, -0.0009954, 0.0009942
8: 0.0072434, 0.0081838, 0.0070549, 0.0081898, -0.0008906, 0.0010758
9: -0.0039177, -0.0021342, -0.0039779, -0.0020687, -0.0015060, 0.0015005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0013966, upper bound: 0.0012023
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014540, upper bound: 0.0015209
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000452, 0.0013226, 0.0000057, 0.0013247, -0.0010638, 0.0010800
1: 0.9930783, 0.9957834, 0.9930739, 0.9958671, -0.0023187, 0.0022882
2: -0.0079833, -0.0067074, -0.0080030, -0.0067003, -0.0010891, 0.0010885
3: 0.0027447, 0.0043428, 0.0026953, 0.0043454, -0.0013566, 0.0013740
4: 0.0025754, 0.0046584, 0.0025110, 0.0046618, -0.0020864, 0.0021474
5: 0.0034695, 0.0064966, 0.0033759, 0.0065014, -0.0024225, 0.0024750
6: -0.0021803, 0.0006188, -0.0021848, 0.0007054, -0.0025175, 0.0024981
7: -0.0080810, -0.0067862, -0.0080831, -0.0067461, -0.0010283, 0.0009992
8: 0.0070643, 0.0081846, 0.0070549, 0.0081898, -0.0010675, 0.0010746
9: -0.0039749, -0.0021259, -0.0039779, -0.0020687, -0.0015550, 0.0015305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011706, upper bound: 0.0014742
time: 0.62 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014540, upper bound: 0.0015437
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0000065, 0.0012909, 0.0000510, 0.0012832, -0.0010352, 0.0010133
1: 0.9931456, 0.9958655, 0.9931619, 0.9957713, -0.0021732, 0.0022161
2: -0.0080026, -0.0068171, -0.0079804, -0.0068437, -0.0009642, 0.0009814
3: 0.0026963, 0.0043031, 0.0027518, 0.0042934, -0.0013124, 0.0012875
4: 0.0025123, 0.0046066, 0.0025848, 0.0045940, -0.0020118, 0.0019835
5: 0.0033778, 0.0064213, 0.0034831, 0.0064030, -0.0023867, 0.0023236
6: -0.0021107, 0.0007036, -0.0020938, 0.0006062, -0.0023397, 0.0023760
7: -0.0080488, -0.0067469, -0.0080410, -0.0067920, -0.0009656, 0.0009967
8: 0.0072084, 0.0081897, 0.0072434, 0.0081838, -0.0009244, 0.0008911
9: -0.0039289, -0.0020699, -0.0039177, -0.0021342, -0.0014595, 0.0014922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015168, upper bound: 0.0011719
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015502, upper bound: 0.0014540
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0000065, 0.0012909, 0.0000452, 0.0013226, -0.0010676, 0.0010302
1: 0.9931456, 0.9958655, 0.9930783, 0.9957834, -0.0022160, 0.0022847
2: -0.0080026, -0.0068171, -0.0079833, -0.0067074, -0.0010953, 0.0009803
3: 0.0026963, 0.0043031, 0.0027447, 0.0043428, -0.0013529, 0.0013136
4: 0.0025123, 0.0046066, 0.0025754, 0.0046584, -0.0020646, 0.0020312
5: 0.0033778, 0.0064213, 0.0034695, 0.0064966, -0.0024634, 0.0023463
6: -0.0021107, 0.0007036, -0.0021803, 0.0006188, -0.0024094, 0.0024469
7: -0.0080488, -0.0067469, -0.0080810, -0.0067862, -0.0009692, 0.0010295
8: 0.0072084, 0.0081897, 0.0070643, 0.0081846, -0.0009241, 0.0010695
9: -0.0039289, -0.0020699, -0.0039749, -0.0021259, -0.0014825, 0.0015390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015168, upper bound: 0.0011719
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015502, upper bound: 0.0014540
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0000057, 0.0013247, 0.0000510, 0.0012832, -0.0010458, 0.0010416
1: 0.9930739, 0.9958671, 0.9931619, 0.9957713, -0.0022331, 0.0022446
2: -0.0080030, -0.0067003, -0.0079804, -0.0068437, -0.0009621, 0.0010911
3: 0.0026953, 0.0043454, 0.0027518, 0.0042934, -0.0013300, 0.0013228
4: 0.0025110, 0.0046618, 0.0025848, 0.0045940, -0.0020830, 0.0020295
5: 0.0033759, 0.0065014, 0.0034831, 0.0064030, -0.0023980, 0.0023905
6: -0.0021848, 0.0007054, -0.0020938, 0.0006062, -0.0024016, 0.0024269
7: -0.0080831, -0.0067461, -0.0080410, -0.0067920, -0.0009942, 0.0009954
8: 0.0070549, 0.0081898, 0.0072434, 0.0081838, -0.0010758, 0.0008906
9: -0.0039779, -0.0020687, -0.0039177, -0.0021342, -0.0015005, 0.0015060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012023, upper bound: 0.0014160
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015209, upper bound: 0.0015012
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000057, 0.0013247, 0.0000452, 0.0013226, -0.0010800, 0.0010638
1: 0.9930739, 0.9958671, 0.9930783, 0.9957834, -0.0022882, 0.0023187
2: -0.0080030, -0.0067003, -0.0079833, -0.0067074, -0.0010885, 0.0010891
3: 0.0026953, 0.0043454, 0.0027447, 0.0043428, -0.0013740, 0.0013566
4: 0.0025110, 0.0046618, 0.0025754, 0.0046584, -0.0021474, 0.0020864
5: 0.0033759, 0.0065014, 0.0034695, 0.0064966, -0.0024750, 0.0024225
6: -0.0021848, 0.0007054, -0.0021803, 0.0006188, -0.0024981, 0.0025175
7: -0.0080831, -0.0067461, -0.0080810, -0.0067862, -0.0009992, 0.0010283
8: 0.0070549, 0.0081898, 0.0070643, 0.0081846, -0.0010746, 0.0010675
9: -0.0039779, -0.0020687, -0.0039749, -0.0021259, -0.0015305, 0.0015550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0014592, upper bound: 0.0011706
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015209, upper bound: 0.0015012
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0000065, 0.0012909, 0.0000065, 0.0012909, -0.0010373, 0.0010373
1: 0.9931456, 0.9958655, 0.9931456, 0.9958655, -0.0022262, 0.0022262
2: -0.0080026, -0.0068171, -0.0080026, -0.0068171, -0.0009764, 0.0009764
3: 0.0026963, 0.0043031, 0.0026963, 0.0043031, -0.0013191, 0.0013191
4: 0.0025123, 0.0046066, 0.0025123, 0.0046066, -0.0020431, 0.0020431
5: 0.0033778, 0.0064213, 0.0033778, 0.0064213, -0.0023733, 0.0023733
6: -0.0021107, 0.0007036, -0.0021107, 0.0007036, -0.0024005, 0.0024005
7: -0.0080488, -0.0067469, -0.0080488, -0.0067469, -0.0009835, 0.0009835
8: 0.0072084, 0.0081897, 0.0072084, 0.0081897, -0.0009234, 0.0009234
9: -0.0039289, -0.0020699, -0.0039289, -0.0020699, -0.0014935, 0.0014935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015168, upper bound: 0.0011832
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015502, upper bound: 0.0014561
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0000065, 0.0012909, 0.0000057, 0.0013247, -0.0010688, 0.0010532
1: 0.9931456, 0.9958655, 0.9930739, 0.9958671, -0.0022650, 0.0022929
2: -0.0080026, -0.0068171, -0.0080030, -0.0067003, -0.0010889, 0.0009759
3: 0.0026963, 0.0043031, 0.0026953, 0.0043454, -0.0013585, 0.0013426
4: 0.0025123, 0.0046066, 0.0025110, 0.0046618, -0.0020944, 0.0020956
5: 0.0033778, 0.0064213, 0.0033759, 0.0065014, -0.0024480, 0.0023956
6: -0.0021107, 0.0007036, -0.0021848, 0.0007054, -0.0024609, 0.0024695
7: -0.0080488, -0.0067469, -0.0080831, -0.0067461, -0.0009879, 0.0010154
8: 0.0072084, 0.0081897, 0.0070549, 0.0081898, -0.0009233, 0.0010756
9: -0.0039289, -0.0020699, -0.0039779, -0.0020687, -0.0015152, 0.0015391

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015168, upper bound: 0.0011832
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015502, upper bound: 0.0014561
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0000057, 0.0013247, 0.0000065, 0.0012909, -0.0010532, 0.0010688
1: 0.9930739, 0.9958671, 0.9931456, 0.9958655, -0.0022929, 0.0022650
2: -0.0080030, -0.0067003, -0.0080026, -0.0068171, -0.0009759, 0.0010889
3: 0.0026953, 0.0043454, 0.0026963, 0.0043031, -0.0013426, 0.0013585
4: 0.0025110, 0.0046618, 0.0025123, 0.0046066, -0.0020956, 0.0020944
5: 0.0033759, 0.0065014, 0.0033778, 0.0064213, -0.0023956, 0.0024480
6: -0.0021848, 0.0007054, -0.0021107, 0.0007036, -0.0024695, 0.0024609
7: -0.0080831, -0.0067461, -0.0080488, -0.0067469, -0.0010154, 0.0009879
8: 0.0070549, 0.0081898, 0.0072084, 0.0081897, -0.0010756, 0.0009233
9: -0.0039779, -0.0020687, -0.0039289, -0.0020699, -0.0015391, 0.0015152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012031, upper bound: 0.0014230
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015209, upper bound: 0.0015033
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000057, 0.0013247, 0.0000057, 0.0013247, -0.0010784, 0.0010784
1: 0.9930739, 0.9958671, 0.9930739, 0.9958671, -0.0023192, 0.0023192
2: -0.0080030, -0.0067003, -0.0080030, -0.0067003, -0.0010825, 0.0010825
3: 0.0026953, 0.0043454, 0.0026953, 0.0043454, -0.0013748, 0.0013748
4: 0.0025110, 0.0046618, 0.0025110, 0.0046618, -0.0021508, 0.0021508
5: 0.0033759, 0.0065014, 0.0033759, 0.0065014, -0.0024535, 0.0024535
6: -0.0021848, 0.0007054, -0.0021848, 0.0007054, -0.0025298, 0.0025298
7: -0.0080831, -0.0067461, -0.0080831, -0.0067461, -0.0010108, 0.0010108
8: 0.0070549, 0.0081898, 0.0070549, 0.0081898, -0.0010736, 0.0010736
9: -0.0039779, -0.0020687, -0.0039779, -0.0020687, -0.0015516, 0.0015516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0014592, upper bound: 0.0011813
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0015209, upper bound: 0.0015033
time: 0.70 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.83 seconds
IS_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0011706, upper bound: 0.0014339
IS_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0014690, upper bound: 0.0015060
IS_A1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0011706, upper bound: 0.0014339
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0014690, upper bound: 0.0015060
IS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0011719, upper bound: 0.0015168
IS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0014540, upper bound: 0.0015502
IS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0011719, upper bound: 0.0015168
IS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0014540, upper bound: 0.0015509
IS_A1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0013966, upper bound: 0.0012023
IS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0014540, upper bound: 0.0015209
IS_A1_B2_B2_A2_B1, status: Status.VERIFIED, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0011706, upper bound: 0.0014742
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0014540, upper bound: 0.0015437
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0015168, upper bound: 0.0011719
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0015502, upper bound: 0.0014540
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0015168, upper bound: 0.0011719
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0015502, upper bound: 0.0014540
IS_A2_B1_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0012023, upper bound: 0.0014160
IS_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0015209, upper bound: 0.0015012
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0014592, upper bound: 0.0011706
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0015209, upper bound: 0.0015012
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0015168, upper bound: 0.0011832
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0015502, upper bound: 0.0014561
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0015168, upper bound: 0.0011832
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0015502, upper bound: 0.0014561
IS_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0012031, upper bound: 0.0014230
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0015209, upper bound: 0.0015033
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0014592, upper bound: 0.0011813
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.83
Output dim: 1, lower bound: -0.0015209, upper bound: 0.0015033

## BFS IS instance: IS_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0000452, 0.0013226, 0.0000546, 0.0012828, -0.0010207, 0.0010421
1: 0.9930783, 0.9957834, 0.9931626, 0.9957637, -0.0022351, 0.0021959
2: -0.0079833, -0.0067074, -0.0079786, -0.0068450, -0.0009404, 0.0010663
3: 0.0027447, 0.0043428, 0.0027564, 0.0042930, -0.0013017, 0.0013241
4: 0.0025754, 0.0046584, 0.0025906, 0.0045935, -0.0020181, 0.0020455
5: 0.0034695, 0.0064966, 0.0034916, 0.0064022, -0.0023238, 0.0023869
6: -0.0021803, 0.0006188, -0.0020930, 0.0005984, -0.0024082, 0.0023886
7: -0.0080810, -0.0067862, -0.0080406, -0.0067956, -0.0009903, 0.0009595
8: 0.0070643, 0.0081846, 0.0072450, 0.0081833, -0.0010618, 0.0008835
9: -0.0039749, -0.0021259, -0.0039172, -0.0021394, -0.0015008, 0.0014687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0014177, upper bound: 0.0012715
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0014177, upper bound: 0.0012715
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0000452, 0.0013226, 0.0000485, 0.0013223, -0.0010527, 0.0010601
1: 0.9930783, 0.9957834, 0.9930791, 0.9957764, -0.0022843, 0.0022646
2: -0.0079833, -0.0067074, -0.0079816, -0.0067087, -0.0010648, 0.0010585
3: 0.0027447, 0.0043428, 0.0027488, 0.0043423, -0.0013427, 0.0013546
4: 0.0025754, 0.0046584, 0.0025808, 0.0046578, -0.0020824, 0.0020777
5: 0.0034695, 0.0064966, 0.0034773, 0.0064956, -0.0023961, 0.0024067
6: -0.0021803, 0.0006188, -0.0021795, 0.0006116, -0.0024997, 0.0024737
7: -0.0080810, -0.0067862, -0.0080806, -0.0067895, -0.0009895, 0.0009880
8: 0.0070643, 0.0081846, 0.0070660, 0.0081841, -0.0010593, 0.0010598
9: -0.0039749, -0.0021259, -0.0039743, -0.0021307, -0.0015242, 0.0015144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0014177, upper bound: 0.0011706
time: 0.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014177, upper bound: 0.0015060
time: 0.54 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0000578, 0.0012829, 0.0000797, 0.0013096, -0.0010389, 0.0009902
1: 0.9931625, 0.9957568, 0.9931059, 0.9957104, -0.0021269, 0.0022278
2: -0.0079770, -0.0068448, -0.0079660, -0.0067525, -0.0010397, 0.0009314
3: 0.0027604, 0.0042930, 0.0027878, 0.0043265, -0.0013197, 0.0012604
4: 0.0025959, 0.0045936, 0.0026316, 0.0046371, -0.0020261, 0.0019619
5: 0.0034993, 0.0064023, 0.0035511, 0.0064656, -0.0023834, 0.0022624
6: -0.0020931, 0.0005913, -0.0021517, 0.0005433, -0.0023107, 0.0023960
7: -0.0080407, -0.0067989, -0.0080678, -0.0068211, -0.0009376, 0.0009907
8: 0.0072448, 0.0081829, 0.0071235, 0.0081800, -0.0008813, 0.0010075
9: -0.0039173, -0.0021441, -0.0039560, -0.0021758, -0.0014255, 0.0014965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012847, upper bound: 0.0013010
time: 0.57 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0012847, upper bound: 0.0015217
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0000510, 0.0012832, 0.0000097, 0.0012905, -0.0010131, 0.0010360
1: 0.9931619, 0.9957713, 0.9931464, 0.9958588, -0.0022202, 0.0021728
2: -0.0079804, -0.0068437, -0.0080011, -0.0068185, -0.0009801, 0.0009543
3: 0.0027518, 0.0042934, 0.0027002, 0.0043026, -0.0012872, 0.0013150
4: 0.0025848, 0.0045940, 0.0025174, 0.0046060, -0.0019831, 0.0020350
5: 0.0034831, 0.0064030, 0.0033852, 0.0064204, -0.0023231, 0.0023814
6: -0.0020938, 0.0006062, -0.0021098, 0.0006967, -0.0023881, 0.0023392
7: -0.0080410, -0.0067920, -0.0080484, -0.0067501, -0.0009922, 0.0009654
8: 0.0072434, 0.0081838, 0.0072102, 0.0081893, -0.0008884, 0.0009226
9: -0.0039177, -0.0021342, -0.0039284, -0.0020744, -0.0014926, 0.0014592

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0014874, upper bound: 0.0013024
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014874, upper bound: 0.0015566
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0000522, 0.0013224, 0.0000797, 0.0013096, -0.0010559, 0.0010225
1: 0.9930789, 0.9957686, 0.9931059, 0.9957104, -0.0021953, 0.0022704
2: -0.0079798, -0.0067084, -0.0079660, -0.0067525, -0.0010388, 0.0010625
3: 0.0027534, 0.0043424, 0.0027878, 0.0043265, -0.0013457, 0.0013009
4: 0.0025868, 0.0046579, 0.0026316, 0.0046371, -0.0020503, 0.0020263
5: 0.0034861, 0.0064959, 0.0035511, 0.0064656, -0.0024064, 0.0023390
6: -0.0021797, 0.0006035, -0.0021517, 0.0005433, -0.0023815, 0.0024656
7: -0.0080807, -0.0067933, -0.0080678, -0.0068211, -0.0009703, 0.0009942
8: 0.0070656, 0.0081836, 0.0071235, 0.0081800, -0.0010597, 0.0010073
9: -0.0039745, -0.0021360, -0.0039560, -0.0021758, -0.0014723, 0.0015194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011625, upper bound: 0.0012804
time: 0.53 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011625, upper bound: 0.0015168
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000452, 0.0013226, 0.0000097, 0.0012905, -0.0010300, 0.0010685
1: 0.9930783, 0.9957834, 0.9931464, 0.9958588, -0.0022890, 0.0022156
2: -0.0079833, -0.0067074, -0.0080011, -0.0068185, -0.0009791, 0.0010856
3: 0.0027447, 0.0043428, 0.0027002, 0.0043026, -0.0013134, 0.0013557
4: 0.0025754, 0.0046584, 0.0025174, 0.0046060, -0.0020306, 0.0020880
5: 0.0034695, 0.0064966, 0.0033852, 0.0064204, -0.0023459, 0.0024584
6: -0.0021803, 0.0006188, -0.0021098, 0.0006967, -0.0024594, 0.0024090
7: -0.0080810, -0.0067862, -0.0080484, -0.0067501, -0.0010252, 0.0009690
8: 0.0070643, 0.0081846, 0.0072102, 0.0081893, -0.0010670, 0.0009224
9: -0.0039749, -0.0021259, -0.0039284, -0.0020744, -0.0015396, 0.0014822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0013966, upper bound: 0.0012826
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0013966, upper bound: 0.0012826
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0000546, 0.0012828, 0.0000057, 0.0013247, -0.0010414, 0.0010457
1: 0.9931626, 0.9957637, 0.9930739, 0.9958671, -0.0022443, 0.0022336
2: -0.0079786, -0.0068450, -0.0080030, -0.0067003, -0.0010825, 0.0009610
3: 0.0027564, 0.0042930, 0.0026953, 0.0043454, -0.0013232, 0.0013298
4: 0.0025906, 0.0045935, 0.0025110, 0.0046618, -0.0020444, 0.0020825
5: 0.0034916, 0.0064022, 0.0033759, 0.0065014, -0.0023852, 0.0023977
6: -0.0020930, 0.0005984, -0.0021848, 0.0007054, -0.0024266, 0.0024066
7: -0.0080406, -0.0067956, -0.0080831, -0.0067461, -0.0009953, 0.0009896
8: 0.0072450, 0.0081833, 0.0070549, 0.0081898, -0.0008890, 0.0010733
9: -0.0039172, -0.0021394, -0.0039779, -0.0020687, -0.0015058, 0.0014998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012715, upper bound: 0.0014592
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0012715, upper bound: 0.0015209
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000452, 0.0013226, 0.0000090, 0.0013243, -0.0010636, 0.0010825
1: 0.9930783, 0.9957834, 0.9930748, 0.9958603, -0.0023273, 0.0022877
2: -0.0079833, -0.0067074, -0.0080014, -0.0067017, -0.0010878, 0.0010791
3: 0.0027447, 0.0043428, 0.0026993, 0.0043449, -0.0013563, 0.0013797
4: 0.0025754, 0.0046584, 0.0025163, 0.0046611, -0.0020857, 0.0021421
5: 0.0034695, 0.0064966, 0.0033836, 0.0065005, -0.0024220, 0.0024713
6: -0.0021803, 0.0006188, -0.0021839, 0.0006983, -0.0025400, 0.0024977
7: -0.0080810, -0.0067862, -0.0080827, -0.0067494, -0.0010239, 0.0009990
8: 0.0070643, 0.0081846, 0.0070568, 0.0081894, -0.0010648, 0.0010727
9: -0.0039749, -0.0021259, -0.0039773, -0.0020734, -0.0015578, 0.0015302

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0013966, upper bound: 0.0012023
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0013966, upper bound: 0.0012023
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0000797, 0.0013096, 0.0000578, 0.0012829, -0.0009902, 0.0010389
1: 0.9931059, 0.9957104, 0.9931625, 0.9957568, -0.0022278, 0.0021269
2: -0.0079660, -0.0067525, -0.0079770, -0.0068448, -0.0009314, 0.0010397
3: 0.0027878, 0.0043265, 0.0027604, 0.0042930, -0.0012604, 0.0013197
4: 0.0026316, 0.0046371, 0.0025959, 0.0045936, -0.0019619, 0.0020261
5: 0.0035511, 0.0064656, 0.0034993, 0.0064023, -0.0022624, 0.0023834
6: -0.0021517, 0.0005433, -0.0020931, 0.0005913, -0.0023960, 0.0023107
7: -0.0080678, -0.0068211, -0.0080407, -0.0067989, -0.0009907, 0.0009376
8: 0.0071235, 0.0081800, 0.0072448, 0.0081829, -0.0010075, 0.0008813
9: -0.0039560, -0.0021758, -0.0039173, -0.0021441, -0.0014965, 0.0014255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0013010, upper bound: 0.0012847
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0013010, upper bound: 0.0012868
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0000097, 0.0012905, 0.0000510, 0.0012832, -0.0010360, 0.0010131
1: 0.9931464, 0.9958588, 0.9931619, 0.9957713, -0.0021728, 0.0022202
2: -0.0080011, -0.0068185, -0.0079804, -0.0068437, -0.0009543, 0.0009801
3: 0.0027002, 0.0043026, 0.0027518, 0.0042934, -0.0013150, 0.0012872
4: 0.0025174, 0.0046060, 0.0025848, 0.0045940, -0.0020350, 0.0019831
5: 0.0033852, 0.0064204, 0.0034831, 0.0064030, -0.0023814, 0.0023231
6: -0.0021098, 0.0006967, -0.0020938, 0.0006062, -0.0023392, 0.0023881
7: -0.0080484, -0.0067501, -0.0080410, -0.0067920, -0.0009654, 0.0009922
8: 0.0072102, 0.0081893, 0.0072434, 0.0081838, -0.0009226, 0.0008884
9: -0.0039284, -0.0020744, -0.0039177, -0.0021342, -0.0014592, 0.0014926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0013024, upper bound: 0.0014874
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0013024, upper bound: 0.0015208
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0000797, 0.0013096, 0.0000522, 0.0013224, -0.0010225, 0.0010559
1: 0.9931059, 0.9957104, 0.9930789, 0.9957686, -0.0022704, 0.0021953
2: -0.0079660, -0.0067525, -0.0079798, -0.0067084, -0.0010625, 0.0010388
3: 0.0027878, 0.0043265, 0.0027534, 0.0043424, -0.0013009, 0.0013457
4: 0.0026316, 0.0046371, 0.0025868, 0.0046579, -0.0020263, 0.0020503
5: 0.0035511, 0.0064656, 0.0034861, 0.0064959, -0.0023390, 0.0024064
6: -0.0021517, 0.0005433, -0.0021797, 0.0006035, -0.0024656, 0.0023815
7: -0.0080678, -0.0068211, -0.0080807, -0.0067933, -0.0009942, 0.0009703
8: 0.0071235, 0.0081800, 0.0070656, 0.0081836, -0.0010073, 0.0010597
9: -0.0039560, -0.0021758, -0.0039745, -0.0021360, -0.0015194, 0.0014723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012804, upper bound: 0.0011625
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012804, upper bound: 0.0011719
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000097, 0.0012905, 0.0000452, 0.0013226, -0.0010685, 0.0010300
1: 0.9931464, 0.9958588, 0.9930783, 0.9957834, -0.0022156, 0.0022890
2: -0.0080011, -0.0068185, -0.0079833, -0.0067074, -0.0010856, 0.0009791
3: 0.0027002, 0.0043026, 0.0027447, 0.0043428, -0.0013557, 0.0013134
4: 0.0025174, 0.0046060, 0.0025754, 0.0046584, -0.0020880, 0.0020306
5: 0.0033852, 0.0064204, 0.0034695, 0.0064966, -0.0024584, 0.0023459
6: -0.0021098, 0.0006967, -0.0021803, 0.0006188, -0.0024090, 0.0024594
7: -0.0080484, -0.0067501, -0.0080810, -0.0067862, -0.0009690, 0.0010252
8: 0.0072102, 0.0081893, 0.0070643, 0.0081846, -0.0009224, 0.0010670
9: -0.0039284, -0.0020744, -0.0039749, -0.0021259, -0.0014822, 0.0015396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012826, upper bound: 0.0013966
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012826, upper bound: 0.0014540
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0000057, 0.0013247, 0.0000546, 0.0012828, -0.0010457, 0.0010414
1: 0.9930739, 0.9958671, 0.9931626, 0.9957637, -0.0022336, 0.0022443
2: -0.0080030, -0.0067003, -0.0079786, -0.0068450, -0.0009610, 0.0010825
3: 0.0026953, 0.0043454, 0.0027564, 0.0042930, -0.0013298, 0.0013232
4: 0.0025110, 0.0046618, 0.0025906, 0.0045935, -0.0020825, 0.0020444
5: 0.0033759, 0.0065014, 0.0034916, 0.0064022, -0.0023977, 0.0023852
6: -0.0021848, 0.0007054, -0.0020930, 0.0005984, -0.0024066, 0.0024266
7: -0.0080831, -0.0067461, -0.0080406, -0.0067956, -0.0009896, 0.0009953
8: 0.0070549, 0.0081898, 0.0072450, 0.0081833, -0.0010733, 0.0008890
9: -0.0039779, -0.0020687, -0.0039172, -0.0021394, -0.0014998, 0.0015058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0014592, upper bound: 0.0012715
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014592, upper bound: 0.0015153
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000090, 0.0013243, 0.0000452, 0.0013226, -0.0010825, 0.0010636
1: 0.9930748, 0.9958603, 0.9930783, 0.9957834, -0.0022877, 0.0023273
2: -0.0080014, -0.0067017, -0.0079833, -0.0067074, -0.0010791, 0.0010878
3: 0.0026993, 0.0043449, 0.0027447, 0.0043428, -0.0013797, 0.0013563
4: 0.0025163, 0.0046611, 0.0025754, 0.0046584, -0.0021421, 0.0020857
5: 0.0033836, 0.0065005, 0.0034695, 0.0064966, -0.0024713, 0.0024220
6: -0.0021839, 0.0006983, -0.0021803, 0.0006188, -0.0024977, 0.0025400
7: -0.0080827, -0.0067494, -0.0080810, -0.0067862, -0.0009990, 0.0010239
8: 0.0070568, 0.0081894, 0.0070643, 0.0081846, -0.0010727, 0.0010648
9: -0.0039773, -0.0020734, -0.0039749, -0.0021259, -0.0015302, 0.0015578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012023, upper bound: 0.0014160
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012023, upper bound: 0.0014160
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0000797, 0.0013096, 0.0000136, 0.0012906, -0.0010004, 0.0010723
1: 0.9931059, 0.9957104, 0.9931462, 0.9958502, -0.0023007, 0.0021521
2: -0.0079660, -0.0067525, -0.0079991, -0.0068182, -0.0009471, 0.0010380
3: 0.0027878, 0.0043265, 0.0027051, 0.0043027, -0.0012760, 0.0013630
4: 0.0026316, 0.0046371, 0.0025239, 0.0046061, -0.0019745, 0.0021008
5: 0.0035511, 0.0064656, 0.0033946, 0.0064206, -0.0022735, 0.0024556
6: -0.0021517, 0.0005433, -0.0021100, 0.0006881, -0.0024771, 0.0023446
7: -0.0080678, -0.0068211, -0.0080485, -0.0067541, -0.0010184, 0.0009342
8: 0.0071235, 0.0081800, 0.0072098, 0.0081888, -0.0010071, 0.0009145
9: -0.0039560, -0.0021758, -0.0039285, -0.0020802, -0.0015442, 0.0014393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0013010, upper bound: 0.0012887
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0013010, upper bound: 0.0012899
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0000097, 0.0012905, 0.0000065, 0.0012909, -0.0010457, 0.0010371
1: 0.9931464, 0.9958588, 0.9931456, 0.9958655, -0.0022259, 0.0022443
2: -0.0080011, -0.0068185, -0.0080026, -0.0068171, -0.0009685, 0.0009751
3: 0.0027002, 0.0043026, 0.0026963, 0.0043031, -0.0013297, 0.0013189
4: 0.0025174, 0.0046060, 0.0025123, 0.0046066, -0.0020685, 0.0020428
5: 0.0033852, 0.0064204, 0.0033778, 0.0064213, -0.0023900, 0.0023729
6: -0.0021098, 0.0006967, -0.0021107, 0.0007036, -0.0024002, 0.0024222
7: -0.0080484, -0.0067501, -0.0080488, -0.0067469, -0.0009834, 0.0009880
8: 0.0072102, 0.0081893, 0.0072084, 0.0081897, -0.0009217, 0.0009214
9: -0.0039284, -0.0020744, -0.0039289, -0.0020699, -0.0014933, 0.0015055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0013024, upper bound: 0.0014874
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0013024, upper bound: 0.0015208
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0000797, 0.0013096, 0.0000128, 0.0013244, -0.0010319, 0.0010884
1: 0.9931059, 0.9957104, 0.9930746, 0.9958521, -0.0023396, 0.0022188
2: -0.0079660, -0.0067525, -0.0079995, -0.0067014, -0.0010596, 0.0010378
3: 0.0027878, 0.0043265, 0.0027041, 0.0043450, -0.0013154, 0.0013867
4: 0.0026316, 0.0046371, 0.0025226, 0.0046613, -0.0020297, 0.0021146
5: 0.0035511, 0.0064656, 0.0033927, 0.0065007, -0.0023481, 0.0024784
6: -0.0021517, 0.0005433, -0.0021841, 0.0006898, -0.0025380, 0.0024135
7: -0.0080678, -0.0068211, -0.0080828, -0.0067533, -0.0010226, 0.0009661
8: 0.0071235, 0.0081800, 0.0070564, 0.0081889, -0.0010071, 0.0010666
9: -0.0039560, -0.0021758, -0.0039774, -0.0020790, -0.0015662, 0.0014848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012813, upper bound: 0.0011754
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012813, upper bound: 0.0011832
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000097, 0.0012905, 0.0000057, 0.0013247, -0.0010773, 0.0010530
1: 0.9931464, 0.9958588, 0.9930739, 0.9958671, -0.0022647, 0.0023112
2: -0.0080011, -0.0068185, -0.0080030, -0.0067003, -0.0010822, 0.0009747
3: 0.0027002, 0.0043026, 0.0026953, 0.0043454, -0.0013692, 0.0013424
4: 0.0025174, 0.0046060, 0.0025110, 0.0046618, -0.0021200, 0.0020950
5: 0.0033852, 0.0064204, 0.0033759, 0.0065014, -0.0024648, 0.0023953
6: -0.0021098, 0.0006967, -0.0021848, 0.0007054, -0.0024605, 0.0024915
7: -0.0080484, -0.0067501, -0.0080831, -0.0067461, -0.0009877, 0.0010201
8: 0.0072102, 0.0081893, 0.0070549, 0.0081898, -0.0009215, 0.0010734
9: -0.0039284, -0.0020744, -0.0039779, -0.0020687, -0.0015150, 0.0015513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012835, upper bound: 0.0014028
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012835, upper bound: 0.0014561
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0000057, 0.0013247, 0.0000097, 0.0012905, -0.0010530, 0.0010773
1: 0.9930739, 0.9958671, 0.9931464, 0.9958588, -0.0023112, 0.0022647
2: -0.0080030, -0.0067003, -0.0080011, -0.0068185, -0.0009747, 0.0010822
3: 0.0026953, 0.0043454, 0.0027002, 0.0043026, -0.0013424, 0.0013692
4: 0.0025110, 0.0046618, 0.0025174, 0.0046060, -0.0020950, 0.0021200
5: 0.0033759, 0.0065014, 0.0033852, 0.0064204, -0.0023953, 0.0024648
6: -0.0021848, 0.0007054, -0.0021098, 0.0006967, -0.0024915, 0.0024605
7: -0.0080831, -0.0067461, -0.0080484, -0.0067501, -0.0010201, 0.0009877
8: 0.0070549, 0.0081898, 0.0072102, 0.0081893, -0.0010734, 0.0009215
9: -0.0039779, -0.0020687, -0.0039284, -0.0020744, -0.0015513, 0.0015150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0014592, upper bound: 0.0012748
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014592, upper bound: 0.0015168
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000090, 0.0013243, 0.0000057, 0.0013247, -0.0010856, 0.0010783
1: 0.9930748, 0.9958603, 0.9930739, 0.9958671, -0.0023188, 0.0023387
2: -0.0080014, -0.0067017, -0.0080030, -0.0067003, -0.0010754, 0.0010812
3: 0.0026993, 0.0043449, 0.0026953, 0.0043454, -0.0013869, 0.0013746
4: 0.0025163, 0.0046611, 0.0025110, 0.0046618, -0.0021455, 0.0021501
5: 0.0033836, 0.0065005, 0.0033759, 0.0065014, -0.0024684, 0.0024531
6: -0.0021839, 0.0006983, -0.0021848, 0.0007054, -0.0025295, 0.0025576
7: -0.0080827, -0.0067494, -0.0080831, -0.0067461, -0.0010106, 0.0010150
8: 0.0070568, 0.0081894, 0.0070549, 0.0081898, -0.0010717, 0.0010714
9: -0.0039773, -0.0020734, -0.0039779, -0.0020687, -0.0015514, 0.0015615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012031, upper bound: 0.0014230
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0012031, upper bound: 0.0015033
time: 0.63 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.63 seconds
IS_A1_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0014177, upper bound: 0.0012715
IS_A1_B1_A2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0014177, upper bound: 0.0012715
IS_A1_B1_A2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0014177, upper bound: 0.0011706
IS_A1_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0014177, upper bound: 0.0015060
IS_A1_B2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012847, upper bound: 0.0013010
IS_A1_B2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012847, upper bound: 0.0015217
IS_A1_B2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0014874, upper bound: 0.0013024
IS_A1_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0014874, upper bound: 0.0015566
IS_A1_B2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0011625, upper bound: 0.0012804
IS_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0011625, upper bound: 0.0015168
IS_A1_B2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0013966, upper bound: 0.0012826
IS_A1_B2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0013966, upper bound: 0.0012826
IS_A1_B2_B2_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012715, upper bound: 0.0014592
IS_A1_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012715, upper bound: 0.0015209
IS_A1_B2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0013966, upper bound: 0.0012023
IS_A1_B2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0013966, upper bound: 0.0012023
IS_A2_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0013010, upper bound: 0.0012847
IS_A2_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0013010, upper bound: 0.0012868
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0013024, upper bound: 0.0014874
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0013024, upper bound: 0.0015208
IS_A2_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012804, upper bound: 0.0011625
IS_A2_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012804, upper bound: 0.0011719
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012826, upper bound: 0.0013966
IS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012826, upper bound: 0.0014540
IS_A2_B1_A2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0014592, upper bound: 0.0012715
IS_A2_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0014592, upper bound: 0.0015153
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012023, upper bound: 0.0014160
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012023, upper bound: 0.0014160
IS_A2_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0013010, upper bound: 0.0012887
IS_A2_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0013010, upper bound: 0.0012899
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0013024, upper bound: 0.0014874
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0013024, upper bound: 0.0015208
IS_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012813, upper bound: 0.0011754
IS_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012813, upper bound: 0.0011832
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012835, upper bound: 0.0014028
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012835, upper bound: 0.0014561
IS_A2_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0014592, upper bound: 0.0012748
IS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0014592, upper bound: 0.0015168
IS_A2_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012031, upper bound: 0.0014230
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.63
Output dim: 1, lower bound: -0.0012031, upper bound: 0.0015033

## BFS IS instance: IS_A1_B1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000485, 0.0013223, 0.0000485, 0.0013223, -0.0010599, 0.0010599
1: 0.9930791, 0.9957764, 0.9930791, 0.9957764, -0.0022839, 0.0022839
2: -0.0079816, -0.0067087, -0.0079816, -0.0067087, -0.0010573, 0.0010573
3: 0.0027488, 0.0043423, 0.0027488, 0.0043423, -0.0013544, 0.0013544
4: 0.0025808, 0.0046578, 0.0025808, 0.0046578, -0.0020770, 0.0020770
5: 0.0034773, 0.0064956, 0.0034773, 0.0064956, -0.0024063, 0.0024063
6: -0.0021795, 0.0006116, -0.0021795, 0.0006116, -0.0024994, 0.0024994
7: -0.0080806, -0.0067895, -0.0080806, -0.0067895, -0.0009893, 0.0009893
8: 0.0070660, 0.0081841, 0.0070660, 0.0081841, -0.0010576, 0.0010576
9: -0.0039743, -0.0021307, -0.0039743, -0.0021307, -0.0015239, 0.0015239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 137

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011459, upper bound: 0.0014829
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011291, upper bound: 0.0014898
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0000546, 0.0012828, 0.0000797, 0.0013096, -0.0010385, 0.0009901
1: 0.9931626, 0.9957637, 0.9931059, 0.9957104, -0.0021268, 0.0022261
2: -0.0079786, -0.0068450, -0.0079660, -0.0067525, -0.0010409, 0.0009313
3: 0.0027564, 0.0042930, 0.0027878, 0.0043265, -0.0013187, 0.0012604
4: 0.0025906, 0.0045935, 0.0026316, 0.0046371, -0.0020215, 0.0019619
5: 0.0034916, 0.0064022, 0.0035511, 0.0064656, -0.0023845, 0.0022623
6: -0.0020930, 0.0005984, -0.0021517, 0.0005433, -0.0023106, 0.0023937
7: -0.0080406, -0.0067956, -0.0080678, -0.0068211, -0.0009375, 0.0009920
8: 0.0072450, 0.0081833, 0.0071235, 0.0081800, -0.0008811, 0.0010078
9: -0.0039172, -0.0021394, -0.0039560, -0.0021758, -0.0014254, 0.0014960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 137

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0012720, upper bound: 0.0015053
time: 0.58 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0012653, upper bound: 0.0015113
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000546, 0.0012828, 0.0000097, 0.0012905, -0.0010126, 0.0010359
1: 0.9931626, 0.9957637, 0.9931464, 0.9958588, -0.0022199, 0.0021726
2: -0.0079786, -0.0068450, -0.0080011, -0.0068185, -0.0009704, 0.0009532
3: 0.0027564, 0.0042930, 0.0027002, 0.0043026, -0.0012872, 0.0013148
4: 0.0025906, 0.0045935, 0.0025174, 0.0046060, -0.0019974, 0.0020348
5: 0.0034916, 0.0064022, 0.0033852, 0.0064204, -0.0023170, 0.0023810
6: -0.0020930, 0.0005984, -0.0021098, 0.0006967, -0.0023878, 0.0023435
7: -0.0080406, -0.0067956, -0.0080484, -0.0067501, -0.0009921, 0.0009604
8: 0.0072450, 0.0081833, 0.0072102, 0.0081893, -0.0008868, 0.0009203
9: -0.0039172, -0.0021394, -0.0039284, -0.0020744, -0.0014924, 0.0014582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 137

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0012608, upper bound: 0.0015472
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0012653, upper bound: 0.0015472
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0000485, 0.0013223, 0.0000797, 0.0013096, -0.0010556, 0.0010225
1: 0.9930791, 0.9957764, 0.9931059, 0.9957104, -0.0021952, 0.0022688
2: -0.0079816, -0.0067087, -0.0079660, -0.0067525, -0.0010396, 0.0010622
3: 0.0027488, 0.0043423, 0.0027878, 0.0043265, -0.0013448, 0.0013008
4: 0.0025808, 0.0046578, 0.0026316, 0.0046371, -0.0020564, 0.0020262
5: 0.0034773, 0.0064956, 0.0035511, 0.0064656, -0.0024069, 0.0023389
6: -0.0021795, 0.0006116, -0.0021517, 0.0005433, -0.0023814, 0.0024633
7: -0.0080806, -0.0067895, -0.0080678, -0.0068211, -0.0009703, 0.0009957
8: 0.0070660, 0.0081841, 0.0071235, 0.0081800, -0.0010593, 0.0010075
9: -0.0039743, -0.0021307, -0.0039560, -0.0021758, -0.0014722, 0.0015193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 137

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011267, upper bound: 0.0015062
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011305, upper bound: 0.0015062
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000546, 0.0012828, 0.0000090, 0.0013243, -0.0010412, 0.0010485
1: 0.9931626, 0.9957637, 0.9930748, 0.9958603, -0.0022511, 0.0022331
2: -0.0079786, -0.0068450, -0.0080014, -0.0067017, -0.0010811, 0.0009511
3: 0.0027564, 0.0042930, 0.0026993, 0.0043449, -0.0013229, 0.0013340
4: 0.0025906, 0.0045935, 0.0025163, 0.0046611, -0.0020440, 0.0020772
5: 0.0034916, 0.0064022, 0.0033836, 0.0065005, -0.0023847, 0.0023968
6: -0.0020930, 0.0005984, -0.0021839, 0.0006983, -0.0024410, 0.0024061
7: -0.0080406, -0.0067956, -0.0080827, -0.0067494, -0.0009931, 0.0009894
8: 0.0072450, 0.0081833, 0.0070568, 0.0081894, -0.0008863, 0.0010714
9: -0.0039172, -0.0021394, -0.0039773, -0.0020734, -0.0015094, 0.0014995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 137

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0012581, upper bound: 0.0015011
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012487, upper bound: 0.0014425
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000097, 0.0012905, 0.0000546, 0.0012828, -0.0010359, 0.0010126
1: 0.9931464, 0.9958588, 0.9931626, 0.9957637, -0.0021726, 0.0022199
2: -0.0080011, -0.0068185, -0.0079786, -0.0068450, -0.0009532, 0.0009704
3: 0.0027002, 0.0043026, 0.0027564, 0.0042930, -0.0013148, 0.0012872
4: 0.0025174, 0.0046060, 0.0025906, 0.0045935, -0.0020348, 0.0019974
5: 0.0033852, 0.0064204, 0.0034916, 0.0064022, -0.0023810, 0.0023170
6: -0.0021098, 0.0006967, -0.0020930, 0.0005984, -0.0023435, 0.0023878
7: -0.0080484, -0.0067501, -0.0080406, -0.0067956, -0.0009604, 0.0009921
8: 0.0072102, 0.0081893, 0.0072450, 0.0081833, -0.0009203, 0.0008868
9: -0.0039284, -0.0020744, -0.0039172, -0.0021394, -0.0014582, 0.0014924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 137

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012900, upper bound: 0.0014973
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0012891, upper bound: 0.0015113
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000090, 0.0013243, 0.0000546, 0.0012828, -0.0010485, 0.0010412
1: 0.9930748, 0.9958603, 0.9931626, 0.9957637, -0.0022331, 0.0022511
2: -0.0080014, -0.0067017, -0.0079786, -0.0068450, -0.0009511, 0.0010811
3: 0.0026993, 0.0043449, 0.0027564, 0.0042930, -0.0013340, 0.0013229
4: 0.0025163, 0.0046611, 0.0025906, 0.0045935, -0.0020772, 0.0020440
5: 0.0033836, 0.0065005, 0.0034916, 0.0064022, -0.0023968, 0.0023847
6: -0.0021839, 0.0006983, -0.0020930, 0.0005984, -0.0024061, 0.0024410
7: -0.0080827, -0.0067494, -0.0080406, -0.0067956, -0.0009893, 0.0009931
8: 0.0070568, 0.0081894, 0.0072450, 0.0081833, -0.0010714, 0.0008863
9: -0.0039773, -0.0020734, -0.0039172, -0.0021394, -0.0014995, 0.0015094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 137

## Relational analysis of IS_A2_B1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011695, upper bound: 0.0015038
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011755, upper bound: 0.0012445
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000097, 0.0012905, 0.0000097, 0.0012905, -0.0010455, 0.0010455
1: 0.9931464, 0.9958588, 0.9931464, 0.9958588, -0.0022440, 0.0022440
2: -0.0080011, -0.0068185, -0.0080011, -0.0068185, -0.0009672, 0.0009672
3: 0.0027002, 0.0043026, 0.0027002, 0.0043026, -0.0013295, 0.0013295
4: 0.0025174, 0.0046060, 0.0025174, 0.0046060, -0.0020683, 0.0020683
5: 0.0033852, 0.0064204, 0.0033852, 0.0064204, -0.0023896, 0.0023896
6: -0.0021098, 0.0006967, -0.0021098, 0.0006967, -0.0024219, 0.0024219
7: -0.0080484, -0.0067501, -0.0080484, -0.0067501, -0.0009879, 0.0009879
8: 0.0072102, 0.0081893, 0.0072102, 0.0081893, -0.0009197, 0.0009197
9: -0.0039284, -0.0020744, -0.0039284, -0.0020744, -0.0015053, 0.0015053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 137

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012901, upper bound: 0.0014972
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0012894, upper bound: 0.0015113
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000090, 0.0013243, 0.0000097, 0.0012905, -0.0010618, 0.0010771
1: 0.9930748, 0.9958603, 0.9931464, 0.9958588, -0.0023108, 0.0022841
2: -0.0080014, -0.0067017, -0.0080011, -0.0068185, -0.0009666, 0.0010808
3: 0.0026993, 0.0043449, 0.0027002, 0.0043026, -0.0013540, 0.0013690
4: 0.0025163, 0.0046611, 0.0025174, 0.0046060, -0.0020897, 0.0021197
5: 0.0033836, 0.0065005, 0.0033852, 0.0064204, -0.0024123, 0.0024643
6: -0.0021839, 0.0006983, -0.0021098, 0.0006967, -0.0024910, 0.0024843
7: -0.0080827, -0.0067494, -0.0080484, -0.0067501, -0.0010199, 0.0009922
8: 0.0070568, 0.0081894, 0.0072102, 0.0081893, -0.0010716, 0.0009195
9: -0.0039773, -0.0020734, -0.0039284, -0.0020744, -0.0015510, 0.0015273

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 137

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011697, upper bound: 0.0015059
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011764, upper bound: 0.0015059
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000090, 0.0013243, 0.0000090, 0.0013243, -0.0010855, 0.0010855
1: 0.9930748, 0.9958603, 0.9930748, 0.9958603, -0.0023384, 0.0023384
2: -0.0080014, -0.0067017, -0.0080014, -0.0067017, -0.0010740, 0.0010740
3: 0.0026993, 0.0043449, 0.0026993, 0.0043449, -0.0013867, 0.0013867
4: 0.0025163, 0.0046611, 0.0025163, 0.0046611, -0.0021448, 0.0021448
5: 0.0033836, 0.0065005, 0.0033836, 0.0065005, -0.0024680, 0.0024680
6: -0.0021839, 0.0006983, -0.0021839, 0.0006983, -0.0025572, 0.0025572
7: -0.0080827, -0.0067494, -0.0080827, -0.0067494, -0.0010148, 0.0010148
8: 0.0070568, 0.0081894, 0.0070568, 0.0081894, -0.0010695, 0.0010695
9: -0.0039773, -0.0020734, -0.0039773, -0.0020734, -0.0015612, 0.0015612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 137

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011799, upper bound: 0.0014912
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011864, upper bound: 0.0014884
time: 0.72 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 6.78 seconds
IS_A1_B1_A2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0011459, upper bound: 0.0014829
IS_A1_B1_A2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0011291, upper bound: 0.0014898
IS_A1_B2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0012720, upper bound: 0.0015053
IS_A1_B2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0012653, upper bound: 0.0015113
IS_A1_B2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0012608, upper bound: 0.0015472
IS_A1_B2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0012653, upper bound: 0.0015472
IS_A1_B2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0011267, upper bound: 0.0015062
IS_A1_B2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0011305, upper bound: 0.0015062
IS_A1_B2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0012581, upper bound: 0.0015011
IS_A1_B2_B2_A1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0012487, upper bound: 0.0014425
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0012900, upper bound: 0.0014973
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0012891, upper bound: 0.0015113
IS_A2_B1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0011695, upper bound: 0.0015038
IS_A2_B1_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0011755, upper bound: 0.0012445
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0012901, upper bound: 0.0014972
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0012894, upper bound: 0.0015113
IS_A2_B2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0011697, upper bound: 0.0015059
IS_A2_B2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0011764, upper bound: 0.0015059
IS_A2_B2_A2_B2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0011799, upper bound: 0.0014912
IS_A2_B2_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 6.78
Output dim: 1, lower bound: -0.0011864, upper bound: 0.0014884

## BFS IS instance: IS_A1_B2_B1_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0000620, 0.0012805, 0.0000797, 0.0013096, -0.0010297, 0.0009887
1: 0.9931676, 0.9957477, 0.9931059, 0.9957104, -0.0021238, 0.0022073
2: -0.0079749, -0.0068531, -0.0079660, -0.0067525, -0.0010370, 0.0009236
3: 0.0027657, 0.0042900, 0.0027878, 0.0043265, -0.0013074, 0.0012586
4: 0.0026028, 0.0045896, 0.0026317, 0.0046371, -0.0020033, 0.0019580
5: 0.0035094, 0.0063966, 0.0035512, 0.0064656, -0.0023650, 0.0022590
6: -0.0020878, 0.0005820, -0.0021517, 0.0005432, -0.0023075, 0.0023727
7: -0.0080382, -0.0068032, -0.0080678, -0.0068211, -0.0009361, 0.0009839
8: 0.0072558, 0.0081823, 0.0071236, 0.0081800, -0.0008705, 0.0010068
9: -0.0039138, -0.0021503, -0.0039560, -0.0021758, -0.0014234, 0.0014834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 137

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0012639, upper bound: 0.0015053
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0012639, upper bound: 0.0015053
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0000623, 0.0012824, 0.0000806, 0.0013093, -0.0010316, 0.0009908
1: 0.9931634, 0.9957474, 0.9931067, 0.9957086, -0.0021281, 0.0022115
2: -0.0079747, -0.0068463, -0.0079656, -0.0067536, -0.0010367, 0.0009298
3: 0.0027660, 0.0042925, 0.0027889, 0.0043261, -0.0013100, 0.0012612
4: 0.0026032, 0.0045928, 0.0026330, 0.0046366, -0.0020109, 0.0019598
5: 0.0035099, 0.0064013, 0.0035532, 0.0064648, -0.0023684, 0.0022638
6: -0.0020922, 0.0005814, -0.0021510, 0.0005414, -0.0023118, 0.0023783
7: -0.0080402, -0.0068035, -0.0080674, -0.0068220, -0.0009382, 0.0009852
8: 0.0072468, 0.0081823, 0.0071250, 0.0081799, -0.0008795, 0.0010055
9: -0.0039167, -0.0021506, -0.0039555, -0.0021770, -0.0014263, 0.0014861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 137

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0012639, upper bound: 0.0015113
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0012639, upper bound: 0.0015113
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0000546, 0.0012828, 0.0000170, 0.0012882, -0.0010110, 0.0010276
1: 0.9931627, 0.9957635, 0.9931513, 0.9958434, -0.0022019, 0.0021692
2: -0.0079786, -0.0068450, -0.0079974, -0.0068264, -0.0009628, 0.0009494
3: 0.0027564, 0.0042930, 0.0027093, 0.0042997, -0.0012852, 0.0013041
4: 0.0025907, 0.0045935, 0.0025293, 0.0046022, -0.0019947, 0.0020172
5: 0.0034917, 0.0064021, 0.0034025, 0.0064149, -0.0023132, 0.0023625
6: -0.0020930, 0.0005983, -0.0021048, 0.0006808, -0.0023674, 0.0023399
7: -0.0080406, -0.0067956, -0.0080461, -0.0067575, -0.0009844, 0.0009588
8: 0.0072451, 0.0081833, 0.0072207, 0.0081883, -0.0008858, 0.0009099
9: -0.0039172, -0.0021395, -0.0039250, -0.0020850, -0.0014806, 0.0014558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011472, upper bound: 0.0010501
time: 0.54 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014252, upper bound: 0.0015409
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000554, 0.0012825, 0.0000175, 0.0012902, -0.0010123, 0.0010288
1: 0.9931633, 0.9957618, 0.9931470, 0.9958422, -0.0022049, 0.0021719
2: -0.0079782, -0.0068460, -0.0079972, -0.0068194, -0.0009684, 0.0009488
3: 0.0027574, 0.0042926, 0.0027100, 0.0043022, -0.0012868, 0.0013060
4: 0.0025921, 0.0045930, 0.0025302, 0.0046056, -0.0019962, 0.0020246
5: 0.0034937, 0.0064014, 0.0034038, 0.0064197, -0.0023164, 0.0023644
6: -0.0020923, 0.0005965, -0.0021093, 0.0006796, -0.0023724, 0.0023425
7: -0.0080403, -0.0067965, -0.0080481, -0.0067580, -0.0009850, 0.0009602
8: 0.0072465, 0.0081832, 0.0072114, 0.0081883, -0.0008845, 0.0009186
9: -0.0039168, -0.0021407, -0.0039280, -0.0020857, -0.0014822, 0.0014577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011520, upper bound: 0.0010291
time: 0.68 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014379, upper bound: 0.0015409
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0000486, 0.0013223, 0.0000868, 0.0013072, -0.0010540, 0.0010140
1: 0.9930792, 0.9957763, 0.9931111, 0.9956954, -0.0021760, 0.0022654
2: -0.0079816, -0.0067087, -0.0079625, -0.0067609, -0.0010315, 0.0010584
3: 0.0027488, 0.0043423, 0.0027966, 0.0043234, -0.0013428, 0.0012894
4: 0.0025808, 0.0046578, 0.0026432, 0.0046332, -0.0020523, 0.0020146
5: 0.0034774, 0.0064956, 0.0035680, 0.0064599, -0.0024031, 0.0023205
6: -0.0021794, 0.0006115, -0.0021464, 0.0005278, -0.0023607, 0.0024598
7: -0.0080806, -0.0067895, -0.0080653, -0.0068283, -0.0009626, 0.0009941
8: 0.0070661, 0.0081841, 0.0071346, 0.0081790, -0.0010582, 0.0009964
9: -0.0039743, -0.0021307, -0.0039525, -0.0021860, -0.0014602, 0.0015170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 137

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011420, upper bound: 0.0015008
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011420, upper bound: 0.0015062
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000494, 0.0013219, 0.0000876, 0.0013086, -0.0010552, 0.0010153
1: 0.9930798, 0.9957746, 0.9931080, 0.9956937, -0.0021798, 0.0022678
2: -0.0079812, -0.0067098, -0.0079621, -0.0067558, -0.0010364, 0.0010580
3: 0.0027499, 0.0043419, 0.0027976, 0.0043253, -0.0013442, 0.0012917
4: 0.0025822, 0.0046573, 0.0026445, 0.0046356, -0.0020533, 0.0020128
5: 0.0034794, 0.0064949, 0.0035698, 0.0064633, -0.0024059, 0.0023225
6: -0.0021788, 0.0006097, -0.0021496, 0.0005261, -0.0023655, 0.0024620
7: -0.0080803, -0.0067904, -0.0080668, -0.0068291, -0.0009634, 0.0009953
8: 0.0070675, 0.0081840, 0.0071279, 0.0081789, -0.0010570, 0.0010035
9: -0.0039739, -0.0021319, -0.0039546, -0.0021872, -0.0014619, 0.0015186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 137

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011459, upper bound: 0.0015008
time: 0.64 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0011459, upper bound: 0.0015062
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_B2_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0000620, 0.0012805, 0.0000090, 0.0013243, -0.0010321, 0.0010471
1: 0.9931676, 0.9957477, 0.9930748, 0.9958602, -0.0022482, 0.0022135
2: -0.0079749, -0.0068531, -0.0080014, -0.0067017, -0.0010774, 0.0009435
3: 0.0027657, 0.0042900, 0.0026994, 0.0043449, -0.0013113, 0.0013322
4: 0.0026028, 0.0045896, 0.0025163, 0.0046611, -0.0020261, 0.0020733
5: 0.0035094, 0.0063966, 0.0033837, 0.0065004, -0.0023647, 0.0023935
6: -0.0020878, 0.0005820, -0.0021839, 0.0006982, -0.0024379, 0.0023849
7: -0.0080382, -0.0068032, -0.0080827, -0.0067494, -0.0009916, 0.0009813
8: 0.0072558, 0.0081823, 0.0070568, 0.0081894, -0.0008757, 0.0010704
9: -0.0039138, -0.0021503, -0.0039773, -0.0020735, -0.0015074, 0.0014864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011986, upper bound: 0.0007570
time: 0.55 seconds

## Relational analysis of IS_A1_B2_B2_A1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B2_A1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0014316, upper bound: 0.0014960
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000175, 0.0012902, 0.0000554, 0.0012825, -0.0010288, 0.0010123
1: 0.9931470, 0.9958422, 0.9931633, 0.9957618, -0.0021719, 0.0022049
2: -0.0079972, -0.0068194, -0.0079782, -0.0068460, -0.0009488, 0.0009684
3: 0.0027100, 0.0043022, 0.0027574, 0.0042926, -0.0013060, 0.0012868
4: 0.0025302, 0.0046056, 0.0025921, 0.0045930, -0.0020246, 0.0019962
5: 0.0034038, 0.0064197, 0.0034937, 0.0064014, -0.0023644, 0.0023164
6: -0.0021093, 0.0006796, -0.0020923, 0.0005965, -0.0023425, 0.0023724
7: -0.0080481, -0.0067580, -0.0080403, -0.0067965, -0.0009602, 0.0009850
8: 0.0072114, 0.0081883, 0.0072465, 0.0081832, -0.0009186, 0.0008845
9: -0.0039280, -0.0020857, -0.0039168, -0.0021407, -0.0014577, 0.0014822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0009600, upper bound: 0.0011904
time: 0.54 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014545, upper bound: 0.0015046
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0000090, 0.0013243, 0.0000620, 0.0012805, -0.0010471, 0.0010321
1: 0.9930748, 0.9958602, 0.9931676, 0.9957477, -0.0022135, 0.0022482
2: -0.0080014, -0.0067017, -0.0079749, -0.0068531, -0.0009435, 0.0010774
3: 0.0026994, 0.0043449, 0.0027657, 0.0042900, -0.0013322, 0.0013113
4: 0.0025163, 0.0046611, 0.0026028, 0.0045896, -0.0020733, 0.0020261
5: 0.0033837, 0.0065004, 0.0035094, 0.0063966, -0.0023935, 0.0023647
6: -0.0021839, 0.0006982, -0.0020878, 0.0005820, -0.0023849, 0.0024379
7: -0.0080827, -0.0067494, -0.0080382, -0.0068032, -0.0009813, 0.0009916
8: 0.0070568, 0.0081894, 0.0072558, 0.0081823, -0.0010704, 0.0008757
9: -0.0039773, -0.0020735, -0.0039138, -0.0021503, -0.0014864, 0.0015074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0007110, upper bound: 0.0012373
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0013885, upper bound: 0.0014985
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000175, 0.0012902, 0.0000105, 0.0012902, -0.0010384, 0.0010464
1: 0.9931470, 0.9958422, 0.9931471, 0.9958569, -0.0022458, 0.0022289
2: -0.0079972, -0.0068194, -0.0080006, -0.0068195, -0.0009629, 0.0009664
3: 0.0027100, 0.0043022, 0.0027013, 0.0043022, -0.0013206, 0.0013306
4: 0.0025302, 0.0046056, 0.0025189, 0.0046055, -0.0020580, 0.0020690
5: 0.0034038, 0.0064197, 0.0033873, 0.0064196, -0.0023729, 0.0023919
6: -0.0021093, 0.0006796, -0.0021092, 0.0006948, -0.0024235, 0.0024065
7: -0.0080481, -0.0067580, -0.0080481, -0.0067510, -0.0009889, 0.0009808
8: 0.0072114, 0.0081883, 0.0072116, 0.0081892, -0.0009181, 0.0009174
9: -0.0039280, -0.0020857, -0.0039279, -0.0020757, -0.0015066, 0.0014951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 137

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014531, upper bound: 0.0015113
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014531, upper bound: 0.0015113
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0000090, 0.0013243, 0.0000170, 0.0012882, -0.0010604, 0.0010680
1: 0.9930748, 0.9958602, 0.9931513, 0.9958434, -0.0022914, 0.0022813
2: -0.0080014, -0.0067017, -0.0079974, -0.0068264, -0.0009591, 0.0010771
3: 0.0026994, 0.0043449, 0.0027093, 0.0042997, -0.0013523, 0.0013575
4: 0.0025163, 0.0046611, 0.0025293, 0.0046022, -0.0020859, 0.0021022
5: 0.0033837, 0.0065004, 0.0034025, 0.0064149, -0.0024091, 0.0024440
6: -0.0021839, 0.0006982, -0.0021048, 0.0006808, -0.0024703, 0.0024814
7: -0.0080827, -0.0067494, -0.0080461, -0.0067575, -0.0010119, 0.0009909
8: 0.0070568, 0.0081894, 0.0072207, 0.0081883, -0.0010705, 0.0009090
9: -0.0039773, -0.0020735, -0.0039250, -0.0020850, -0.0015380, 0.0015254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0005194, upper bound: 0.0011646
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0013885, upper bound: 0.0015009
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000099, 0.0013240, 0.0000175, 0.0012902, -0.0010627, 0.0010699
1: 0.9930755, 0.9958584, 0.9931470, 0.9958422, -0.0022956, 0.0022860
2: -0.0080010, -0.0067028, -0.0079972, -0.0068194, -0.0009658, 0.0010764
3: 0.0027004, 0.0043445, 0.0027100, 0.0043022, -0.0013551, 0.0013600
4: 0.0025178, 0.0046606, 0.0025302, 0.0046056, -0.0020878, 0.0021093
5: 0.0033857, 0.0064997, 0.0034038, 0.0064197, -0.0024146, 0.0024476
6: -0.0021832, 0.0006963, -0.0021093, 0.0006796, -0.0024756, 0.0024860
7: -0.0080823, -0.0067503, -0.0080481, -0.0067580, -0.0010128, 0.0009932
8: 0.0070583, 0.0081893, 0.0072114, 0.0081883, -0.0010692, 0.0009179
9: -0.0039768, -0.0020747, -0.0039280, -0.0020857, -0.0015407, 0.0015286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 137

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0013973, upper bound: 0.0014934
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0013973, upper bound: 0.0015059
time: 0.78 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 6.86 seconds
IS_A1_B2_B1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0012639, upper bound: 0.0015053
IS_A1_B2_B1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0012639, upper bound: 0.0015053
IS_A1_B2_B1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0012639, upper bound: 0.0015113
IS_A1_B2_B1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0012639, upper bound: 0.0015113
IS_A1_B2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0011472, upper bound: 0.0010501
IS_A1_B2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0014252, upper bound: 0.0015409
IS_A1_B2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0011520, upper bound: 0.0010291
IS_A1_B2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0014379, upper bound: 0.0015409
IS_A1_B2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0011420, upper bound: 0.0015008
IS_A1_B2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0011420, upper bound: 0.0015062
IS_A1_B2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0011459, upper bound: 0.0015008
IS_A1_B2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0011459, upper bound: 0.0015062
IS_A1_B2_B2_A1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0011986, upper bound: 0.0007570
IS_A1_B2_B2_A1_A2_B2_A1_A2, status: Status.VERIFIED, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0014316, upper bound: 0.0014960
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0009600, upper bound: 0.0011904
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0014545, upper bound: 0.0015046
IS_A2_B1_A2_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0007110, upper bound: 0.0012373
IS_A2_B1_A2_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0013885, upper bound: 0.0014985
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0014531, upper bound: 0.0015113
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0014531, upper bound: 0.0015113
IS_A2_B2_A2_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0005194, upper bound: 0.0011646
IS_A2_B2_A2_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0013885, upper bound: 0.0015009
IS_A2_B2_A2_B1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0013973, upper bound: 0.0014934
IS_A2_B2_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 6.86
Output dim: 1, lower bound: -0.0013973, upper bound: 0.0015059

## BFS IS instance: IS_A1_B2_B1_A1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0000620, 0.0012805, 0.0000868, 0.0013072, -0.0010282, 0.0009804
1: 0.9931676, 0.9957477, 0.9931111, 0.9956954, -0.0021047, 0.0022039
2: -0.0079749, -0.0068531, -0.0079625, -0.0067609, -0.0010290, 0.0009199
3: 0.0027657, 0.0042900, 0.0027966, 0.0043234, -0.0013054, 0.0012473
4: 0.0026028, 0.0045896, 0.0026432, 0.0046332, -0.0020007, 0.0019464
5: 0.0035094, 0.0063966, 0.0035680, 0.0064599, -0.0023613, 0.0022406
6: -0.0020878, 0.0005820, -0.0021464, 0.0005278, -0.0022869, 0.0023693
7: -0.0080382, -0.0068032, -0.0080653, -0.0068283, -0.0009285, 0.0009823
8: 0.0072558, 0.0081823, 0.0071346, 0.0081790, -0.0008695, 0.0009958
9: -0.0039138, -0.0021503, -0.0039525, -0.0021860, -0.0014115, 0.0014812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0001991, upper bound: 0.0009216
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 200
type: A, layer: 3, pos: 208
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 200
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 239
type: A, layer: 3, pos: 239

Time for candidate selection: 12.75 seconds

### Candidate
type: B, layer: 3, pos: 151

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010994, upper bound: 0.0013415
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011869, upper bound: 0.0014133
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0000620, 0.0012805, 0.0000876, 0.0013086, -0.0010303, 0.0009802
1: 0.9931676, 0.9957477, 0.9931080, 0.9956937, -0.0021063, 0.0022084
2: -0.0079749, -0.0068531, -0.0079621, -0.0067558, -0.0010343, 0.0009197
3: 0.0027657, 0.0042900, 0.0027976, 0.0043253, -0.0013081, 0.0012482
4: 0.0026028, 0.0045896, 0.0026445, 0.0046356, -0.0020042, 0.0019452
5: 0.0035094, 0.0063966, 0.0035698, 0.0064633, -0.0023664, 0.0022387
6: -0.0020878, 0.0005820, -0.0021496, 0.0005261, -0.0022890, 0.0023739
7: -0.0080382, -0.0068032, -0.0080668, -0.0068291, -0.0009276, 0.0009844
8: 0.0072558, 0.0081823, 0.0071279, 0.0081789, -0.0008694, 0.0010030
9: -0.0039138, -0.0021503, -0.0039546, -0.0021872, -0.0014110, 0.0014842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0001991, upper bound: 0.0009216
time: 0.61 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 200
type: A, layer: 3, pos: 208
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 200
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 239
type: A, layer: 3, pos: 239

Time for candidate selection: 12.63 seconds

### Candidate
type: B, layer: 3, pos: 151

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010994, upper bound: 0.0013415
time: 0.56 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011869, upper bound: 0.0014133
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0000623, 0.0012824, 0.0000868, 0.0013072, -0.0010285, 0.0009834
1: 0.9931634, 0.9957474, 0.9931111, 0.9956954, -0.0021112, 0.0022055
2: -0.0079747, -0.0068463, -0.0079625, -0.0067609, -0.0010290, 0.0009265
3: 0.0027660, 0.0042925, 0.0027966, 0.0043234, -0.0013066, 0.0012511
4: 0.0026032, 0.0045928, 0.0026432, 0.0046332, -0.0020053, 0.0019497
5: 0.0035099, 0.0064013, 0.0035680, 0.0064599, -0.0023617, 0.0022479
6: -0.0020922, 0.0005814, -0.0021464, 0.0005278, -0.0022936, 0.0023721
7: -0.0080402, -0.0068035, -0.0080653, -0.0068283, -0.0009316, 0.0009827
8: 0.0072468, 0.0081823, 0.0071346, 0.0081790, -0.0008787, 0.0009958
9: -0.0039167, -0.0021506, -0.0039525, -0.0021860, -0.0014159, 0.0014817

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0001532, upper bound: 0.0009145
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 200
type: A, layer: 3, pos: 208
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 200
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 239
type: A, layer: 3, pos: 239

Time for candidate selection: 12.67 seconds

### Candidate
type: B, layer: 3, pos: 151

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010929, upper bound: 0.0013486
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011759, upper bound: 0.0014177
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000623, 0.0012824, 0.0000876, 0.0013086, -0.0010311, 0.0009831
1: 0.9931634, 0.9957474, 0.9931080, 0.9956937, -0.0021116, 0.0022103
2: -0.0079747, -0.0068463, -0.0079621, -0.0067558, -0.0010343, 0.0009264
3: 0.0027660, 0.0042925, 0.0027976, 0.0043253, -0.0013093, 0.0012514
4: 0.0026032, 0.0045928, 0.0026445, 0.0046356, -0.0020100, 0.0019484
5: 0.0035099, 0.0064013, 0.0035698, 0.0064633, -0.0023671, 0.0022462
6: -0.0020922, 0.0005814, -0.0021496, 0.0005261, -0.0022950, 0.0023771
7: -0.0080402, -0.0068035, -0.0080668, -0.0068291, -0.0009308, 0.0009846
8: 0.0072468, 0.0081823, 0.0071279, 0.0081789, -0.0008786, 0.0010030
9: -0.0039167, -0.0021506, -0.0039546, -0.0021872, -0.0014153, 0.0014853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0001532, upper bound: 0.0009145
time: 0.52 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 200
type: A, layer: 3, pos: 208
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 200
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 239
type: A, layer: 3, pos: 239

Time for candidate selection: 12.67 seconds

### Candidate
type: B, layer: 3, pos: 151

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010929, upper bound: 0.0013486
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011759, upper bound: 0.0014177
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0000573, 0.0012827, 0.0000170, 0.0012882, -0.0009639, 0.0010275
1: 0.9931628, 0.9957578, 0.9931513, 0.9958434, -0.0022016, 0.0020755
2: -0.0079772, -0.0068452, -0.0079974, -0.0068264, -0.0009318, 0.0009492
3: 0.0027598, 0.0042929, 0.0027093, 0.0042997, -0.0012307, 0.0013040
4: 0.0025952, 0.0045934, 0.0025293, 0.0046022, -0.0019769, 0.0020170
5: 0.0034982, 0.0064020, 0.0034025, 0.0064149, -0.0021925, 0.0023622
6: -0.0020929, 0.0005923, -0.0021048, 0.0006808, -0.0023672, 0.0022664
7: -0.0080406, -0.0067984, -0.0080461, -0.0067575, -0.0009843, 0.0009030
8: 0.0072453, 0.0081830, 0.0072207, 0.0081883, -0.0008855, 0.0009017
9: -0.0039171, -0.0021434, -0.0039250, -0.0020850, -0.0014804, 0.0013862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 137

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014252, upper bound: 0.0015381
time: 0.66 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014252, upper bound: 0.0015409
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000582, 0.0012824, 0.0000175, 0.0012902, -0.0009652, 0.0010287
1: 0.9931634, 0.9957560, 0.9931470, 0.9958422, -0.0022047, 0.0020783
2: -0.0079768, -0.0068463, -0.0079972, -0.0068194, -0.0009378, 0.0009486
3: 0.0027609, 0.0042925, 0.0027100, 0.0043022, -0.0012323, 0.0013058
4: 0.0025965, 0.0045929, 0.0025302, 0.0046056, -0.0019783, 0.0020244
5: 0.0035002, 0.0064013, 0.0034038, 0.0064197, -0.0021958, 0.0023642
6: -0.0020922, 0.0005905, -0.0021093, 0.0006796, -0.0023722, 0.0022690
7: -0.0080402, -0.0067993, -0.0080481, -0.0067580, -0.0009849, 0.0009045
8: 0.0072468, 0.0081828, 0.0072114, 0.0081883, -0.0008842, 0.0009106
9: -0.0039167, -0.0021446, -0.0039280, -0.0020857, -0.0014820, 0.0013882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 137
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 137

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014252, upper bound: 0.0015351
time: 0.63 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014252, upper bound: 0.0015409
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0000559, 0.0013199, 0.0000868, 0.0013072, -0.0010451, 0.0010125
1: 0.9930841, 0.9957609, 0.9931111, 0.9956954, -0.0021727, 0.0022461
2: -0.0079779, -0.0067167, -0.0079625, -0.0067609, -0.0010278, 0.0010508
3: 0.0027580, 0.0043394, 0.0027966, 0.0043234, -0.0013314, 0.0012875
4: 0.0025928, 0.0046540, 0.0026432, 0.0046332, -0.0020403, 0.0020108
5: 0.0034948, 0.0064902, 0.0035680, 0.0064599, -0.0023836, 0.0023168
6: -0.0021744, 0.0005954, -0.0021464, 0.0005278, -0.0023573, 0.0024393
7: -0.0080783, -0.0067970, -0.0080653, -0.0068283, -0.0009611, 0.0009861
8: 0.0070765, 0.0081832, 0.0071346, 0.0081790, -0.0010478, 0.0009954
9: -0.0039710, -0.0021414, -0.0039525, -0.0021860, -0.0014580, 0.0015041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 200
type: A, layer: 3, pos: 208
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 200
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 239
type: B, layer: 3, pos: 239

Time for candidate selection: 11.92 seconds

### Candidate
type: A, layer: 3, pos: 151

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0008499, upper bound: 0.0013622
time: 0.60 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010501, upper bound: 0.0014125
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0000564, 0.0013214, 0.0000868, 0.0013072, -0.0010457, 0.0010146
1: 0.9930810, 0.9957598, 0.9931111, 0.9956954, -0.0021771, 0.0022479
2: -0.0079777, -0.0067118, -0.0079625, -0.0067609, -0.0010276, 0.0010556
3: 0.0027587, 0.0043412, 0.0027966, 0.0043234, -0.0013323, 0.0012901
4: 0.0025937, 0.0046563, 0.0026432, 0.0046332, -0.0020395, 0.0020132
5: 0.0034960, 0.0064935, 0.0035680, 0.0064599, -0.0023839, 0.0023217
6: -0.0021775, 0.0005943, -0.0021464, 0.0005278, -0.0023618, 0.0024404
7: -0.0080797, -0.0067975, -0.0080653, -0.0068283, -0.0009632, 0.0009856
8: 0.0070701, 0.0081831, 0.0071346, 0.0081790, -0.0010546, 0.0009954
9: -0.0039730, -0.0021421, -0.0039525, -0.0021860, -0.0014609, 0.0015049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 200
type: A, layer: 3, pos: 208
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 200
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 239
type: B, layer: 3, pos: 239

Time for candidate selection: 12.19 seconds

### Candidate
type: A, layer: 3, pos: 151

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0008499, upper bound: 0.0013622
time: 0.59 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010501, upper bound: 0.0014138
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0000559, 0.0013199, 0.0000876, 0.0013086, -0.0010472, 0.0010124
1: 0.9930841, 0.9957609, 0.9931080, 0.9956937, -0.0021743, 0.0022506
2: -0.0079779, -0.0067167, -0.0079621, -0.0067558, -0.0010331, 0.0010506
3: 0.0027580, 0.0043394, 0.0027976, 0.0043253, -0.0013341, 0.0012884
4: 0.0025928, 0.0046540, 0.0026445, 0.0046356, -0.0020427, 0.0020096
5: 0.0034948, 0.0064902, 0.0035698, 0.0064633, -0.0023887, 0.0023148
6: -0.0021744, 0.0005954, -0.0021496, 0.0005261, -0.0023594, 0.0024440
7: -0.0080783, -0.0067970, -0.0080668, -0.0068291, -0.0009602, 0.0009883
8: 0.0070765, 0.0081832, 0.0071279, 0.0081789, -0.0010478, 0.0010026
9: -0.0039710, -0.0021414, -0.0039546, -0.0021872, -0.0014575, 0.0015072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 200
type: A, layer: 3, pos: 208
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 200
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 239
type: A, layer: 3, pos: 239

Time for candidate selection: 12.03 seconds

### Candidate
type: A, layer: 3, pos: 151

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0008499, upper bound: 0.0013616
time: 0.65 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010501, upper bound: 0.0014101
time: 0.56 seconds

## BFS IS instance: IS_A1_B2_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000564, 0.0013214, 0.0000876, 0.0013086, -0.0010478, 0.0010147
1: 0.9930810, 0.9957598, 0.9931080, 0.9956937, -0.0021784, 0.0022520
2: -0.0079777, -0.0067118, -0.0079621, -0.0067558, -0.0010329, 0.0010555
3: 0.0027587, 0.0043412, 0.0027976, 0.0043253, -0.0013349, 0.0012909
4: 0.0025937, 0.0046563, 0.0026445, 0.0046356, -0.0020419, 0.0020119
5: 0.0034960, 0.0064935, 0.0035698, 0.0064633, -0.0023889, 0.0023210
6: -0.0021775, 0.0005943, -0.0021496, 0.0005261, -0.0023641, 0.0024458
7: -0.0080797, -0.0067975, -0.0080668, -0.0068291, -0.0009628, 0.0009881
8: 0.0070701, 0.0081831, 0.0071279, 0.0081789, -0.0010546, 0.0010026
9: -0.0039730, -0.0021421, -0.0039546, -0.0021872, -0.0014610, 0.0015080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### IS candidates at layer 3
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 208
type: B, layer: 3, pos: 200
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 200
type: B, layer: 3, pos: 253
type: A, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: A, layer: 3, pos: 162
type: B, layer: 3, pos: 239
type: A, layer: 3, pos: 239

Time for candidate selection: 12.42 seconds

### Candidate
type: A, layer: 3, pos: 151

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0008499, upper bound: 0.0013617
time: 0.69 seconds

## Relational analysis of IS_A1_B2_B1_A2_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_B1_A2_B1_A2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010501, upper bound: 0.0014138
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000175, 0.0012902, 0.0000582, 0.0012824, -0.0010287, 0.0009652
1: 0.9931470, 0.9958422, 0.9931634, 0.9957560, -0.0020783, 0.0022047
2: -0.0079972, -0.0068194, -0.0079768, -0.0068463, -0.0009486, 0.0009378
3: 0.0027100, 0.0043022, 0.0027609, 0.0042925, -0.0013058, 0.0012323
4: 0.0025302, 0.0046056, 0.0025965, 0.0045929, -0.0020244, 0.0019783
5: 0.0034038, 0.0064197, 0.0035002, 0.0064013, -0.0023642, 0.0021958
6: -0.0021093, 0.0006796, -0.0020922, 0.0005905, -0.0022690, 0.0023722
7: -0.0080481, -0.0067580, -0.0080402, -0.0067993, -0.0009045, 0.0009849
8: 0.0072114, 0.0081883, 0.0072468, 0.0081828, -0.0009106, 0.0008842
9: -0.0039280, -0.0020857, -0.0039167, -0.0021446, -0.0013882, 0.0014820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 137
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 137

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014466, upper bound: 0.0015046
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0014466, upper bound: 0.0015046
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0000090, 0.0013243, 0.0000648, 0.0012804, -0.0010470, 0.0009735
1: 0.9930748, 0.9958602, 0.9931678, 0.9957420, -0.0020951, 0.0022479
2: -0.0080014, -0.0067017, -0.0079735, -0.0068533, -0.0009432, 0.0010474
3: 0.0026994, 0.0043449, 0.0027692, 0.0042899, -0.0013321, 0.0012423
4: 0.0025163, 0.0046611, 0.0026073, 0.0045895, -0.0020732, 0.0019895
5: 0.0033837, 0.0065004, 0.0035159, 0.0063964, -0.0023932, 0.0022161
6: -0.0021839, 0.0006982, -0.0020877, 0.0005759, -0.0022861, 0.0024377
7: -0.0080827, -0.0067494, -0.0080382, -0.0068060, -0.0009141, 0.0009915
8: 0.0070568, 0.0081894, 0.0072560, 0.0081820, -0.0010634, 0.0008754
9: -0.0039773, -0.0020735, -0.0039137, -0.0021543, -0.0014002, 0.0015072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 137

## Relational analysis of IS_A2_B1_A2_B1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0013885, upper bound: 0.0014974
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0013885, upper bound: 0.0014985
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0000175, 0.0012902, 0.0000170, 0.0012882, -0.0010362, 0.0010384
1: 0.9931470, 0.9958422, 0.9931513, 0.9958434, -0.0022287, 0.0022240
2: -0.0079972, -0.0068194, -0.0079974, -0.0068264, -0.0009556, 0.0009631
3: 0.0027100, 0.0043022, 0.0027093, 0.0042997, -0.0013177, 0.0013204
4: 0.0025302, 0.0046056, 0.0025293, 0.0046022, -0.0020526, 0.0020539
5: 0.0034038, 0.0064197, 0.0034025, 0.0064149, -0.0023680, 0.0023738
6: -0.0021093, 0.0006796, -0.0021048, 0.0006808, -0.0024054, 0.0024013
7: -0.0080481, -0.0067580, -0.0080461, -0.0067575, -0.0009819, 0.0009788
8: 0.0072114, 0.0081883, 0.0072207, 0.0081883, -0.0009172, 0.0009081
9: -0.0039280, -0.0020857, -0.0039250, -0.0020850, -0.0014951, 0.0014918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 239

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010860, upper bound: 0.0013594
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010225, upper bound: 0.0010711
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0000175, 0.0012902, 0.0000175, 0.0012902, -0.0010385, 0.0010385
1: 0.9931470, 0.9958422, 0.9931470, 0.9958422, -0.0022290, 0.0022290
2: -0.0079972, -0.0068194, -0.0079972, -0.0068194, -0.0009627, 0.0009627
3: 0.0027100, 0.0043022, 0.0027100, 0.0043022, -0.0013206, 0.0013206
4: 0.0025302, 0.0046056, 0.0025302, 0.0046056, -0.0020580, 0.0020580
5: 0.0034038, 0.0064197, 0.0034038, 0.0064197, -0.0023730, 0.0023730
6: -0.0021093, 0.0006796, -0.0021093, 0.0006796, -0.0024066, 0.0024066
7: -0.0080481, -0.0067580, -0.0080481, -0.0067580, -0.0009809, 0.0009809
8: 0.0072114, 0.0081883, 0.0072114, 0.0081883, -0.0009171, 0.0009171
9: -0.0039280, -0.0020857, -0.0039280, -0.0020857, -0.0014951, 0.0014951

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 239

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012866, upper bound: 0.0011248
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010225, upper bound: 0.0010711
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0000090, 0.0013243, 0.0000196, 0.0012881, -0.0010603, 0.0009810
1: 0.9930748, 0.9958602, 0.9931514, 0.9958376, -0.0021143, 0.0022811
2: -0.0080014, -0.0067017, -0.0079961, -0.0068266, -0.0009588, 0.0010412
3: 0.0026994, 0.0043449, 0.0027126, 0.0042996, -0.0013522, 0.0012539
4: 0.0025163, 0.0046611, 0.0025337, 0.0046021, -0.0020858, 0.0020229
5: 0.0033837, 0.0065004, 0.0034088, 0.0064148, -0.0024089, 0.0022249
6: -0.0021839, 0.0006982, -0.0021047, 0.0006749, -0.0023148, 0.0024811
7: -0.0080827, -0.0067494, -0.0080460, -0.0067602, -0.0009152, 0.0009908
8: 0.0070568, 0.0081894, 0.0072209, 0.0081880, -0.0010622, 0.0009088
9: -0.0039773, -0.0020735, -0.0039249, -0.0020888, -0.0014103, 0.0015252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 137

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0013885, upper bound: 0.0014989
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0013885, upper bound: 0.0015009
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000172, 0.0013236, 0.0000175, 0.0012902, -0.0010544, 0.0010698
1: 0.9930763, 0.9958427, 0.9931470, 0.9958422, -0.0022953, 0.0022683
2: -0.0079973, -0.0067041, -0.0079972, -0.0068194, -0.0009621, 0.0010745
3: 0.0027096, 0.0043440, 0.0027100, 0.0043022, -0.0013447, 0.0013598
4: 0.0025297, 0.0046600, 0.0025302, 0.0046056, -0.0020758, 0.0021091
5: 0.0034031, 0.0064988, 0.0034038, 0.0064197, -0.0023951, 0.0024472
6: -0.0021824, 0.0006802, -0.0021093, 0.0006796, -0.0024752, 0.0024679
7: -0.0080819, -0.0067578, -0.0080481, -0.0067580, -0.0010126, 0.0009851
8: 0.0070600, 0.0081883, 0.0072114, 0.0081883, -0.0010672, 0.0009169
9: -0.0039763, -0.0020854, -0.0039280, -0.0020857, -0.0015404, 0.0015166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 239

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0009954, upper bound: 0.0013505
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0009243, upper bound: 0.0010622
time: 0.66 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 10.84 seconds
IS_A1_B2_B1_A1_B1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0010994, upper bound: 0.0013415
IS_A1_B2_B1_A1_B1_A2_A1_B1_B2, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0011869, upper bound: 0.0014133
IS_A1_B2_B1_A1_B1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0010994, upper bound: 0.0013415
IS_A1_B2_B1_A1_B1_A2_A1_B2_B2, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0011869, upper bound: 0.0014133
IS_A1_B2_B1_A1_B1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0010929, upper bound: 0.0013486
IS_A1_B2_B1_A1_B1_A2_A2_B1_B2, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0011759, upper bound: 0.0014177
IS_A1_B2_B1_A1_B1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0010929, upper bound: 0.0013486
IS_A1_B2_B1_A1_B1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0011759, upper bound: 0.0014177
IS_A1_B2_B1_A1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0014252, upper bound: 0.0015381
IS_A1_B2_B1_A1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0014252, upper bound: 0.0015409
IS_A1_B2_B1_A1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0014252, upper bound: 0.0015351
IS_A1_B2_B1_A1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0014252, upper bound: 0.0015409
IS_A1_B2_B1_A2_B1_A2_B1_A1_A1, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0008499, upper bound: 0.0013622
IS_A1_B2_B1_A2_B1_A2_B1_A1_A2, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0010501, upper bound: 0.0014125
IS_A1_B2_B1_A2_B1_A2_B1_A2_A1, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0008499, upper bound: 0.0013622
IS_A1_B2_B1_A2_B1_A2_B1_A2_A2, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0010501, upper bound: 0.0014138
IS_A1_B2_B1_A2_B1_A2_B2_A1_A1, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0008499, upper bound: 0.0013616
IS_A1_B2_B1_A2_B1_A2_B2_A1_A2, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0010501, upper bound: 0.0014101
IS_A1_B2_B1_A2_B1_A2_B2_A2_A1, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0008499, upper bound: 0.0013617
IS_A1_B2_B1_A2_B1_A2_B2_A2_A2, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0010501, upper bound: 0.0014138
IS_A2_B1_A1_B1_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0014466, upper bound: 0.0015046
IS_A2_B1_A1_B1_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0014466, upper bound: 0.0015046
IS_A2_B1_A2_B1_B2_A2_B1_B2_A1, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0013885, upper bound: 0.0014974
IS_A2_B1_A2_B1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0013885, upper bound: 0.0014985
IS_A2_B2_A1_B1_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0010860, upper bound: 0.0013594
IS_A2_B2_A1_B1_A2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0010225, upper bound: 0.0010711
IS_A2_B2_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0012866, upper bound: 0.0011248
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0010225, upper bound: 0.0010711
IS_A2_B2_A2_B1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0013885, upper bound: 0.0014989
IS_A2_B2_A2_B1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0013885, upper bound: 0.0015009
IS_A2_B2_A2_B1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0009954, upper bound: 0.0013505
IS_A2_B2_A2_B1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 9, time: 10.84
Output dim: 1, lower bound: -0.0009243, upper bound: 0.0010622

## BFS IS instance: IS_A1_B2_B1_A1_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0000648, 0.0012804, 0.0000170, 0.0012882, -0.0009551, 0.0010262
1: 0.9931678, 0.9957420, 0.9931513, 0.9958434, -0.0021988, 0.0020562
2: -0.0079735, -0.0068533, -0.0079974, -0.0068264, -0.0009283, 0.0009416
3: 0.0027692, 0.0042899, 0.0027093, 0.0042997, -0.0012193, 0.0013023
4: 0.0026073, 0.0045895, 0.0025293, 0.0046022, -0.0019595, 0.0020148
5: 0.0035159, 0.0063964, 0.0034025, 0.0064149, -0.0021726, 0.0023590
6: -0.0020877, 0.0005759, -0.0021048, 0.0006808, -0.0023642, 0.0022458
7: -0.0080382, -0.0068060, -0.0080461, -0.0067575, -0.0009829, 0.0008955
8: 0.0072560, 0.0081820, 0.0072207, 0.0081883, -0.0008750, 0.0009007
9: -0.0039137, -0.0021543, -0.0039250, -0.0020850, -0.0014785, 0.0013736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010702, upper bound: 0.0013869
time: 0.67 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010101, upper bound: 0.0010970
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0000650, 0.0012824, 0.0000170, 0.0012882, -0.0009556, 0.0010299
1: 0.9931636, 0.9957415, 0.9931513, 0.9958434, -0.0022067, 0.0020582
2: -0.0079734, -0.0068465, -0.0079974, -0.0068264, -0.0009277, 0.0009484
3: 0.0027694, 0.0042924, 0.0027093, 0.0042997, -0.0012204, 0.0013069
4: 0.0026077, 0.0045927, 0.0025293, 0.0046022, -0.0019626, 0.0020209
5: 0.0035164, 0.0064011, 0.0034025, 0.0064149, -0.0021731, 0.0023679
6: -0.0020920, 0.0005755, -0.0021048, 0.0006808, -0.0023724, 0.0022475
7: -0.0080402, -0.0068062, -0.0080461, -0.0067575, -0.0009868, 0.0008947
8: 0.0072471, 0.0081819, 0.0072207, 0.0081883, -0.0008842, 0.0009006
9: -0.0039166, -0.0021545, -0.0039250, -0.0020850, -0.0014839, 0.0013742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 239

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010702, upper bound: 0.0013869
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010101, upper bound: 0.0010980
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0000648, 0.0012804, 0.0000175, 0.0012902, -0.0009575, 0.0010261
1: 0.9931678, 0.9957420, 0.9931470, 0.9958422, -0.0021991, 0.0020613
2: -0.0079735, -0.0068533, -0.0079972, -0.0068194, -0.0009347, 0.0009412
3: 0.0027692, 0.0042899, 0.0027100, 0.0043022, -0.0012223, 0.0013027
4: 0.0026073, 0.0045895, 0.0025302, 0.0046056, -0.0019634, 0.0020189
5: 0.0035159, 0.0063964, 0.0034038, 0.0064197, -0.0021782, 0.0023574
6: -0.0020877, 0.0005759, -0.0021093, 0.0006796, -0.0023666, 0.0022511
7: -0.0080382, -0.0068060, -0.0080481, -0.0067580, -0.0009821, 0.0008979
8: 0.0072560, 0.0081820, 0.0072114, 0.0081883, -0.0008749, 0.0009097
9: -0.0039137, -0.0021543, -0.0039280, -0.0020857, -0.0014783, 0.0013771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2_A2_A1_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010702, upper bound: 0.0013732
time: 0.71 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2_A2_A1_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010101, upper bound: 0.0010862
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_B1_A1_B2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0000650, 0.0012824, 0.0000175, 0.0012902, -0.0009578, 0.0010289
1: 0.9931636, 0.9957415, 0.9931470, 0.9958422, -0.0022050, 0.0020626
2: -0.0079734, -0.0068465, -0.0079972, -0.0068194, -0.0009344, 0.0009481
3: 0.0027694, 0.0042924, 0.0027100, 0.0043022, -0.0012230, 0.0013060
4: 0.0026077, 0.0045927, 0.0025302, 0.0046056, -0.0019687, 0.0020247
5: 0.0035164, 0.0064011, 0.0034038, 0.0064197, -0.0021780, 0.0023646
6: -0.0020920, 0.0005755, -0.0021093, 0.0006796, -0.0023726, 0.0022535
7: -0.0080402, -0.0068062, -0.0080481, -0.0067580, -0.0009851, 0.0008969
8: 0.0072471, 0.0081819, 0.0072114, 0.0081883, -0.0008840, 0.0009096
9: -0.0039166, -0.0021545, -0.0039280, -0.0020857, -0.0014823, 0.0013774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 239
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 229
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 239

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010702, upper bound: 0.0013776
time: 0.70 seconds

## Relational analysis of IS_A1_B2_B1_A1_B2_A2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_B1_A1_B2_A2_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010101, upper bound: 0.0010900
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0000175, 0.0012902, 0.0000648, 0.0012804, -0.0010261, 0.0009575
1: 0.9931470, 0.9958422, 0.9931678, 0.9957420, -0.0020613, 0.0021991
2: -0.0079972, -0.0068194, -0.0079735, -0.0068533, -0.0009412, 0.0009347
3: 0.0027100, 0.0043022, 0.0027692, 0.0042899, -0.0013027, 0.0012223
4: 0.0025302, 0.0046056, 0.0026073, 0.0045895, -0.0020189, 0.0019634
5: 0.0034038, 0.0064197, 0.0035159, 0.0063964, -0.0023574, 0.0021782
6: -0.0021093, 0.0006796, -0.0020877, 0.0005759, -0.0022511, 0.0023666
7: -0.0080481, -0.0067580, -0.0080382, -0.0068060, -0.0008979, 0.0009821
8: 0.0072114, 0.0081883, 0.0072560, 0.0081820, -0.0009097, 0.0008749
9: -0.0039280, -0.0020857, -0.0039137, -0.0021543, -0.0013771, 0.0014783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 239

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012826, upper bound: 0.0011206
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010236, upper bound: 0.0010671
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0000175, 0.0012902, 0.0000650, 0.0012824, -0.0010289, 0.0009578
1: 0.9931470, 0.9958422, 0.9931636, 0.9957415, -0.0020626, 0.0022050
2: -0.0079972, -0.0068194, -0.0079734, -0.0068465, -0.0009481, 0.0009344
3: 0.0027100, 0.0043022, 0.0027694, 0.0042924, -0.0013060, 0.0012230
4: 0.0025302, 0.0046056, 0.0026077, 0.0045927, -0.0020247, 0.0019687
5: 0.0034038, 0.0064197, 0.0035164, 0.0064011, -0.0023646, 0.0021780
6: -0.0021093, 0.0006796, -0.0020920, 0.0005755, -0.0022535, 0.0023726
7: -0.0080481, -0.0067580, -0.0080402, -0.0068062, -0.0008969, 0.0009851
8: 0.0072114, 0.0081883, 0.0072471, 0.0081819, -0.0009096, 0.0008840
9: -0.0039280, -0.0020857, -0.0039166, -0.0021545, -0.0013774, 0.0014823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 229
type: B, layer: 1, pos: 239

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 239

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0012826, upper bound: 0.0011206
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0010236, upper bound: 0.0010671
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000172, 0.0013236, 0.0000648, 0.0012804, -0.0010385, 0.0009736
1: 0.9930763, 0.9958427, 0.9931678, 0.9957420, -0.0020954, 0.0022306
2: -0.0079973, -0.0067041, -0.0079735, -0.0068533, -0.0009390, 0.0010444
3: 0.0027096, 0.0043440, 0.0027692, 0.0042899, -0.0013219, 0.0012424
4: 0.0025297, 0.0046600, 0.0026073, 0.0045895, -0.0020598, 0.0019897
5: 0.0034031, 0.0064988, 0.0035159, 0.0063964, -0.0023722, 0.0022164
6: -0.0021824, 0.0006802, -0.0020877, 0.0005759, -0.0022864, 0.0024194
7: -0.0080819, -0.0067578, -0.0080382, -0.0068060, -0.0009143, 0.0009824
8: 0.0070600, 0.0081883, 0.0072560, 0.0081820, -0.0010603, 0.0008743
9: -0.0039763, -0.0020854, -0.0039137, -0.0021543, -0.0014004, 0.0014949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B1_A2_B1_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A2_B1_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B1_A2_B1_B2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0008759, upper bound: 0.0009230
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B1_B2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_B2_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0013871, upper bound: 0.0014969
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0000161, 0.0013219, 0.0000196, 0.0012881, -0.0010512, 0.0009799
1: 0.9930799, 0.9958451, 0.9931514, 0.9958376, -0.0021119, 0.0022616
2: -0.0079978, -0.0067099, -0.0079961, -0.0068266, -0.0009552, 0.0010333
3: 0.0027082, 0.0043419, 0.0027126, 0.0042996, -0.0013408, 0.0012525
4: 0.0025279, 0.0046572, 0.0025337, 0.0046021, -0.0020742, 0.0020210
5: 0.0034005, 0.0064948, 0.0034088, 0.0064148, -0.0023887, 0.0022221
6: -0.0021787, 0.0006827, -0.0021047, 0.0006749, -0.0023123, 0.0024605
7: -0.0080802, -0.0067566, -0.0080460, -0.0067602, -0.0009140, 0.0009829
8: 0.0070676, 0.0081885, 0.0072209, 0.0081880, -0.0010515, 0.0009078
9: -0.0039738, -0.0020837, -0.0039249, -0.0020888, -0.0014086, 0.0015122

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 239

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011938, upper bound: 0.0011227
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0009206, upper bound: 0.0010655
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0000172, 0.0013236, 0.0000196, 0.0012881, -0.0010521, 0.0009805
1: 0.9930763, 0.9958427, 0.9931514, 0.9958376, -0.0021132, 0.0022635
2: -0.0079973, -0.0067041, -0.0079961, -0.0068266, -0.0009548, 0.0010385
3: 0.0027096, 0.0043440, 0.0027126, 0.0042996, -0.0013418, 0.0012533
4: 0.0025297, 0.0046600, 0.0025337, 0.0046021, -0.0020724, 0.0020221
5: 0.0034031, 0.0064988, 0.0034088, 0.0064148, -0.0023900, 0.0022236
6: -0.0021824, 0.0006802, -0.0021047, 0.0006749, -0.0023137, 0.0024624
7: -0.0080819, -0.0067578, -0.0080460, -0.0067602, -0.0009147, 0.0009825
8: 0.0070600, 0.0081883, 0.0072209, 0.0081880, -0.0010591, 0.0009077
9: -0.0039763, -0.0020854, -0.0039249, -0.0020888, -0.0014095, 0.0015134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=2, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 239
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 166

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 239

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0011938, upper bound: 0.0011230
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0009206, upper bound: 0.0010668
time: 0.58 seconds

## Summary of splitting at layer (split count: 9)
- Time for IS candidates: 8.62 seconds
IS_A1_B2_B1_A1_B2_A2_B1_A2_A1_B1, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0010702, upper bound: 0.0013869
IS_A1_B2_B1_A1_B2_A2_B1_A2_A1_B2, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0010101, upper bound: 0.0010970
IS_A1_B2_B1_A1_B2_A2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0010702, upper bound: 0.0013869
IS_A1_B2_B1_A1_B2_A2_B1_A2_A2_B2, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0010101, upper bound: 0.0010980
IS_A1_B2_B1_A1_B2_A2_B2_A2_A1_B1, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0010702, upper bound: 0.0013732
IS_A1_B2_B1_A1_B2_A2_B2_A2_A1_B2, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0010101, upper bound: 0.0010862
IS_A1_B2_B1_A1_B2_A2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0010702, upper bound: 0.0013776
IS_A1_B2_B1_A1_B2_A2_B2_A2_A2_B2, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0010101, upper bound: 0.0010900
IS_A2_B1_A1_B1_A2_B2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0012826, upper bound: 0.0011206
IS_A2_B1_A1_B1_A2_B2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0010236, upper bound: 0.0010671
IS_A2_B1_A1_B1_A2_B2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0012826, upper bound: 0.0011206
IS_A2_B1_A1_B1_A2_B2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0010236, upper bound: 0.0010671
IS_A2_B1_A2_B1_B2_A2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0008759, upper bound: 0.0009230
IS_A2_B1_A2_B1_B2_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0013871, upper bound: 0.0014969
IS_A2_B2_A2_B1_B2_A2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0011938, upper bound: 0.0011227
IS_A2_B2_A2_B1_B2_A2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0009206, upper bound: 0.0010655
IS_A2_B2_A2_B1_B2_A2_B1_B2_A2_A1, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0011938, upper bound: 0.0011230
IS_A2_B2_A2_B1_B2_A2_B1_B2_A2_A2, status: Status.VERIFIED, split count: 10, time: 8.62
Output dim: 1, lower bound: -0.0009206, upper bound: 0.0010668

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.93 + 482.15 = 485.08 seconds
