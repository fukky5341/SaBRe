## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00076797


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041539, -0.0041192, -0.0041539, -0.0041192, -0.0000293, 0.0000293)
1: (-0.0082184, -0.0069199, -0.0082184, -0.0069199, -0.0010976, 0.0010976)
2: (0.9666011, 0.9681593, 0.9666011, 0.9681593, -0.0013172, 0.0013172)
3: (-0.0000391, 0.0114544, -0.0000391, 0.0114544, -0.0097155, 0.0097155)
4: (-0.0015642, -0.0006901, -0.0015642, -0.0006901, -0.0007389, 0.0007389)
5: (0.0156894, 0.0165729, 0.0156894, 0.0165729, -0.0007468, 0.0007468)
6: (0.0038457, 0.0042754, 0.0038457, 0.0042754, -0.0003632, 0.0003632)
7: (-0.0107467, -0.0077681, -0.0107467, -0.0077681, -0.0025178, 0.0025178)
8: (0.0082032, 0.0105663, 0.0082032, 0.0105663, -0.0019975, 0.0019975)
9: (0.0124788, 0.0167291, 0.0124788, 0.0167291, -0.0035928, 0.0035928)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.52 + 1.62 = 3.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0009036, upper bound: 0.0009037

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008489, upper bound: 0.0008293
time: 0.71 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008489, upper bound: 0.0008489
time: 0.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.59 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 2, lower bound: -0.0008489, upper bound: 0.0008293
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 2, lower bound: -0.0008489, upper bound: 0.0008489

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0041548, -0.0041221, -0.0041538, -0.0041198, -0.0000280, 0.0000259
1: -0.0082530, -0.0070273, -0.0082169, -0.0069415, -0.0010475, 0.0009710
2: 0.9665595, 0.9680303, 0.9666027, 0.9681334, -0.0012570, 0.0011652
3: -0.0003451, 0.0105032, -0.0000261, 0.0112632, -0.0092718, 0.0085943
4: -0.0014919, -0.0006668, -0.0015497, -0.0006910, -0.0006536, 0.0007052
5: 0.0157626, 0.0165964, 0.0157041, 0.0165719, -0.0006606, 0.0007127
6: 0.0038343, 0.0042399, 0.0038462, 0.0042683, -0.0003467, 0.0003213
7: -0.0105002, -0.0076888, -0.0106972, -0.0077715, -0.0022273, 0.0024029
8: 0.0083987, 0.0106292, 0.0082425, 0.0105636, -0.0017670, 0.0019063
9: 0.0128306, 0.0168423, 0.0125495, 0.0167243, -0.0031782, 0.0034287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008293, upper bound: 0.0008293
time: 0.73 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008293, upper bound: 0.0008292
time: 0.85 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0041538, -0.0041200, -0.0041539, -0.0041192, -0.0000293, 0.0000260
1: -0.0082168, -0.0069500, -0.0082184, -0.0069199, -0.0010955, 0.0009751
2: 0.9666030, 0.9681231, 0.9666011, 0.9681593, -0.0013146, 0.0011702
3: -0.0000250, 0.0111875, -0.0000391, 0.0114544, -0.0096965, 0.0086313
4: -0.0015439, -0.0006911, -0.0015642, -0.0006901, -0.0006565, 0.0007375
5: 0.0157100, 0.0165718, 0.0156894, 0.0165729, -0.0006635, 0.0007453
6: 0.0038462, 0.0042655, 0.0038457, 0.0042754, -0.0003625, 0.0003227
7: -0.0106776, -0.0077718, -0.0107467, -0.0077681, -0.0022369, 0.0025129
8: 0.0082581, 0.0105634, 0.0082032, 0.0105663, -0.0017746, 0.0019936
9: 0.0125775, 0.0167239, 0.0124788, 0.0167291, -0.0031918, 0.0035858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008292, upper bound: 0.0008489
time: 0.75 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008292, upper bound: 0.0008489
time: 0.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.17 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 2, lower bound: -0.0008293, upper bound: 0.0008293
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 2, lower bound: -0.0008293, upper bound: 0.0008292
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 2, lower bound: -0.0008292, upper bound: 0.0008489
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 2, lower bound: -0.0008292, upper bound: 0.0008489

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041548, -0.0041221, -0.0041548, -0.0041221, -0.0000255, 0.0000255
1: -0.0082530, -0.0070273, -0.0082530, -0.0070273, -0.0009532, 0.0009532
2: 0.9665595, 0.9680303, 0.9665595, 0.9680303, -0.0011439, 0.0011439
3: -0.0003451, 0.0105032, -0.0003451, 0.0105032, -0.0084374, 0.0084374
4: -0.0014919, -0.0006668, -0.0014919, -0.0006668, -0.0006417, 0.0006417
5: 0.0157626, 0.0165964, 0.0157626, 0.0165964, -0.0006486, 0.0006486
6: 0.0038343, 0.0042399, 0.0038343, 0.0042399, -0.0003155, 0.0003155
7: -0.0105002, -0.0076888, -0.0105002, -0.0076888, -0.0021866, 0.0021866
8: 0.0083987, 0.0106292, 0.0083987, 0.0106292, -0.0017348, 0.0017348
9: 0.0128306, 0.0168423, 0.0128306, 0.0168423, -0.0031202, 0.0031202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007822, upper bound: 0.0007974
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007934, upper bound: 0.0007934
time: 0.76 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041548, -0.0041221, -0.0041538, -0.0041200, -0.0000283, 0.0000259
1: -0.0082530, -0.0070273, -0.0082168, -0.0069500, -0.0010585, 0.0009703
2: 0.9665595, 0.9680303, 0.9666030, 0.9681231, -0.0012703, 0.0011645
3: -0.0003451, 0.0105032, -0.0000250, 0.0111875, -0.0093694, 0.0085888
4: -0.0014919, -0.0006668, -0.0015439, -0.0006911, -0.0006532, 0.0007126
5: 0.0157626, 0.0165964, 0.0157100, 0.0165718, -0.0006602, 0.0007202
6: 0.0038343, 0.0042399, 0.0038462, 0.0042655, -0.0003503, 0.0003211
7: -0.0105002, -0.0076888, -0.0106776, -0.0077718, -0.0022259, 0.0024282
8: 0.0083987, 0.0106292, 0.0082581, 0.0105634, -0.0017659, 0.0019264
9: 0.0128306, 0.0168423, 0.0125775, 0.0167239, -0.0031761, 0.0034648

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007822, upper bound: 0.0007975
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007934, upper bound: 0.0007934
time: 0.84 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041538, -0.0041200, -0.0041548, -0.0041221, -0.0000259, 0.0000283
1: -0.0082168, -0.0069500, -0.0082530, -0.0070273, -0.0009703, 0.0010585
2: 0.9666030, 0.9681231, 0.9665595, 0.9680303, -0.0011645, 0.0012703
3: -0.0000250, 0.0111875, -0.0003451, 0.0105032, -0.0085888, 0.0093694
4: -0.0015439, -0.0006911, -0.0014919, -0.0006668, -0.0007126, 0.0006532
5: 0.0157100, 0.0165718, 0.0157626, 0.0165964, -0.0007202, 0.0006602
6: 0.0038462, 0.0042655, 0.0038343, 0.0042399, -0.0003211, 0.0003503
7: -0.0106776, -0.0077718, -0.0105002, -0.0076888, -0.0024282, 0.0022259
8: 0.0082581, 0.0105634, 0.0083987, 0.0106292, -0.0019264, 0.0017659
9: 0.0125775, 0.0167239, 0.0128306, 0.0168423, -0.0034648, 0.0031761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007974, upper bound: 0.0008054
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007934, upper bound: 0.0008173
time: 0.78 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041538, -0.0041200, -0.0041538, -0.0041200, -0.0000260, 0.0000260
1: -0.0082168, -0.0069500, -0.0082168, -0.0069500, -0.0009730, 0.0009730
2: 0.9666030, 0.9681231, 0.9666030, 0.9681231, -0.0011677, 0.0011677
3: -0.0000250, 0.0111875, -0.0000250, 0.0111875, -0.0086124, 0.0086124
4: -0.0015439, -0.0006911, -0.0015439, -0.0006911, -0.0006550, 0.0006550
5: 0.0157100, 0.0165718, 0.0157100, 0.0165718, -0.0006620, 0.0006620
6: 0.0038462, 0.0042655, 0.0038462, 0.0042655, -0.0003220, 0.0003220
7: -0.0106776, -0.0077718, -0.0106776, -0.0077718, -0.0022320, 0.0022320
8: 0.0082581, 0.0105634, 0.0082581, 0.0105634, -0.0017708, 0.0017708
9: 0.0125775, 0.0167239, 0.0125775, 0.0167239, -0.0031849, 0.0031849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007822, upper bound: 0.0008195
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007934, upper bound: 0.0008191
time: 0.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.28 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 2, lower bound: -0.0007822, upper bound: 0.0007974
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 2, lower bound: -0.0007934, upper bound: 0.0007934
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 2, lower bound: -0.0007822, upper bound: 0.0007975
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 2, lower bound: -0.0007934, upper bound: 0.0007934
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 2, lower bound: -0.0007974, upper bound: 0.0008054
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 2, lower bound: -0.0007934, upper bound: 0.0008173
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 2, lower bound: -0.0007822, upper bound: 0.0008195
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.28
Output dim: 2, lower bound: -0.0007934, upper bound: 0.0008191

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041540, -0.0041221, -0.0041548, -0.0041221, -0.0000246, 0.0000254
1: -0.0082245, -0.0070292, -0.0082530, -0.0070273, -0.0009211, 0.0009514
2: 0.9665937, 0.9680281, 0.9665595, 0.9680303, -0.0011053, 0.0011417
3: -0.0000936, 0.0104870, -0.0003451, 0.0105032, -0.0081526, 0.0084210
4: -0.0014906, -0.0006859, -0.0014919, -0.0006668, -0.0006405, 0.0006201
5: 0.0157638, 0.0165771, 0.0157626, 0.0165964, -0.0006473, 0.0006267
6: 0.0038437, 0.0042393, 0.0038343, 0.0042399, -0.0003048, 0.0003148
7: -0.0104961, -0.0077540, -0.0105002, -0.0076888, -0.0021824, 0.0021128
8: 0.0084021, 0.0105775, 0.0083987, 0.0106292, -0.0017314, 0.0016762
9: 0.0128366, 0.0167492, 0.0128306, 0.0168423, -0.0031141, 0.0030148

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007822, upper bound: 0.0007822
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007822, upper bound: 0.0007934
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041538, -0.0041211, -0.0041546, -0.0041221, -0.0000248, 0.0000267
1: -0.0082149, -0.0069895, -0.0082438, -0.0070282, -0.0009283, 0.0009981
2: 0.9666053, 0.9680758, 0.9665706, 0.9680294, -0.0011139, 0.0011978
3: -0.0000080, 0.0108385, -0.0002636, 0.0104953, -0.0082162, 0.0088349
4: -0.0015174, -0.0006924, -0.0014913, -0.0006730, -0.0006719, 0.0006249
5: 0.0157368, 0.0165705, 0.0157632, 0.0165902, -0.0006791, 0.0006316
6: 0.0038469, 0.0042524, 0.0038373, 0.0042396, -0.0003072, 0.0003303
7: -0.0105871, -0.0077762, -0.0104982, -0.0077099, -0.0022896, 0.0021293
8: 0.0083298, 0.0105599, 0.0084004, 0.0106124, -0.0018165, 0.0016893
9: 0.0127066, 0.0167176, 0.0128335, 0.0168121, -0.0032671, 0.0030384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007485, upper bound: 0.0007424
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007374, upper bound: 0.0007374
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041540, -0.0041221, -0.0041538, -0.0041200, -0.0000274, 0.0000259
1: -0.0082245, -0.0070292, -0.0082168, -0.0069500, -0.0010264, 0.0009685
2: 0.9665937, 0.9680281, 0.9666030, 0.9681231, -0.0012317, 0.0011622
3: -0.0000936, 0.0104870, -0.0000250, 0.0111875, -0.0090847, 0.0085724
4: -0.0014906, -0.0006859, -0.0015439, -0.0006911, -0.0006520, 0.0006909
5: 0.0157638, 0.0165771, 0.0157100, 0.0165718, -0.0006589, 0.0006983
6: 0.0038437, 0.0042393, 0.0038462, 0.0042655, -0.0003397, 0.0003205
7: -0.0104961, -0.0077540, -0.0106776, -0.0077718, -0.0022216, 0.0023544
8: 0.0084021, 0.0105775, 0.0082581, 0.0105634, -0.0017625, 0.0018678
9: 0.0128366, 0.0167492, 0.0125775, 0.0167239, -0.0031701, 0.0033595

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008054, upper bound: 0.0007822
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008054, upper bound: 0.0007934
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041538, -0.0041211, -0.0041536, -0.0041200, -0.0000276, 0.0000271
1: -0.0082149, -0.0069895, -0.0082077, -0.0069508, -0.0010338, 0.0010139
2: 0.9666053, 0.9680758, 0.9666139, 0.9681222, -0.0012406, 0.0012167
3: -0.0000080, 0.0108385, 0.0000559, 0.0111806, -0.0091501, 0.0089743
4: -0.0015174, -0.0006924, -0.0015434, -0.0006973, -0.0006825, 0.0006959
5: 0.0157368, 0.0165705, 0.0157105, 0.0165656, -0.0006898, 0.0007033
6: 0.0038469, 0.0042524, 0.0038493, 0.0042652, -0.0003421, 0.0003355
7: -0.0105871, -0.0077762, -0.0106758, -0.0077927, -0.0023258, 0.0023713
8: 0.0083298, 0.0105599, 0.0082595, 0.0105468, -0.0018452, 0.0018813
9: 0.0127066, 0.0167176, 0.0125800, 0.0166940, -0.0033187, 0.0033837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007779, upper bound: 0.0007424
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007374
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041538, -0.0041200, -0.0041540, -0.0041221, -0.0000259, 0.0000274
1: -0.0082168, -0.0069500, -0.0082245, -0.0070292, -0.0009685, 0.0010264
2: 0.9666030, 0.9681231, 0.9665937, 0.9680281, -0.0011622, 0.0012317
3: -0.0000250, 0.0111875, -0.0000936, 0.0104870, -0.0085724, 0.0090847
4: -0.0015439, -0.0006911, -0.0014906, -0.0006859, -0.0006909, 0.0006520
5: 0.0157100, 0.0165718, 0.0157638, 0.0165771, -0.0006983, 0.0006589
6: 0.0038462, 0.0042655, 0.0038437, 0.0042393, -0.0003205, 0.0003397
7: -0.0106776, -0.0077718, -0.0104961, -0.0077540, -0.0023544, 0.0022216
8: 0.0082581, 0.0105634, 0.0084021, 0.0105775, -0.0018678, 0.0017625
9: 0.0125775, 0.0167239, 0.0128366, 0.0167492, -0.0033595, 0.0031701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007822, upper bound: 0.0008054
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007822, upper bound: 0.0008054
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041536, -0.0041200, -0.0041538, -0.0041211, -0.0000271, 0.0000276
1: -0.0082077, -0.0069508, -0.0082149, -0.0069895, -0.0010139, 0.0010338
2: 0.9666139, 0.9681222, 0.9666053, 0.9680758, -0.0012167, 0.0012406
3: 0.0000559, 0.0111806, -0.0000080, 0.0108385, -0.0089743, 0.0091501
4: -0.0015434, -0.0006973, -0.0015174, -0.0006924, -0.0006959, 0.0006825
5: 0.0157105, 0.0165656, 0.0157368, 0.0165705, -0.0007033, 0.0006898
6: 0.0038493, 0.0042652, 0.0038469, 0.0042524, -0.0003355, 0.0003421
7: -0.0106758, -0.0077927, -0.0105871, -0.0077762, -0.0023713, 0.0023258
8: 0.0082595, 0.0105468, 0.0083298, 0.0105599, -0.0018813, 0.0018452
9: 0.0125800, 0.0166940, 0.0127066, 0.0167176, -0.0033837, 0.0033187

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007424, upper bound: 0.0007779
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007374, upper bound: 0.0007766
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041531, -0.0041200, -0.0041538, -0.0041200, -0.0000251, 0.0000259
1: -0.0081890, -0.0069517, -0.0082168, -0.0069500, -0.0009411, 0.0009709
2: 0.9666364, 0.9681212, 0.9666030, 0.9681231, -0.0011294, 0.0011651
3: 0.0002211, 0.0111724, -0.0000250, 0.0111875, -0.0083304, 0.0085936
4: -0.0015428, -0.0007098, -0.0015439, -0.0006911, -0.0006536, 0.0006336
5: 0.0157111, 0.0165529, 0.0157100, 0.0165718, -0.0006606, 0.0006403
6: 0.0038555, 0.0042649, 0.0038462, 0.0042655, -0.0003115, 0.0003213
7: -0.0106737, -0.0078356, -0.0106776, -0.0077718, -0.0022271, 0.0021589
8: 0.0082612, 0.0105128, 0.0082581, 0.0105634, -0.0017669, 0.0017128
9: 0.0125831, 0.0166329, 0.0125775, 0.0167239, -0.0031779, 0.0030806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007840, upper bound: 0.0008058
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007840, upper bound: 0.0008190
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041528, -0.0041190, -0.0041536, -0.0041200, -0.0000253, 0.0000272
1: -0.0081800, -0.0069111, -0.0082077, -0.0069508, -0.0009471, 0.0010189
2: 0.9666471, 0.9681699, 0.9666139, 0.9681222, -0.0011365, 0.0012227
3: 0.0003011, 0.0115323, 0.0000559, 0.0111806, -0.0083829, 0.0090182
4: -0.0015701, -0.0007159, -0.0015434, -0.0006973, -0.0006859, 0.0006376
5: 0.0156834, 0.0165468, 0.0157105, 0.0165656, -0.0006932, 0.0006444
6: 0.0038584, 0.0042784, 0.0038493, 0.0042652, -0.0003134, 0.0003372
7: -0.0107669, -0.0078563, -0.0106758, -0.0077927, -0.0023371, 0.0021725
8: 0.0081872, 0.0104963, 0.0082595, 0.0105468, -0.0018542, 0.0017236
9: 0.0124500, 0.0166033, 0.0125800, 0.0166940, -0.0033349, 0.0031000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007666, upper bound: 0.0007889
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007648, upper bound: 0.0007860
time: 0.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.24 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0007822, upper bound: 0.0007822
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0007822, upper bound: 0.0007934
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0007485, upper bound: 0.0007424
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0007374, upper bound: 0.0007374
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0008054, upper bound: 0.0007822
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0008054, upper bound: 0.0007934
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0007779, upper bound: 0.0007424
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007374
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0007822, upper bound: 0.0008054
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0007822, upper bound: 0.0008054
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0007424, upper bound: 0.0007779
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0007374, upper bound: 0.0007766
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0007840, upper bound: 0.0008058
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0007840, upper bound: 0.0008190
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0007666, upper bound: 0.0007889
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.24
Output dim: 2, lower bound: -0.0007648, upper bound: 0.0007860

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041540, -0.0041221, -0.0041540, -0.0041221, -0.0000245, 0.0000245
1: -0.0082245, -0.0070292, -0.0082245, -0.0070292, -0.0009192, 0.0009192
2: 0.9665937, 0.9680281, 0.9665937, 0.9680281, -0.0011031, 0.0011031
3: -0.0000936, 0.0104870, -0.0000936, 0.0104870, -0.0081362, 0.0081362
4: -0.0014906, -0.0006859, -0.0014906, -0.0006859, -0.0006188, 0.0006188
5: 0.0157638, 0.0165771, 0.0157638, 0.0165771, -0.0006254, 0.0006254
6: 0.0038437, 0.0042393, 0.0038437, 0.0042393, -0.0003042, 0.0003042
7: -0.0104961, -0.0077540, -0.0104961, -0.0077540, -0.0021086, 0.0021086
8: 0.0084021, 0.0105775, 0.0084021, 0.0105775, -0.0016728, 0.0016728
9: 0.0128366, 0.0167492, 0.0128366, 0.0167492, -0.0030088, 0.0030088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007373, upper bound: 0.0007624
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007586
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041540, -0.0041221, -0.0041538, -0.0041211, -0.0000260, 0.0000248
1: -0.0082245, -0.0070292, -0.0082149, -0.0069895, -0.0009727, 0.0009275
2: 0.9665937, 0.9680281, 0.9666053, 0.9680758, -0.0011672, 0.0011130
3: -0.0000936, 0.0104870, -0.0000080, 0.0108385, -0.0086093, 0.0082095
4: -0.0014906, -0.0006859, -0.0015174, -0.0006924, -0.0006244, 0.0006548
5: 0.0157638, 0.0165771, 0.0157368, 0.0165705, -0.0006310, 0.0006618
6: 0.0038437, 0.0042393, 0.0038469, 0.0042524, -0.0003219, 0.0003069
7: -0.0104961, -0.0077540, -0.0105871, -0.0077762, -0.0021276, 0.0022312
8: 0.0084021, 0.0105775, 0.0083298, 0.0105599, -0.0016879, 0.0017701
9: 0.0128366, 0.0167492, 0.0127066, 0.0167176, -0.0030359, 0.0031837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007373, upper bound: 0.0007624
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007586
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041540, -0.0041221, -0.0041531, -0.0041200, -0.0000274, 0.0000250
1: -0.0082245, -0.0070292, -0.0081890, -0.0069517, -0.0010248, 0.0009376
2: 0.9665937, 0.9680281, 0.9666364, 0.9681212, -0.0012298, 0.0011251
3: -0.0000936, 0.0104870, 0.0002211, 0.0111724, -0.0090711, 0.0082987
4: -0.0014906, -0.0006859, -0.0015428, -0.0007098, -0.0006312, 0.0006899
5: 0.0157638, 0.0165771, 0.0157111, 0.0165529, -0.0006379, 0.0006973
6: 0.0038437, 0.0042393, 0.0038555, 0.0042649, -0.0003392, 0.0003103
7: -0.0104961, -0.0077540, -0.0106737, -0.0078356, -0.0021507, 0.0023509
8: 0.0084021, 0.0105775, 0.0082612, 0.0105128, -0.0017063, 0.0018651
9: 0.0128366, 0.0167492, 0.0125831, 0.0166329, -0.0030689, 0.0033545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007710, upper bound: 0.0007598
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007700, upper bound: 0.0007586
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041540, -0.0041221, -0.0041528, -0.0041190, -0.0000287, 0.0000250
1: -0.0082245, -0.0070292, -0.0081800, -0.0069111, -0.0010759, 0.0009380
2: 0.9665937, 0.9680281, 0.9666471, 0.9681699, -0.0012911, 0.0011256
3: -0.0000936, 0.0104870, 0.0003011, 0.0115323, -0.0095230, 0.0083023
4: -0.0014906, -0.0006859, -0.0015701, -0.0007159, -0.0006314, 0.0007243
5: 0.0157638, 0.0165771, 0.0156834, 0.0165468, -0.0006382, 0.0007320
6: 0.0038437, 0.0042393, 0.0038584, 0.0042784, -0.0003561, 0.0003104
7: -0.0104961, -0.0077540, -0.0107669, -0.0078563, -0.0021516, 0.0024680
8: 0.0084021, 0.0105775, 0.0081872, 0.0104963, -0.0017070, 0.0019580
9: 0.0128366, 0.0167492, 0.0124500, 0.0166033, -0.0030702, 0.0035216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007744, upper bound: 0.0007624
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007700, upper bound: 0.0007586
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041538, -0.0041211, -0.0041531, -0.0041201, -0.0000276, 0.0000266
1: -0.0082149, -0.0069895, -0.0081878, -0.0069518, -0.0010326, 0.0009956
2: 0.9666053, 0.9680758, 0.9666378, 0.9681210, -0.0012391, 0.0011948
3: -0.0000080, 0.0108385, 0.0002320, 0.0111714, -0.0091395, 0.0088124
4: -0.0015174, -0.0006924, -0.0015427, -0.0007107, -0.0006702, 0.0006951
5: 0.0157368, 0.0165705, 0.0157112, 0.0165521, -0.0006774, 0.0007025
6: 0.0038469, 0.0042524, 0.0038559, 0.0042649, -0.0003417, 0.0003295
7: -0.0105871, -0.0077762, -0.0106734, -0.0078384, -0.0022838, 0.0023686
8: 0.0083298, 0.0105599, 0.0082614, 0.0105106, -0.0018119, 0.0018791
9: 0.0127066, 0.0167176, 0.0125834, 0.0166289, -0.0032588, 0.0033798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004180, upper bound: 0.0004343
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007374
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007374
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041536, -0.0041211, -0.0041528, -0.0041196, -0.0000279, 0.0000268
1: -0.0082092, -0.0069896, -0.0081793, -0.0069340, -0.0010433, 0.0010043
2: 0.9666121, 0.9680756, 0.9666479, 0.9681424, -0.0012521, 0.0012052
3: 0.0000426, 0.0108369, 0.0003068, 0.0113295, -0.0092350, 0.0088893
4: -0.0015172, -0.0006963, -0.0015547, -0.0007164, -0.0006761, 0.0007024
5: 0.0157369, 0.0165666, 0.0156990, 0.0165463, -0.0006833, 0.0007099
6: 0.0038488, 0.0042524, 0.0038587, 0.0042708, -0.0003453, 0.0003324
7: -0.0105867, -0.0077893, -0.0107144, -0.0078578, -0.0023037, 0.0023933
8: 0.0083301, 0.0105495, 0.0082289, 0.0104952, -0.0018277, 0.0018988
9: 0.0127072, 0.0166989, 0.0125250, 0.0166012, -0.0032872, 0.0034151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004081, upper bound: 0.0003906
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007344
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007344
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041531, -0.0041200, -0.0041540, -0.0041221, -0.0000250, 0.0000274
1: -0.0081890, -0.0069517, -0.0082245, -0.0070292, -0.0009376, 0.0010248
2: 0.9666364, 0.9681212, 0.9665937, 0.9680281, -0.0011251, 0.0012298
3: 0.0002211, 0.0111724, -0.0000936, 0.0104870, -0.0082987, 0.0090711
4: -0.0015428, -0.0007098, -0.0014906, -0.0006859, -0.0006899, 0.0006312
5: 0.0157111, 0.0165529, 0.0157638, 0.0165771, -0.0006973, 0.0006379
6: 0.0038555, 0.0042649, 0.0038437, 0.0042393, -0.0003103, 0.0003392
7: -0.0106737, -0.0078356, -0.0104961, -0.0077540, -0.0023509, 0.0021507
8: 0.0082612, 0.0105128, 0.0084021, 0.0105775, -0.0018651, 0.0017063
9: 0.0125831, 0.0166329, 0.0128366, 0.0167492, -0.0033545, 0.0030689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007597, upper bound: 0.0007710
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007586, upper bound: 0.0007700
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041528, -0.0041190, -0.0041540, -0.0041221, -0.0000250, 0.0000287
1: -0.0081800, -0.0069111, -0.0082245, -0.0070292, -0.0009380, 0.0010759
2: 0.9666471, 0.9681699, 0.9665937, 0.9680281, -0.0011256, 0.0012911
3: 0.0003011, 0.0115323, -0.0000936, 0.0104870, -0.0083023, 0.0095230
4: -0.0015701, -0.0007159, -0.0014906, -0.0006859, -0.0007243, 0.0006314
5: 0.0156834, 0.0165468, 0.0157638, 0.0165771, -0.0007320, 0.0006382
6: 0.0038584, 0.0042784, 0.0038437, 0.0042393, -0.0003104, 0.0003561
7: -0.0107669, -0.0078563, -0.0104961, -0.0077540, -0.0024680, 0.0021516
8: 0.0081872, 0.0104963, 0.0084021, 0.0105775, -0.0019580, 0.0017070
9: 0.0124500, 0.0166033, 0.0128366, 0.0167492, -0.0035216, 0.0030702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007623, upper bound: 0.0007744
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007586, upper bound: 0.0007700
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041531, -0.0041201, -0.0041538, -0.0041211, -0.0000266, 0.0000276
1: -0.0081878, -0.0069518, -0.0082149, -0.0069895, -0.0009956, 0.0010326
2: 0.9666378, 0.9681210, 0.9666053, 0.9680758, -0.0011948, 0.0012391
3: 0.0002320, 0.0111714, -0.0000080, 0.0108385, -0.0088124, 0.0091395
4: -0.0015427, -0.0007107, -0.0015174, -0.0006924, -0.0006951, 0.0006702
5: 0.0157112, 0.0165521, 0.0157368, 0.0165705, -0.0007025, 0.0006774
6: 0.0038559, 0.0042649, 0.0038469, 0.0042524, -0.0003295, 0.0003417
7: -0.0106734, -0.0078384, -0.0105871, -0.0077762, -0.0023686, 0.0022838
8: 0.0082614, 0.0105106, 0.0083298, 0.0105599, -0.0018791, 0.0018119
9: 0.0125834, 0.0166289, 0.0127066, 0.0167176, -0.0033798, 0.0032588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004343, upper bound: 0.0004180
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007374, upper bound: 0.0007766
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007374, upper bound: 0.0007766
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041528, -0.0041196, -0.0041536, -0.0041211, -0.0000268, 0.0000279
1: -0.0081793, -0.0069340, -0.0082092, -0.0069896, -0.0010043, 0.0010433
2: 0.9666479, 0.9681424, 0.9666121, 0.9680756, -0.0012052, 0.0012521
3: 0.0003068, 0.0113295, 0.0000426, 0.0108369, -0.0088893, 0.0092350
4: -0.0015547, -0.0007164, -0.0015172, -0.0006963, -0.0007024, 0.0006761
5: 0.0156990, 0.0165463, 0.0157369, 0.0165666, -0.0007099, 0.0006833
6: 0.0038587, 0.0042708, 0.0038488, 0.0042524, -0.0003324, 0.0003453
7: -0.0107144, -0.0078578, -0.0105867, -0.0077893, -0.0023933, 0.0023037
8: 0.0082289, 0.0104952, 0.0083301, 0.0105495, -0.0018988, 0.0018277
9: 0.0125250, 0.0166012, 0.0127072, 0.0166989, -0.0034151, 0.0032872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003906, upper bound: 0.0004081
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007766
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007700
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041531, -0.0041200, -0.0041531, -0.0041200, -0.0000251, 0.0000251
1: -0.0081890, -0.0069517, -0.0081890, -0.0069517, -0.0009390, 0.0009390
2: 0.9666364, 0.9681212, 0.9666364, 0.9681212, -0.0011269, 0.0011269
3: 0.0002211, 0.0111724, 0.0002211, 0.0111724, -0.0083116, 0.0083116
4: -0.0015428, -0.0007098, -0.0015428, -0.0007098, -0.0006321, 0.0006321
5: 0.0157111, 0.0165529, 0.0157111, 0.0165529, -0.0006389, 0.0006389
6: 0.0038555, 0.0042649, 0.0038555, 0.0042649, -0.0003108, 0.0003108
7: -0.0106737, -0.0078356, -0.0106737, -0.0078356, -0.0021540, 0.0021540
8: 0.0082612, 0.0105128, 0.0082612, 0.0105128, -0.0017089, 0.0017089
9: 0.0125831, 0.0166329, 0.0125831, 0.0166329, -0.0030736, 0.0030736

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007518, upper bound: 0.0007877
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007544, upper bound: 0.0007877
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041531, -0.0041200, -0.0041528, -0.0041190, -0.0000265, 0.0000253
1: -0.0081890, -0.0069517, -0.0081800, -0.0069111, -0.0009937, 0.0009466
2: 0.9666364, 0.9681212, 0.9666471, 0.9681699, -0.0011925, 0.0011360
3: 0.0002211, 0.0111724, 0.0003011, 0.0115323, -0.0087957, 0.0083787
4: -0.0015428, -0.0007098, -0.0015701, -0.0007159, -0.0006373, 0.0006690
5: 0.0157111, 0.0165529, 0.0156834, 0.0165468, -0.0006441, 0.0006761
6: 0.0038555, 0.0042649, 0.0038584, 0.0042784, -0.0003289, 0.0003133
7: -0.0106737, -0.0078356, -0.0107669, -0.0078563, -0.0021714, 0.0022795
8: 0.0082612, 0.0105128, 0.0081872, 0.0104963, -0.0017227, 0.0018084
9: 0.0125831, 0.0166329, 0.0124500, 0.0166033, -0.0030984, 0.0032526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007518, upper bound: 0.0007892
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007544, upper bound: 0.0007891
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041528, -0.0041190, -0.0041531, -0.0041201, -0.0000253, 0.0000267
1: -0.0081800, -0.0069111, -0.0081878, -0.0069518, -0.0009460, 0.0009984
2: 0.9666471, 0.9681699, 0.9666378, 0.9681210, -0.0011353, 0.0011981
3: 0.0003011, 0.0115323, 0.0002320, 0.0111714, -0.0083735, 0.0088372
4: -0.0015701, -0.0007159, -0.0015427, -0.0007107, -0.0006721, 0.0006369
5: 0.0156834, 0.0165468, 0.0157112, 0.0165521, -0.0006793, 0.0006437
6: 0.0038584, 0.0042784, 0.0038559, 0.0042649, -0.0003131, 0.0003304
7: -0.0107669, -0.0078563, -0.0106734, -0.0078384, -0.0022902, 0.0021701
8: 0.0081872, 0.0104963, 0.0082614, 0.0105106, -0.0018170, 0.0017216
9: 0.0124500, 0.0166033, 0.0125834, 0.0166289, -0.0032680, 0.0030965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005789, upper bound: 0.0005331
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004426
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041527, -0.0041190, -0.0041528, -0.0041196, -0.0000260, 0.0000268
1: -0.0081738, -0.0069113, -0.0081793, -0.0069340, -0.0009725, 0.0010025
2: 0.9666546, 0.9681696, 0.9666479, 0.9681424, -0.0011670, 0.0012031
3: 0.0003555, 0.0115307, 0.0003068, 0.0113295, -0.0086078, 0.0088738
4: -0.0015700, -0.0007201, -0.0015547, -0.0007164, -0.0006749, 0.0006547
5: 0.0156836, 0.0165426, 0.0156990, 0.0165463, -0.0006821, 0.0006617
6: 0.0038605, 0.0042783, 0.0038587, 0.0042708, -0.0003218, 0.0003318
7: -0.0107665, -0.0078704, -0.0107144, -0.0078578, -0.0022997, 0.0022308
8: 0.0081875, 0.0104852, 0.0082289, 0.0104952, -0.0018245, 0.0017698
9: 0.0124506, 0.0165832, 0.0125250, 0.0166012, -0.0032815, 0.0031831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004981, upper bound: 0.0005841
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004116, upper bound: 0.0004191
time: 0.66 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.92 seconds
IS_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007373, upper bound: 0.0007624
IS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007586
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007373, upper bound: 0.0007624
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007586
IS_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007710, upper bound: 0.0007598
IS_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007700, upper bound: 0.0007586
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007744, upper bound: 0.0007624
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007700, upper bound: 0.0007586
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007374
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007374
IS_A1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007344
IS_A1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007344
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007597, upper bound: 0.0007710
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007586, upper bound: 0.0007700
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007623, upper bound: 0.0007744
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007586, upper bound: 0.0007700
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007374, upper bound: 0.0007766
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007374, upper bound: 0.0007766
IS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007766
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007700
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007518, upper bound: 0.0007877
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007544, upper bound: 0.0007877
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007518, upper bound: 0.0007892
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0007544, upper bound: 0.0007891
IS_A2_B2_A2_B1_B1, status: Status.VERIFIED, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0005789, upper bound: 0.0005331
IS_A2_B2_A2_B1_B2, status: Status.VERIFIED, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0004152, upper bound: 0.0004426
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0004981, upper bound: 0.0005841
IS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.92
Output dim: 2, lower bound: -0.0004116, upper bound: 0.0004191

## BFS IS instance: IS_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041540, -0.0041221, -0.0041526, -0.0041201, -0.0000273, 0.0000246
1: -0.0082245, -0.0070292, -0.0081690, -0.0069527, -0.0010237, 0.0009199
2: 0.9665937, 0.9680281, 0.9666603, 0.9681199, -0.0012285, 0.0011040
3: -0.0000936, 0.0104870, 0.0003983, 0.0111636, -0.0090608, 0.0081427
4: -0.0014906, -0.0006859, -0.0015421, -0.0007233, -0.0006193, 0.0006891
5: 0.0157638, 0.0165771, 0.0157118, 0.0165393, -0.0006259, 0.0006965
6: 0.0038437, 0.0042393, 0.0038621, 0.0042646, -0.0003388, 0.0003044
7: -0.0104961, -0.0077540, -0.0106714, -0.0078815, -0.0021103, 0.0023482
8: 0.0084021, 0.0105775, 0.0082630, 0.0104763, -0.0016742, 0.0018629
9: 0.0128366, 0.0167492, 0.0125863, 0.0165673, -0.0030112, 0.0033507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007943, upper bound: 0.0007725
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007943, upper bound: 0.0007725
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041539, -0.0041221, -0.0041523, -0.0041196, -0.0000276, 0.0000248
1: -0.0082188, -0.0070293, -0.0081611, -0.0069349, -0.0010346, 0.0009288
2: 0.9666005, 0.9680279, 0.9666699, 0.9681413, -0.0012416, 0.0011146
3: -0.0000429, 0.0104855, 0.0004683, 0.0113214, -0.0091579, 0.0082209
4: -0.0014905, -0.0006898, -0.0015541, -0.0007286, -0.0006252, 0.0006965
5: 0.0157639, 0.0165732, 0.0156997, 0.0165339, -0.0006319, 0.0007040
6: 0.0038456, 0.0042392, 0.0038647, 0.0042705, -0.0003424, 0.0003074
7: -0.0104957, -0.0077671, -0.0107123, -0.0078996, -0.0021305, 0.0023734
8: 0.0084024, 0.0105671, 0.0082305, 0.0104620, -0.0016903, 0.0018829
9: 0.0128371, 0.0167305, 0.0125280, 0.0165415, -0.0030401, 0.0033866

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005778, upper bound: 0.0005789
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004317, upper bound: 0.0003640
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041535, -0.0041221, -0.0041528, -0.0041190, -0.0000282, 0.0000250
1: -0.0082029, -0.0070302, -0.0081800, -0.0069111, -0.0010560, 0.0009368
2: 0.9666197, 0.9680269, 0.9666471, 0.9681699, -0.0012673, 0.0011242
3: 0.0000982, 0.0104784, 0.0003011, 0.0115323, -0.0093471, 0.0082922
4: -0.0014900, -0.0007005, -0.0015701, -0.0007159, -0.0006307, 0.0007109
5: 0.0157645, 0.0165624, 0.0156834, 0.0165468, -0.0006374, 0.0007185
6: 0.0038509, 0.0042390, 0.0038584, 0.0042784, -0.0003495, 0.0003100
7: -0.0104938, -0.0078037, -0.0107669, -0.0078563, -0.0021490, 0.0024224
8: 0.0084039, 0.0105380, 0.0081872, 0.0104963, -0.0017049, 0.0019218
9: 0.0128398, 0.0166783, 0.0124500, 0.0166033, -0.0030664, 0.0034565

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005154, upper bound: 0.0004296
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004691, upper bound: 0.0004899
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007700, upper bound: 0.0007585
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007700, upper bound: 0.0007586
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041533, -0.0041215, -0.0041527, -0.0041190, -0.0000284, 0.0000256
1: -0.0081980, -0.0070063, -0.0081738, -0.0069113, -0.0010650, 0.0009604
2: 0.9666255, 0.9680556, 0.9666546, 0.9681696, -0.0012781, 0.0011526
3: 0.0001410, 0.0106899, 0.0003555, 0.0115307, -0.0094268, 0.0085011
4: -0.0015061, -0.0007038, -0.0015700, -0.0007201, -0.0006466, 0.0007170
5: 0.0157482, 0.0165591, 0.0156836, 0.0165426, -0.0006535, 0.0007246
6: 0.0038525, 0.0042469, 0.0038605, 0.0042783, -0.0003525, 0.0003178
7: -0.0105486, -0.0078148, -0.0107665, -0.0078704, -0.0022031, 0.0024430
8: 0.0083604, 0.0105293, 0.0081875, 0.0104852, -0.0017479, 0.0019382
9: 0.0127615, 0.0166625, 0.0124506, 0.0165832, -0.0031437, 0.0034860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003691, upper bound: 0.0003200
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004240, upper bound: 0.0004357
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007700, upper bound: 0.0007585
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007700, upper bound: 0.0007586
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041532, -0.0041211, -0.0041531, -0.0041201, -0.0000270, 0.0000266
1: -0.0081935, -0.0069904, -0.0081878, -0.0069518, -0.0010127, 0.0009945
2: 0.9666309, 0.9680746, 0.9666378, 0.9681210, -0.0012153, 0.0011934
3: 0.0001809, 0.0108300, 0.0002320, 0.0111714, -0.0089639, 0.0088025
4: -0.0015167, -0.0007068, -0.0015427, -0.0007107, -0.0006695, 0.0006818
5: 0.0157374, 0.0165560, 0.0157112, 0.0165521, -0.0006766, 0.0006890
6: 0.0038539, 0.0042521, 0.0038559, 0.0042649, -0.0003351, 0.0003291
7: -0.0105849, -0.0078251, -0.0106734, -0.0078384, -0.0022812, 0.0023231
8: 0.0083316, 0.0105211, 0.0082614, 0.0105106, -0.0018098, 0.0018430
9: 0.0127097, 0.0166477, 0.0125834, 0.0166289, -0.0032552, 0.0033149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004180, upper bound: 0.0004343
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007779, upper bound: 0.0007373
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007779, upper bound: 0.0007373
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041531, -0.0041204, -0.0041531, -0.0041201, -0.0000270, 0.0000273
1: -0.0081884, -0.0069661, -0.0081878, -0.0069518, -0.0010103, 0.0010229
2: 0.9666371, 0.9681039, 0.9666378, 0.9681210, -0.0012124, 0.0012276
3: 0.0002267, 0.0110455, 0.0002320, 0.0111714, -0.0089426, 0.0090542
4: -0.0015331, -0.0007103, -0.0015427, -0.0007107, -0.0006886, 0.0006801
5: 0.0157209, 0.0165525, 0.0157112, 0.0165521, -0.0006960, 0.0006874
6: 0.0038557, 0.0042602, 0.0038559, 0.0042649, -0.0003344, 0.0003385
7: -0.0106408, -0.0078370, -0.0106734, -0.0078384, -0.0023465, 0.0023176
8: 0.0082873, 0.0105116, 0.0082614, 0.0105106, -0.0018616, 0.0018386
9: 0.0126300, 0.0166308, 0.0125834, 0.0166289, -0.0033482, 0.0033070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004180, upper bound: 0.0004343
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007779, upper bound: 0.0007373
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007779, upper bound: 0.0007373
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0041536, -0.0041211, -0.0041523, -0.0041196, -0.0000278, 0.0000262
1: -0.0082092, -0.0069896, -0.0081611, -0.0069349, -0.0010428, 0.0009822
2: 0.9666121, 0.9680756, 0.9666699, 0.9681413, -0.0012514, 0.0011787
3: 0.0000426, 0.0108369, 0.0004683, 0.0113214, -0.0092300, 0.0086940
4: -0.0015172, -0.0006963, -0.0015541, -0.0007286, -0.0006612, 0.0007020
5: 0.0157369, 0.0165666, 0.0156997, 0.0165339, -0.0006683, 0.0007095
6: 0.0038488, 0.0042524, 0.0038647, 0.0042705, -0.0003451, 0.0003251
7: -0.0105867, -0.0077893, -0.0107123, -0.0078996, -0.0022531, 0.0023920
8: 0.0083301, 0.0105495, 0.0082305, 0.0104620, -0.0017875, 0.0018977
9: 0.0127072, 0.0166989, 0.0125280, 0.0165415, -0.0032150, 0.0034133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004081, upper bound: 0.0003906
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007344
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007344
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0041536, -0.0041211, -0.0041521, -0.0041185, -0.0000284, 0.0000255
1: -0.0082092, -0.0069896, -0.0081518, -0.0068947, -0.0010627, 0.0009547
2: 0.9666121, 0.9680756, 0.9666809, 0.9681895, -0.0012752, 0.0011457
3: 0.0000426, 0.0108369, 0.0005499, 0.0116773, -0.0094060, 0.0084503
4: -0.0015172, -0.0006963, -0.0015812, -0.0007349, -0.0006427, 0.0007154
5: 0.0157369, 0.0165666, 0.0156723, 0.0165276, -0.0006496, 0.0007230
6: 0.0038488, 0.0042524, 0.0038677, 0.0042838, -0.0003517, 0.0003159
7: -0.0105867, -0.0077893, -0.0108045, -0.0079208, -0.0021900, 0.0024376
8: 0.0083301, 0.0105495, 0.0081573, 0.0104452, -0.0017374, 0.0019339
9: 0.0127072, 0.0166989, 0.0123964, 0.0165113, -0.0031249, 0.0034783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004081, upper bound: 0.0003906
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007344
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007344
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0041526, -0.0041201, -0.0041540, -0.0041221, -0.0000246, 0.0000273
1: -0.0081690, -0.0069527, -0.0082245, -0.0070292, -0.0009199, 0.0010237
2: 0.9666603, 0.9681199, 0.9665937, 0.9680281, -0.0011040, 0.0012285
3: 0.0003983, 0.0111636, -0.0000936, 0.0104870, -0.0081427, 0.0090608
4: -0.0015421, -0.0007233, -0.0014906, -0.0006859, -0.0006891, 0.0006193
5: 0.0157118, 0.0165393, 0.0157638, 0.0165771, -0.0006965, 0.0006259
6: 0.0038621, 0.0042646, 0.0038437, 0.0042393, -0.0003044, 0.0003388
7: -0.0106714, -0.0078815, -0.0104961, -0.0077540, -0.0023482, 0.0021103
8: 0.0082630, 0.0104763, 0.0084021, 0.0105775, -0.0018629, 0.0016742
9: 0.0125863, 0.0165673, 0.0128366, 0.0167492, -0.0033507, 0.0030112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007725, upper bound: 0.0007944
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007725, upper bound: 0.0007946
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041523, -0.0041196, -0.0041539, -0.0041221, -0.0000248, 0.0000276
1: -0.0081611, -0.0069349, -0.0082188, -0.0070293, -0.0009288, 0.0010346
2: 0.9666699, 0.9681413, 0.9666005, 0.9680279, -0.0011146, 0.0012416
3: 0.0004683, 0.0113214, -0.0000429, 0.0104855, -0.0082209, 0.0091579
4: -0.0015541, -0.0007286, -0.0014905, -0.0006898, -0.0006965, 0.0006252
5: 0.0156997, 0.0165339, 0.0157639, 0.0165732, -0.0007040, 0.0006319
6: 0.0038647, 0.0042705, 0.0038456, 0.0042392, -0.0003074, 0.0003424
7: -0.0107123, -0.0078996, -0.0104957, -0.0077671, -0.0023734, 0.0021305
8: 0.0082305, 0.0104620, 0.0084024, 0.0105671, -0.0018829, 0.0016903
9: 0.0125280, 0.0165415, 0.0128371, 0.0167305, -0.0033866, 0.0030401

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005789, upper bound: 0.0005778
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003640, upper bound: 0.0004317
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041528, -0.0041190, -0.0041535, -0.0041221, -0.0000250, 0.0000282
1: -0.0081800, -0.0069111, -0.0082029, -0.0070302, -0.0009368, 0.0010560
2: 0.9666471, 0.9681699, 0.9666197, 0.9680269, -0.0011242, 0.0012673
3: 0.0003011, 0.0115323, 0.0000982, 0.0104784, -0.0082922, 0.0093471
4: -0.0015701, -0.0007159, -0.0014900, -0.0007005, -0.0007109, 0.0006307
5: 0.0156834, 0.0165468, 0.0157645, 0.0165624, -0.0007185, 0.0006374
6: 0.0038584, 0.0042784, 0.0038509, 0.0042390, -0.0003100, 0.0003495
7: -0.0107669, -0.0078563, -0.0104938, -0.0078037, -0.0024224, 0.0021490
8: 0.0081872, 0.0104963, 0.0084039, 0.0105380, -0.0019218, 0.0017049
9: 0.0124500, 0.0166033, 0.0128398, 0.0166783, -0.0034565, 0.0030664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004296, upper bound: 0.0005154
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004899, upper bound: 0.0004691
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007585, upper bound: 0.0007700
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007585, upper bound: 0.0007700
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041527, -0.0041190, -0.0041533, -0.0041215, -0.0000256, 0.0000284
1: -0.0081738, -0.0069113, -0.0081980, -0.0070063, -0.0009604, 0.0010650
2: 0.9666546, 0.9681696, 0.9666255, 0.9680556, -0.0011526, 0.0012781
3: 0.0003555, 0.0115307, 0.0001410, 0.0106899, -0.0085011, 0.0094268
4: -0.0015700, -0.0007201, -0.0015061, -0.0007038, -0.0007170, 0.0006466
5: 0.0156836, 0.0165426, 0.0157482, 0.0165591, -0.0007246, 0.0006535
6: 0.0038605, 0.0042783, 0.0038525, 0.0042469, -0.0003178, 0.0003525
7: -0.0107665, -0.0078704, -0.0105486, -0.0078148, -0.0024430, 0.0022031
8: 0.0081875, 0.0104852, 0.0083604, 0.0105293, -0.0019382, 0.0017479
9: 0.0124506, 0.0165832, 0.0127615, 0.0166625, -0.0034860, 0.0031437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003200, upper bound: 0.0003691
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004240
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007585, upper bound: 0.0007700
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007585, upper bound: 0.0007700
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041531, -0.0041201, -0.0041532, -0.0041211, -0.0000266, 0.0000270
1: -0.0081878, -0.0069518, -0.0081935, -0.0069904, -0.0009945, 0.0010127
2: 0.9666378, 0.9681210, 0.9666309, 0.9680746, -0.0011934, 0.0012153
3: 0.0002320, 0.0111714, 0.0001809, 0.0108300, -0.0088025, 0.0089639
4: -0.0015427, -0.0007107, -0.0015167, -0.0007068, -0.0006818, 0.0006695
5: 0.0157112, 0.0165521, 0.0157374, 0.0165560, -0.0006890, 0.0006766
6: 0.0038559, 0.0042649, 0.0038539, 0.0042521, -0.0003291, 0.0003351
7: -0.0106734, -0.0078384, -0.0105849, -0.0078251, -0.0023231, 0.0022812
8: 0.0082614, 0.0105106, 0.0083316, 0.0105211, -0.0018430, 0.0018098
9: 0.0125834, 0.0166289, 0.0127097, 0.0166477, -0.0033149, 0.0032552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004343, upper bound: 0.0004180
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007373, upper bound: 0.0007779
time: 0.85 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007373, upper bound: 0.0007710
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041531, -0.0041201, -0.0041531, -0.0041204, -0.0000273, 0.0000270
1: -0.0081878, -0.0069518, -0.0081884, -0.0069661, -0.0010229, 0.0010103
2: 0.9666378, 0.9681210, 0.9666371, 0.9681039, -0.0012276, 0.0012124
3: 0.0002320, 0.0111714, 0.0002267, 0.0110455, -0.0090542, 0.0089426
4: -0.0015427, -0.0007107, -0.0015331, -0.0007103, -0.0006801, 0.0006886
5: 0.0157112, 0.0165521, 0.0157209, 0.0165525, -0.0006874, 0.0006960
6: 0.0038559, 0.0042649, 0.0038557, 0.0042602, -0.0003385, 0.0003344
7: -0.0106734, -0.0078384, -0.0106408, -0.0078370, -0.0023176, 0.0023465
8: 0.0082614, 0.0105106, 0.0082873, 0.0105116, -0.0018386, 0.0018616
9: 0.0125834, 0.0166289, 0.0126300, 0.0166308, -0.0033070, 0.0033482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004343, upper bound: 0.0004180
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007373, upper bound: 0.0007779
time: 0.85 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007373, upper bound: 0.0007710
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0041523, -0.0041196, -0.0041536, -0.0041211, -0.0000262, 0.0000278
1: -0.0081611, -0.0069349, -0.0082092, -0.0069896, -0.0009822, 0.0010428
2: 0.9666699, 0.9681413, 0.9666121, 0.9680756, -0.0011787, 0.0012514
3: 0.0004683, 0.0113214, 0.0000426, 0.0108369, -0.0086940, 0.0092300
4: -0.0015541, -0.0007286, -0.0015172, -0.0006963, -0.0007020, 0.0006612
5: 0.0156997, 0.0165339, 0.0157369, 0.0165666, -0.0007095, 0.0006683
6: 0.0038647, 0.0042705, 0.0038488, 0.0042524, -0.0003251, 0.0003451
7: -0.0107123, -0.0078996, -0.0105867, -0.0077893, -0.0023920, 0.0022531
8: 0.0082305, 0.0104620, 0.0083301, 0.0105495, -0.0018977, 0.0017875
9: 0.0125280, 0.0165415, 0.0127072, 0.0166989, -0.0034133, 0.0032150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003906, upper bound: 0.0004081
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_B2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007766
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007766
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0041521, -0.0041185, -0.0041536, -0.0041211, -0.0000255, 0.0000284
1: -0.0081518, -0.0068947, -0.0082092, -0.0069896, -0.0009547, 0.0010627
2: 0.9666809, 0.9681895, 0.9666121, 0.9680756, -0.0011457, 0.0012752
3: 0.0005499, 0.0116773, 0.0000426, 0.0108369, -0.0084503, 0.0094060
4: -0.0015812, -0.0007349, -0.0015172, -0.0006963, -0.0007154, 0.0006427
5: 0.0156723, 0.0165276, 0.0157369, 0.0165666, -0.0007230, 0.0006496
6: 0.0038677, 0.0042838, 0.0038488, 0.0042524, -0.0003159, 0.0003517
7: -0.0108045, -0.0079208, -0.0105867, -0.0077893, -0.0024376, 0.0021900
8: 0.0081573, 0.0104452, 0.0083301, 0.0105495, -0.0019339, 0.0017374
9: 0.0123964, 0.0165113, 0.0127072, 0.0166989, -0.0034783, 0.0031249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003906, upper bound: 0.0004081
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007700
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007700
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041526, -0.0041201, -0.0041531, -0.0041200, -0.0000245, 0.0000250
1: -0.0081690, -0.0069527, -0.0081890, -0.0069517, -0.0009190, 0.0009380
2: 0.9666603, 0.9681199, 0.9666364, 0.9681212, -0.0011028, 0.0011256
3: 0.0003983, 0.0111636, 0.0002211, 0.0111724, -0.0081341, 0.0083025
4: -0.0015421, -0.0007233, -0.0015428, -0.0007098, -0.0006315, 0.0006186
5: 0.0157118, 0.0165393, 0.0157111, 0.0165529, -0.0006382, 0.0006253
6: 0.0038621, 0.0042646, 0.0038555, 0.0042649, -0.0003041, 0.0003104
7: -0.0106714, -0.0078815, -0.0106737, -0.0078356, -0.0021517, 0.0021080
8: 0.0082630, 0.0104763, 0.0082612, 0.0105128, -0.0017070, 0.0016724
9: 0.0125863, 0.0165673, 0.0125831, 0.0166329, -0.0030703, 0.0030080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007739, upper bound: 0.0007945
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007739, upper bound: 0.0007947
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041523, -0.0041196, -0.0041529, -0.0041201, -0.0000247, 0.0000258
1: -0.0081611, -0.0069349, -0.0081829, -0.0069519, -0.0009236, 0.0009646
2: 0.9666699, 0.9681413, 0.9666437, 0.9681209, -0.0011083, 0.0011576
3: 0.0004683, 0.0113214, 0.0002752, 0.0111707, -0.0081747, 0.0085381
4: -0.0015541, -0.0007286, -0.0015426, -0.0007140, -0.0006494, 0.0006217
5: 0.0156997, 0.0165339, 0.0157112, 0.0165488, -0.0006563, 0.0006284
6: 0.0038647, 0.0042705, 0.0038575, 0.0042648, -0.0003056, 0.0003192
7: -0.0107123, -0.0078996, -0.0106733, -0.0078496, -0.0022127, 0.0021186
8: 0.0082305, 0.0104620, 0.0082615, 0.0105017, -0.0017555, 0.0016808
9: 0.0125280, 0.0165415, 0.0125837, 0.0166129, -0.0031574, 0.0030230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006694, upper bound: 0.0006226
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005512, upper bound: 0.0005698
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041526, -0.0041201, -0.0041528, -0.0041190, -0.0000260, 0.0000253
1: -0.0081690, -0.0069527, -0.0081800, -0.0069111, -0.0009737, 0.0009456
2: 0.9666603, 0.9681199, 0.9666471, 0.9681699, -0.0011684, 0.0011347
3: 0.0003983, 0.0111636, 0.0003011, 0.0115323, -0.0086182, 0.0083697
4: -0.0015421, -0.0007233, -0.0015701, -0.0007159, -0.0006366, 0.0006555
5: 0.0157118, 0.0165393, 0.0156834, 0.0165468, -0.0006434, 0.0006625
6: 0.0038621, 0.0042646, 0.0038584, 0.0042784, -0.0003222, 0.0003129
7: -0.0106714, -0.0078815, -0.0107669, -0.0078563, -0.0021691, 0.0022335
8: 0.0082630, 0.0104763, 0.0081872, 0.0104963, -0.0017208, 0.0017719
9: 0.0125863, 0.0165673, 0.0124500, 0.0166033, -0.0030951, 0.0031870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005109, upper bound: 0.0006253
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004385, upper bound: 0.0004867
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041523, -0.0041196, -0.0041527, -0.0041190, -0.0000261, 0.0000260
1: -0.0081611, -0.0069349, -0.0081738, -0.0069113, -0.0009783, 0.0009720
2: 0.9666699, 0.9681413, 0.9666546, 0.9681696, -0.0011739, 0.0011665
3: 0.0004683, 0.0113214, 0.0003555, 0.0115307, -0.0086588, 0.0086038
4: -0.0015541, -0.0007286, -0.0015700, -0.0007201, -0.0006544, 0.0006586
5: 0.0156997, 0.0165339, 0.0156836, 0.0165426, -0.0006614, 0.0006656
6: 0.0038647, 0.0042705, 0.0038605, 0.0042783, -0.0003237, 0.0003217
7: -0.0107123, -0.0078996, -0.0107665, -0.0078704, -0.0022298, 0.0022440
8: 0.0082305, 0.0104620, 0.0081875, 0.0104852, -0.0017690, 0.0017803
9: 0.0125280, 0.0165415, 0.0124506, 0.0165832, -0.0031817, 0.0032020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005841, upper bound: 0.0005750
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004287, upper bound: 0.0004865
time: 0.65 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.90 seconds
IS_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007943, upper bound: 0.0007725
IS_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007943, upper bound: 0.0007725
IS_A1_B2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0005778, upper bound: 0.0005789
IS_A1_B2_A1_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0004317, upper bound: 0.0003640
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007700, upper bound: 0.0007585
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007700, upper bound: 0.0007586
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007700, upper bound: 0.0007585
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007700, upper bound: 0.0007586
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007779, upper bound: 0.0007373
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007779, upper bound: 0.0007373
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007779, upper bound: 0.0007373
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007779, upper bound: 0.0007373
IS_A1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007344
IS_A1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007344
IS_A1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007344
IS_A1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007766, upper bound: 0.0007344
IS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007725, upper bound: 0.0007944
IS_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007725, upper bound: 0.0007946
IS_A2_B1_B1_A1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0005789, upper bound: 0.0005778
IS_A2_B1_B1_A1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0003640, upper bound: 0.0004317
IS_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007585, upper bound: 0.0007700
IS_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007585, upper bound: 0.0007700
IS_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007585, upper bound: 0.0007700
IS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007585, upper bound: 0.0007700
IS_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007373, upper bound: 0.0007779
IS_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007373, upper bound: 0.0007710
IS_A2_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007373, upper bound: 0.0007779
IS_A2_B1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007373, upper bound: 0.0007710
IS_A2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007766
IS_A2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007766
IS_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007700
IS_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007344, upper bound: 0.0007700
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007739, upper bound: 0.0007945
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0007739, upper bound: 0.0007947
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0006694, upper bound: 0.0006226
IS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0005512, upper bound: 0.0005698
IS_A2_B2_A1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0005109, upper bound: 0.0006253
IS_A2_B2_A1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0004385, upper bound: 0.0004867
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0005841, upper bound: 0.0005750
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 2.90
Output dim: 2, lower bound: -0.0004287, upper bound: 0.0004865

## BFS IS instance: IS_A1_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041535, -0.0041221, -0.0041526, -0.0041201, -0.0000268, 0.0000245
1: -0.0082029, -0.0070302, -0.0081690, -0.0069527, -0.0010038, 0.0009188
2: 0.9666197, 0.9680269, 0.9666603, 0.9681199, -0.0012046, 0.0011026
3: 0.0000982, 0.0104784, 0.0003983, 0.0111636, -0.0088849, 0.0081325
4: -0.0014900, -0.0007005, -0.0015421, -0.0007233, -0.0006185, 0.0006757
5: 0.0157645, 0.0165624, 0.0157118, 0.0165393, -0.0006251, 0.0006830
6: 0.0038509, 0.0042390, 0.0038621, 0.0042646, -0.0003322, 0.0003041
7: -0.0104938, -0.0078037, -0.0106714, -0.0078815, -0.0021076, 0.0023026
8: 0.0084039, 0.0105380, 0.0082630, 0.0104763, -0.0016721, 0.0018268
9: 0.0128398, 0.0166783, 0.0125863, 0.0165673, -0.0030074, 0.0032856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005896, upper bound: 0.0004803
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004335, upper bound: 0.0003862
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041533, -0.0041215, -0.0041526, -0.0041201, -0.0000268, 0.0000253
1: -0.0081980, -0.0070063, -0.0081690, -0.0069527, -0.0010033, 0.0009487
2: 0.9666255, 0.9680556, 0.9666603, 0.9681199, -0.0012040, 0.0011385
3: 0.0001410, 0.0106899, 0.0003983, 0.0111636, -0.0088807, 0.0083971
4: -0.0015061, -0.0007038, -0.0015421, -0.0007233, -0.0006386, 0.0006754
5: 0.0157482, 0.0165591, 0.0157118, 0.0165393, -0.0006455, 0.0006826
6: 0.0038525, 0.0042469, 0.0038621, 0.0042646, -0.0003320, 0.0003140
7: -0.0105486, -0.0078148, -0.0106714, -0.0078815, -0.0021762, 0.0023015
8: 0.0083604, 0.0105293, 0.0082630, 0.0104763, -0.0017265, 0.0018259
9: 0.0127615, 0.0166625, 0.0125863, 0.0165673, -0.0031052, 0.0032841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005896, upper bound: 0.0004803
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004335, upper bound: 0.0003862
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041535, -0.0041221, -0.0041523, -0.0041190, -0.0000282, 0.0000245
1: -0.0082029, -0.0070302, -0.0081603, -0.0069121, -0.0010549, 0.0009184
2: 0.9666197, 0.9680269, 0.9666708, 0.9681687, -0.0012659, 0.0011021
3: 0.0000982, 0.0104784, 0.0004750, 0.0115237, -0.0093371, 0.0081287
4: -0.0014900, -0.0007005, -0.0015695, -0.0007292, -0.0006182, 0.0007101
5: 0.0157645, 0.0165624, 0.0156841, 0.0165334, -0.0006248, 0.0007177
6: 0.0038509, 0.0042390, 0.0038649, 0.0042780, -0.0003491, 0.0003039
7: -0.0104938, -0.0078037, -0.0107647, -0.0079014, -0.0021066, 0.0024198
8: 0.0084039, 0.0105380, 0.0081889, 0.0104606, -0.0016713, 0.0019198
9: 0.0128398, 0.0166783, 0.0124532, 0.0165390, -0.0030060, 0.0034529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004691, upper bound: 0.0004899
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005154, upper bound: 0.0004296
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007535, upper bound: 0.0007398
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007630, upper bound: 0.0007494
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041535, -0.0041221, -0.0041521, -0.0041185, -0.0000286, 0.0000243
1: -0.0082029, -0.0070302, -0.0081518, -0.0068947, -0.0010699, 0.0009094
2: 0.9666197, 0.9680269, 0.9666809, 0.9681895, -0.0012840, 0.0010913
3: 0.0000982, 0.0104784, 0.0005499, 0.0116773, -0.0094704, 0.0080493
4: -0.0014900, -0.0007005, -0.0015812, -0.0007349, -0.0006122, 0.0007203
5: 0.0157645, 0.0165624, 0.0156723, 0.0165276, -0.0006187, 0.0007280
6: 0.0038509, 0.0042390, 0.0038677, 0.0042838, -0.0003541, 0.0003009
7: -0.0104938, -0.0078037, -0.0108045, -0.0079208, -0.0020860, 0.0024543
8: 0.0084039, 0.0105380, 0.0081573, 0.0104452, -0.0016550, 0.0019472
9: 0.0128398, 0.0166783, 0.0123964, 0.0165113, -0.0029766, 0.0035021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004691, upper bound: 0.0004899
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005154, upper bound: 0.0004296
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007535, upper bound: 0.0007398
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007630, upper bound: 0.0007494
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041533, -0.0041215, -0.0041523, -0.0041190, -0.0000282, 0.0000253
1: -0.0081980, -0.0070063, -0.0081603, -0.0069121, -0.0010544, 0.0009482
2: 0.9666255, 0.9680556, 0.9666708, 0.9681687, -0.0012653, 0.0011379
3: 0.0001410, 0.0106899, 0.0004750, 0.0115237, -0.0093330, 0.0083932
4: -0.0015061, -0.0007038, -0.0015695, -0.0007292, -0.0006384, 0.0007098
5: 0.0157482, 0.0165591, 0.0156841, 0.0165334, -0.0006452, 0.0007174
6: 0.0038525, 0.0042469, 0.0038649, 0.0042780, -0.0003489, 0.0003138
7: -0.0105486, -0.0078148, -0.0107647, -0.0079014, -0.0021752, 0.0024187
8: 0.0083604, 0.0105293, 0.0081889, 0.0104606, -0.0017257, 0.0019189
9: 0.0127615, 0.0166625, 0.0124532, 0.0165390, -0.0031038, 0.0034513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003691, upper bound: 0.0003200
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004240, upper bound: 0.0004357
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007463, upper bound: 0.0007326
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007586, upper bound: 0.0007451
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041533, -0.0041215, -0.0041521, -0.0041185, -0.0000285, 0.0000250
1: -0.0081980, -0.0070063, -0.0081518, -0.0068947, -0.0010691, 0.0009355
2: 0.9666255, 0.9680556, 0.9666809, 0.9681895, -0.0012830, 0.0011226
3: 0.0001410, 0.0106899, 0.0005499, 0.0116773, -0.0094632, 0.0082800
4: -0.0015061, -0.0007038, -0.0015812, -0.0007349, -0.0006297, 0.0007197
5: 0.0157482, 0.0165591, 0.0156723, 0.0165276, -0.0006365, 0.0007274
6: 0.0038525, 0.0042469, 0.0038677, 0.0042838, -0.0003538, 0.0003096
7: -0.0105486, -0.0078148, -0.0108045, -0.0079208, -0.0021458, 0.0024525
8: 0.0083604, 0.0105293, 0.0081573, 0.0104452, -0.0017024, 0.0019457
9: 0.0127615, 0.0166625, 0.0123964, 0.0165113, -0.0030619, 0.0034995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003691, upper bound: 0.0003200
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004240, upper bound: 0.0004357
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007463, upper bound: 0.0007326
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007586, upper bound: 0.0007451
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041532, -0.0041211, -0.0041526, -0.0041201, -0.0000270, 0.0000260
1: -0.0081935, -0.0069904, -0.0081690, -0.0069527, -0.0010117, 0.0009723
2: 0.9666309, 0.9680746, 0.9666603, 0.9681199, -0.0012140, 0.0011668
3: 0.0001809, 0.0108300, 0.0003983, 0.0111636, -0.0089546, 0.0086059
4: -0.0015167, -0.0007068, -0.0015421, -0.0007233, -0.0006545, 0.0006811
5: 0.0157374, 0.0165560, 0.0157118, 0.0165393, -0.0006615, 0.0006883
6: 0.0038539, 0.0042521, 0.0038621, 0.0042646, -0.0003348, 0.0003218
7: -0.0105849, -0.0078251, -0.0106714, -0.0078815, -0.0022303, 0.0023207
8: 0.0083316, 0.0105211, 0.0082630, 0.0104763, -0.0017694, 0.0018411
9: 0.0127097, 0.0166477, 0.0125863, 0.0165673, -0.0031824, 0.0033114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004949, upper bound: 0.0005148
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007823, upper bound: 0.0007452
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007857, upper bound: 0.0007484
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041532, -0.0041211, -0.0041523, -0.0041190, -0.0000276, 0.0000252
1: -0.0081935, -0.0069904, -0.0081603, -0.0069121, -0.0010328, 0.0009447
2: 0.9666309, 0.9680746, 0.9666708, 0.9681687, -0.0012394, 0.0011337
3: 0.0001809, 0.0108300, 0.0004750, 0.0115237, -0.0091413, 0.0083618
4: -0.0015167, -0.0007068, -0.0015695, -0.0007292, -0.0006360, 0.0006953
5: 0.0157374, 0.0165560, 0.0156841, 0.0165334, -0.0006428, 0.0007027
6: 0.0038539, 0.0042521, 0.0038649, 0.0042780, -0.0003418, 0.0003126
7: -0.0105849, -0.0078251, -0.0107647, -0.0079014, -0.0021670, 0.0023691
8: 0.0083316, 0.0105211, 0.0081889, 0.0104606, -0.0017192, 0.0018795
9: 0.0127097, 0.0166477, 0.0124532, 0.0165390, -0.0030922, 0.0033805

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004949, upper bound: 0.0005148
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007823, upper bound: 0.0007452
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007857, upper bound: 0.0007484
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041531, -0.0041204, -0.0041526, -0.0041201, -0.0000270, 0.0000267
1: -0.0081884, -0.0069661, -0.0081690, -0.0069527, -0.0010096, 0.0010007
2: 0.9666371, 0.9681039, 0.9666603, 0.9681199, -0.0012116, 0.0012009
3: 0.0002267, 0.0110455, 0.0003983, 0.0111636, -0.0089364, 0.0088576
4: -0.0015331, -0.0007103, -0.0015421, -0.0007233, -0.0006737, 0.0006797
5: 0.0157209, 0.0165525, 0.0157118, 0.0165393, -0.0006809, 0.0006869
6: 0.0038557, 0.0042602, 0.0038621, 0.0042646, -0.0003341, 0.0003312
7: -0.0106408, -0.0078370, -0.0106714, -0.0078815, -0.0022955, 0.0023159
8: 0.0082873, 0.0105116, 0.0082630, 0.0104763, -0.0018212, 0.0018374
9: 0.0126300, 0.0166308, 0.0125863, 0.0165673, -0.0032755, 0.0033047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004180, upper bound: 0.0004343
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007543, upper bound: 0.0007106
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007660, upper bound: 0.0007233
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041531, -0.0041204, -0.0041523, -0.0041190, -0.0000275, 0.0000260
1: -0.0081884, -0.0069661, -0.0081603, -0.0069121, -0.0010304, 0.0009736
2: 0.9666371, 0.9681039, 0.9666708, 0.9681687, -0.0012365, 0.0011683
3: 0.0002267, 0.0110455, 0.0004750, 0.0115237, -0.0091200, 0.0086174
4: -0.0015331, -0.0007103, -0.0015695, -0.0007292, -0.0006554, 0.0006936
5: 0.0157209, 0.0165525, 0.0156841, 0.0165334, -0.0006624, 0.0007010
6: 0.0038557, 0.0042602, 0.0038649, 0.0042780, -0.0003410, 0.0003222
7: -0.0106408, -0.0078370, -0.0107647, -0.0079014, -0.0022333, 0.0023635
8: 0.0082873, 0.0105116, 0.0081889, 0.0104606, -0.0017718, 0.0018751
9: 0.0126300, 0.0166308, 0.0124532, 0.0165390, -0.0031867, 0.0033726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004180, upper bound: 0.0004343
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007543, upper bound: 0.0007106
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007660, upper bound: 0.0007233
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041532, -0.0041211, -0.0041523, -0.0041196, -0.0000275, 0.0000257
1: -0.0081935, -0.0069904, -0.0081611, -0.0069349, -0.0010280, 0.0009620
2: 0.9666309, 0.9680746, 0.9666699, 0.9681413, -0.0012336, 0.0011544
3: 0.0001809, 0.0108300, 0.0004683, 0.0113214, -0.0090988, 0.0085150
4: -0.0015167, -0.0007068, -0.0015541, -0.0007286, -0.0006476, 0.0006920
5: 0.0157374, 0.0165560, 0.0156997, 0.0165339, -0.0006545, 0.0006994
6: 0.0038539, 0.0042521, 0.0038647, 0.0042705, -0.0003402, 0.0003184
7: -0.0105849, -0.0078251, -0.0107123, -0.0078996, -0.0022067, 0.0023580
8: 0.0083316, 0.0105211, 0.0082305, 0.0104620, -0.0017507, 0.0018708
9: 0.0127097, 0.0166477, 0.0125280, 0.0165415, -0.0031488, 0.0033647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005011, upper bound: 0.0004531
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007645, upper bound: 0.0007058
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007742, upper bound: 0.0007204
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041531, -0.0041204, -0.0041523, -0.0041196, -0.0000274, 0.0000264
1: -0.0081884, -0.0069661, -0.0081611, -0.0069349, -0.0010250, 0.0009889
2: 0.9666371, 0.9681039, 0.9666699, 0.9681413, -0.0012301, 0.0011867
3: 0.0002267, 0.0110455, 0.0004683, 0.0113214, -0.0090729, 0.0087528
4: -0.0015331, -0.0007103, -0.0015541, -0.0007286, -0.0006657, 0.0006900
5: 0.0157209, 0.0165525, 0.0156997, 0.0165339, -0.0006728, 0.0006974
6: 0.0038557, 0.0042602, 0.0038647, 0.0042705, -0.0003392, 0.0003273
7: -0.0106408, -0.0078370, -0.0107123, -0.0078996, -0.0022684, 0.0023513
8: 0.0082873, 0.0105116, 0.0082305, 0.0104620, -0.0017996, 0.0018654
9: 0.0126300, 0.0166308, 0.0125280, 0.0165415, -0.0032368, 0.0033551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005011, upper bound: 0.0004531
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007645, upper bound: 0.0007058
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007742, upper bound: 0.0007203
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041532, -0.0041211, -0.0041521, -0.0041185, -0.0000280, 0.0000250
1: -0.0081935, -0.0069904, -0.0081518, -0.0068947, -0.0010484, 0.0009351
2: 0.9666309, 0.9680746, 0.9666809, 0.9681895, -0.0012581, 0.0011222
3: 0.0001809, 0.0108300, 0.0005499, 0.0116773, -0.0092797, 0.0082768
4: -0.0015167, -0.0007068, -0.0015812, -0.0007349, -0.0006295, 0.0007058
5: 0.0157374, 0.0165560, 0.0156723, 0.0165276, -0.0006362, 0.0007133
6: 0.0038539, 0.0042521, 0.0038677, 0.0042838, -0.0003470, 0.0003095
7: -0.0105849, -0.0078251, -0.0108045, -0.0079208, -0.0021450, 0.0024049
8: 0.0083316, 0.0105211, 0.0081573, 0.0104452, -0.0017017, 0.0019080
9: 0.0127097, 0.0166477, 0.0123964, 0.0165113, -0.0030608, 0.0034316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004081, upper bound: 0.0003906
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007521, upper bound: 0.0007058
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007644, upper bound: 0.0007203
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041531, -0.0041204, -0.0041521, -0.0041185, -0.0000280, 0.0000257
1: -0.0081884, -0.0069661, -0.0081518, -0.0068947, -0.0010467, 0.0009616
2: 0.9666371, 0.9681039, 0.9666809, 0.9681895, -0.0012561, 0.0011539
3: 0.0002267, 0.0110455, 0.0005499, 0.0116773, -0.0092646, 0.0085112
4: -0.0015331, -0.0007103, -0.0015812, -0.0007349, -0.0006473, 0.0007046
5: 0.0157209, 0.0165525, 0.0156723, 0.0165276, -0.0006542, 0.0007122
6: 0.0038557, 0.0042602, 0.0038677, 0.0042838, -0.0003464, 0.0003182
7: -0.0106408, -0.0078370, -0.0108045, -0.0079208, -0.0022057, 0.0024010
8: 0.0082873, 0.0105116, 0.0081573, 0.0104452, -0.0017499, 0.0019048
9: 0.0126300, 0.0166308, 0.0123964, 0.0165113, -0.0031474, 0.0034260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004081, upper bound: 0.0003906
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007521, upper bound: 0.0007058
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007644, upper bound: 0.0007204
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041526, -0.0041201, -0.0041535, -0.0041221, -0.0000245, 0.0000268
1: -0.0081690, -0.0069527, -0.0082029, -0.0070302, -0.0009188, 0.0010038
2: 0.9666603, 0.9681199, 0.9666197, 0.9680269, -0.0011026, 0.0012046
3: 0.0003983, 0.0111636, 0.0000982, 0.0104784, -0.0081325, 0.0088849
4: -0.0015421, -0.0007233, -0.0014900, -0.0007005, -0.0006757, 0.0006185
5: 0.0157118, 0.0165393, 0.0157645, 0.0165624, -0.0006830, 0.0006251
6: 0.0038621, 0.0042646, 0.0038509, 0.0042390, -0.0003041, 0.0003322
7: -0.0106714, -0.0078815, -0.0104938, -0.0078037, -0.0023026, 0.0021076
8: 0.0082630, 0.0104763, 0.0084039, 0.0105380, -0.0018268, 0.0016721
9: 0.0125863, 0.0165673, 0.0128398, 0.0166783, -0.0032856, 0.0030074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004803, upper bound: 0.0005896
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003862, upper bound: 0.0004335
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041526, -0.0041201, -0.0041533, -0.0041215, -0.0000253, 0.0000268
1: -0.0081690, -0.0069527, -0.0081980, -0.0070063, -0.0009487, 0.0010033
2: 0.9666603, 0.9681199, 0.9666255, 0.9680556, -0.0011385, 0.0012040
3: 0.0003983, 0.0111636, 0.0001410, 0.0106899, -0.0083971, 0.0088807
4: -0.0015421, -0.0007233, -0.0015061, -0.0007038, -0.0006754, 0.0006386
5: 0.0157118, 0.0165393, 0.0157482, 0.0165591, -0.0006826, 0.0006455
6: 0.0038621, 0.0042646, 0.0038525, 0.0042469, -0.0003140, 0.0003320
7: -0.0106714, -0.0078815, -0.0105486, -0.0078148, -0.0023015, 0.0021762
8: 0.0082630, 0.0104763, 0.0083604, 0.0105293, -0.0018259, 0.0017265
9: 0.0125863, 0.0165673, 0.0127615, 0.0166625, -0.0032841, 0.0031052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004803, upper bound: 0.0005896
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003862, upper bound: 0.0004335
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041523, -0.0041190, -0.0041535, -0.0041221, -0.0000245, 0.0000282
1: -0.0081603, -0.0069121, -0.0082029, -0.0070302, -0.0009184, 0.0010549
2: 0.9666708, 0.9681687, 0.9666197, 0.9680269, -0.0011021, 0.0012659
3: 0.0004750, 0.0115237, 0.0000982, 0.0104784, -0.0081287, 0.0093371
4: -0.0015695, -0.0007292, -0.0014900, -0.0007005, -0.0007101, 0.0006182
5: 0.0156841, 0.0165334, 0.0157645, 0.0165624, -0.0007177, 0.0006248
6: 0.0038649, 0.0042780, 0.0038509, 0.0042390, -0.0003039, 0.0003491
7: -0.0107647, -0.0079014, -0.0104938, -0.0078037, -0.0024198, 0.0021066
8: 0.0081889, 0.0104606, 0.0084039, 0.0105380, -0.0019198, 0.0016713
9: 0.0124532, 0.0165390, 0.0128398, 0.0166783, -0.0034529, 0.0030060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004899, upper bound: 0.0004691
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004296, upper bound: 0.0005154
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007397, upper bound: 0.0007535
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007494, upper bound: 0.0007630
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041521, -0.0041185, -0.0041535, -0.0041221, -0.0000243, 0.0000286
1: -0.0081518, -0.0068947, -0.0082029, -0.0070302, -0.0009094, 0.0010699
2: 0.9666809, 0.9681895, 0.9666197, 0.9680269, -0.0010913, 0.0012840
3: 0.0005499, 0.0116773, 0.0000982, 0.0104784, -0.0080493, 0.0094704
4: -0.0015812, -0.0007349, -0.0014900, -0.0007005, -0.0007203, 0.0006122
5: 0.0156723, 0.0165276, 0.0157645, 0.0165624, -0.0007280, 0.0006187
6: 0.0038677, 0.0042838, 0.0038509, 0.0042390, -0.0003009, 0.0003541
7: -0.0108045, -0.0079208, -0.0104938, -0.0078037, -0.0024543, 0.0020860
8: 0.0081573, 0.0104452, 0.0084039, 0.0105380, -0.0019472, 0.0016550
9: 0.0123964, 0.0165113, 0.0128398, 0.0166783, -0.0035021, 0.0029766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004899, upper bound: 0.0004691
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004296, upper bound: 0.0005154
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007397, upper bound: 0.0007535
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007494, upper bound: 0.0007630
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041523, -0.0041190, -0.0041533, -0.0041215, -0.0000253, 0.0000282
1: -0.0081603, -0.0069121, -0.0081980, -0.0070063, -0.0009482, 0.0010544
2: 0.9666708, 0.9681687, 0.9666255, 0.9680556, -0.0011379, 0.0012653
3: 0.0004750, 0.0115237, 0.0001410, 0.0106899, -0.0083932, 0.0093330
4: -0.0015695, -0.0007292, -0.0015061, -0.0007038, -0.0007098, 0.0006384
5: 0.0156841, 0.0165334, 0.0157482, 0.0165591, -0.0007174, 0.0006452
6: 0.0038649, 0.0042780, 0.0038525, 0.0042469, -0.0003138, 0.0003489
7: -0.0107647, -0.0079014, -0.0105486, -0.0078148, -0.0024187, 0.0021752
8: 0.0081889, 0.0104606, 0.0083604, 0.0105293, -0.0019189, 0.0017257
9: 0.0124532, 0.0165390, 0.0127615, 0.0166625, -0.0034513, 0.0031038

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003200, upper bound: 0.0003691
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004240
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007326, upper bound: 0.0007463
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007451, upper bound: 0.0007586
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041521, -0.0041185, -0.0041533, -0.0041215, -0.0000250, 0.0000285
1: -0.0081518, -0.0068947, -0.0081980, -0.0070063, -0.0009355, 0.0010691
2: 0.9666809, 0.9681895, 0.9666255, 0.9680556, -0.0011226, 0.0012830
3: 0.0005499, 0.0116773, 0.0001410, 0.0106899, -0.0082800, 0.0094632
4: -0.0015812, -0.0007349, -0.0015061, -0.0007038, -0.0007197, 0.0006297
5: 0.0156723, 0.0165276, 0.0157482, 0.0165591, -0.0007274, 0.0006365
6: 0.0038677, 0.0042838, 0.0038525, 0.0042469, -0.0003096, 0.0003538
7: -0.0108045, -0.0079208, -0.0105486, -0.0078148, -0.0024525, 0.0021458
8: 0.0081573, 0.0104452, 0.0083604, 0.0105293, -0.0019457, 0.0017024
9: 0.0123964, 0.0165113, 0.0127615, 0.0166625, -0.0034995, 0.0030619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003200, upper bound: 0.0003691
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004357, upper bound: 0.0004240
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007326, upper bound: 0.0007463
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007451, upper bound: 0.0007586
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041526, -0.0041201, -0.0041532, -0.0041211, -0.0000260, 0.0000270
1: -0.0081690, -0.0069527, -0.0081935, -0.0069904, -0.0009723, 0.0010117
2: 0.9666603, 0.9681199, 0.9666309, 0.9680746, -0.0011668, 0.0012140
3: 0.0003983, 0.0111636, 0.0001809, 0.0108300, -0.0086059, 0.0089546
4: -0.0015421, -0.0007233, -0.0015167, -0.0007068, -0.0006811, 0.0006545
5: 0.0157118, 0.0165393, 0.0157374, 0.0165560, -0.0006883, 0.0006615
6: 0.0038621, 0.0042646, 0.0038539, 0.0042521, -0.0003218, 0.0003348
7: -0.0106714, -0.0078815, -0.0105849, -0.0078251, -0.0023207, 0.0022303
8: 0.0082630, 0.0104763, 0.0083316, 0.0105211, -0.0018411, 0.0017694
9: 0.0125863, 0.0165673, 0.0127097, 0.0166477, -0.0033114, 0.0031824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005148, upper bound: 0.0004949
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007452, upper bound: 0.0007823
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007485, upper bound: 0.0007857
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041523, -0.0041190, -0.0041532, -0.0041211, -0.0000252, 0.0000276
1: -0.0081603, -0.0069121, -0.0081935, -0.0069904, -0.0009447, 0.0010328
2: 0.9666708, 0.9681687, 0.9666309, 0.9680746, -0.0011337, 0.0012394
3: 0.0004750, 0.0115237, 0.0001809, 0.0108300, -0.0083618, 0.0091413
4: -0.0015695, -0.0007292, -0.0015167, -0.0007068, -0.0006953, 0.0006360
5: 0.0156841, 0.0165334, 0.0157374, 0.0165560, -0.0007027, 0.0006428
6: 0.0038649, 0.0042780, 0.0038539, 0.0042521, -0.0003126, 0.0003418
7: -0.0107647, -0.0079014, -0.0105849, -0.0078251, -0.0023691, 0.0021670
8: 0.0081889, 0.0104606, 0.0083316, 0.0105211, -0.0018795, 0.0017192
9: 0.0124532, 0.0165390, 0.0127097, 0.0166477, -0.0033805, 0.0030922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005148, upper bound: 0.0004949
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007452, upper bound: 0.0007720
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007485, upper bound: 0.0007734
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041526, -0.0041201, -0.0041531, -0.0041204, -0.0000267, 0.0000270
1: -0.0081690, -0.0069527, -0.0081884, -0.0069661, -0.0010007, 0.0010096
2: 0.9666603, 0.9681199, 0.9666371, 0.9681039, -0.0012009, 0.0012116
3: 0.0003983, 0.0111636, 0.0002267, 0.0110455, -0.0088576, 0.0089364
4: -0.0015421, -0.0007233, -0.0015331, -0.0007103, -0.0006797, 0.0006737
5: 0.0157118, 0.0165393, 0.0157209, 0.0165525, -0.0006869, 0.0006809
6: 0.0038621, 0.0042646, 0.0038557, 0.0042602, -0.0003312, 0.0003341
7: -0.0106714, -0.0078815, -0.0106408, -0.0078370, -0.0023159, 0.0022955
8: 0.0082630, 0.0104763, 0.0082873, 0.0105116, -0.0018374, 0.0018212
9: 0.0125863, 0.0165673, 0.0126300, 0.0166308, -0.0033047, 0.0032755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004343, upper bound: 0.0004180
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007106, upper bound: 0.0007543
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007233, upper bound: 0.0007660
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041523, -0.0041190, -0.0041531, -0.0041204, -0.0000260, 0.0000275
1: -0.0081603, -0.0069121, -0.0081884, -0.0069661, -0.0009736, 0.0010304
2: 0.9666708, 0.9681687, 0.9666371, 0.9681039, -0.0011683, 0.0012365
3: 0.0004750, 0.0115237, 0.0002267, 0.0110455, -0.0086174, 0.0091200
4: -0.0015695, -0.0007292, -0.0015331, -0.0007103, -0.0006936, 0.0006554
5: 0.0156841, 0.0165334, 0.0157209, 0.0165525, -0.0007010, 0.0006624
6: 0.0038649, 0.0042780, 0.0038557, 0.0042602, -0.0003222, 0.0003410
7: -0.0107647, -0.0079014, -0.0106408, -0.0078370, -0.0023635, 0.0022333
8: 0.0081889, 0.0104606, 0.0082873, 0.0105116, -0.0018751, 0.0017718
9: 0.0124532, 0.0165390, 0.0126300, 0.0166308, -0.0033726, 0.0031867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004343, upper bound: 0.0004180
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007106, upper bound: 0.0007491
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007233, upper bound: 0.0007594
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041523, -0.0041196, -0.0041532, -0.0041211, -0.0000257, 0.0000275
1: -0.0081611, -0.0069349, -0.0081935, -0.0069904, -0.0009620, 0.0010280
2: 0.9666699, 0.9681413, 0.9666309, 0.9680746, -0.0011544, 0.0012336
3: 0.0004683, 0.0113214, 0.0001809, 0.0108300, -0.0085150, 0.0090988
4: -0.0015541, -0.0007286, -0.0015167, -0.0007068, -0.0006920, 0.0006476
5: 0.0156997, 0.0165339, 0.0157374, 0.0165560, -0.0006994, 0.0006545
6: 0.0038647, 0.0042705, 0.0038539, 0.0042521, -0.0003184, 0.0003402
7: -0.0107123, -0.0078996, -0.0105849, -0.0078251, -0.0023580, 0.0022067
8: 0.0082305, 0.0104620, 0.0083316, 0.0105211, -0.0018708, 0.0017507
9: 0.0125280, 0.0165415, 0.0127097, 0.0166477, -0.0033647, 0.0031488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004531, upper bound: 0.0005011
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007058, upper bound: 0.0007645
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007204, upper bound: 0.0007742
time: 0.99 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041523, -0.0041196, -0.0041531, -0.0041204, -0.0000264, 0.0000274
1: -0.0081611, -0.0069349, -0.0081884, -0.0069661, -0.0009889, 0.0010250
2: 0.9666699, 0.9681413, 0.9666371, 0.9681039, -0.0011867, 0.0012301
3: 0.0004683, 0.0113214, 0.0002267, 0.0110455, -0.0087528, 0.0090729
4: -0.0015541, -0.0007286, -0.0015331, -0.0007103, -0.0006900, 0.0006657
5: 0.0156997, 0.0165339, 0.0157209, 0.0165525, -0.0006974, 0.0006728
6: 0.0038647, 0.0042705, 0.0038557, 0.0042602, -0.0003273, 0.0003392
7: -0.0107123, -0.0078996, -0.0106408, -0.0078370, -0.0023513, 0.0022684
8: 0.0082305, 0.0104620, 0.0082873, 0.0105116, -0.0018654, 0.0017996
9: 0.0125280, 0.0165415, 0.0126300, 0.0166308, -0.0033551, 0.0032368

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004531, upper bound: 0.0005011
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007058, upper bound: 0.0007645
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007204, upper bound: 0.0007742
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041521, -0.0041185, -0.0041532, -0.0041211, -0.0000250, 0.0000280
1: -0.0081518, -0.0068947, -0.0081935, -0.0069904, -0.0009351, 0.0010484
2: 0.9666809, 0.9681895, 0.9666309, 0.9680746, -0.0011222, 0.0012581
3: 0.0005499, 0.0116773, 0.0001809, 0.0108300, -0.0082768, 0.0092797
4: -0.0015812, -0.0007349, -0.0015167, -0.0007068, -0.0007058, 0.0006295
5: 0.0156723, 0.0165276, 0.0157374, 0.0165560, -0.0007133, 0.0006362
6: 0.0038677, 0.0042838, 0.0038539, 0.0042521, -0.0003095, 0.0003470
7: -0.0108045, -0.0079208, -0.0105849, -0.0078251, -0.0024049, 0.0021450
8: 0.0081573, 0.0104452, 0.0083316, 0.0105211, -0.0019080, 0.0017017
9: 0.0123964, 0.0165113, 0.0127097, 0.0166477, -0.0034316, 0.0030608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003906, upper bound: 0.0004081
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007084, upper bound: 0.0007463
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007235, upper bound: 0.0007586
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041521, -0.0041185, -0.0041531, -0.0041204, -0.0000257, 0.0000280
1: -0.0081518, -0.0068947, -0.0081884, -0.0069661, -0.0009616, 0.0010467
2: 0.9666809, 0.9681895, 0.9666371, 0.9681039, -0.0011539, 0.0012561
3: 0.0005499, 0.0116773, 0.0002267, 0.0110455, -0.0085112, 0.0092646
4: -0.0015812, -0.0007349, -0.0015331, -0.0007103, -0.0007046, 0.0006473
5: 0.0156723, 0.0165276, 0.0157209, 0.0165525, -0.0007122, 0.0006542
6: 0.0038677, 0.0042838, 0.0038557, 0.0042602, -0.0003182, 0.0003464
7: -0.0108045, -0.0079208, -0.0106408, -0.0078370, -0.0024010, 0.0022057
8: 0.0081573, 0.0104452, 0.0082873, 0.0105116, -0.0019048, 0.0017499
9: 0.0123964, 0.0165113, 0.0126300, 0.0166308, -0.0034260, 0.0031474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003906, upper bound: 0.0004081
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007084, upper bound: 0.0007463
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007235, upper bound: 0.0007586
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041526, -0.0041201, -0.0041526, -0.0041201, -0.0000245, 0.0000245
1: -0.0081690, -0.0069527, -0.0081690, -0.0069527, -0.0009179, 0.0009179
2: 0.9666603, 0.9681199, 0.9666603, 0.9681199, -0.0011016, 0.0011016
3: 0.0003983, 0.0111636, 0.0003983, 0.0111636, -0.0081250, 0.0081250
4: -0.0015421, -0.0007233, -0.0015421, -0.0007233, -0.0006180, 0.0006180
5: 0.0157118, 0.0165393, 0.0157118, 0.0165393, -0.0006246, 0.0006246
6: 0.0038621, 0.0042646, 0.0038621, 0.0042646, -0.0003038, 0.0003038
7: -0.0106714, -0.0078815, -0.0106714, -0.0078815, -0.0021057, 0.0021057
8: 0.0082630, 0.0104763, 0.0082630, 0.0104763, -0.0016705, 0.0016705
9: 0.0125863, 0.0165673, 0.0125863, 0.0165673, -0.0030046, 0.0030046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006010, upper bound: 0.0006858
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005557, upper bound: 0.0005698
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041526, -0.0041201, -0.0041523, -0.0041196, -0.0000254, 0.0000245
1: -0.0081690, -0.0069527, -0.0081611, -0.0069349, -0.0009499, 0.0009174
2: 0.9666603, 0.9681199, 0.9666699, 0.9681413, -0.0011399, 0.0011009
3: 0.0003983, 0.0111636, 0.0004683, 0.0113214, -0.0084075, 0.0081199
4: -0.0015421, -0.0007233, -0.0015541, -0.0007286, -0.0006176, 0.0006394
5: 0.0157118, 0.0165393, 0.0156997, 0.0165339, -0.0006242, 0.0006463
6: 0.0038621, 0.0042646, 0.0038647, 0.0042705, -0.0003143, 0.0003036
7: -0.0106714, -0.0078815, -0.0107123, -0.0078996, -0.0021043, 0.0021789
8: 0.0082630, 0.0104763, 0.0082305, 0.0104620, -0.0016695, 0.0017286
9: 0.0125863, 0.0165673, 0.0125280, 0.0165415, -0.0030027, 0.0031091

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006010, upper bound: 0.0006858
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005557, upper bound: 0.0005698
time: 0.69 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.03 seconds
IS_A1_B2_A1_B1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0005896, upper bound: 0.0004803
IS_A1_B2_A1_B1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0004335, upper bound: 0.0003862
IS_A1_B2_A1_B1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0005896, upper bound: 0.0004803
IS_A1_B2_A1_B1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0004335, upper bound: 0.0003862
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007535, upper bound: 0.0007398
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007630, upper bound: 0.0007494
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007535, upper bound: 0.0007398
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007630, upper bound: 0.0007494
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007463, upper bound: 0.0007326
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007586, upper bound: 0.0007451
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007463, upper bound: 0.0007326
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007586, upper bound: 0.0007451
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007823, upper bound: 0.0007452
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007857, upper bound: 0.0007484
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007823, upper bound: 0.0007452
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007857, upper bound: 0.0007484
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007543, upper bound: 0.0007106
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007660, upper bound: 0.0007233
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007543, upper bound: 0.0007106
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007660, upper bound: 0.0007233
IS_A1_B2_A2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007645, upper bound: 0.0007058
IS_A1_B2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007742, upper bound: 0.0007204
IS_A1_B2_A2_B2_B1_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007645, upper bound: 0.0007058
IS_A1_B2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007742, upper bound: 0.0007203
IS_A1_B2_A2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007521, upper bound: 0.0007058
IS_A1_B2_A2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007644, upper bound: 0.0007203
IS_A1_B2_A2_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007521, upper bound: 0.0007058
IS_A1_B2_A2_B2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007644, upper bound: 0.0007204
IS_A2_B1_B1_A1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0004803, upper bound: 0.0005896
IS_A2_B1_B1_A1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0003862, upper bound: 0.0004335
IS_A2_B1_B1_A1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0004803, upper bound: 0.0005896
IS_A2_B1_B1_A1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0003862, upper bound: 0.0004335
IS_A2_B1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007397, upper bound: 0.0007535
IS_A2_B1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007494, upper bound: 0.0007630
IS_A2_B1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007397, upper bound: 0.0007535
IS_A2_B1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007494, upper bound: 0.0007630
IS_A2_B1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007326, upper bound: 0.0007463
IS_A2_B1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007451, upper bound: 0.0007586
IS_A2_B1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007326, upper bound: 0.0007463
IS_A2_B1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007451, upper bound: 0.0007586
IS_A2_B1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007452, upper bound: 0.0007823
IS_A2_B1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007485, upper bound: 0.0007857
IS_A2_B1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007452, upper bound: 0.0007720
IS_A2_B1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007485, upper bound: 0.0007734
IS_A2_B1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007106, upper bound: 0.0007543
IS_A2_B1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007233, upper bound: 0.0007660
IS_A2_B1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007106, upper bound: 0.0007491
IS_A2_B1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007233, upper bound: 0.0007594
IS_A2_B1_B2_A2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007058, upper bound: 0.0007645
IS_A2_B1_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007204, upper bound: 0.0007742
IS_A2_B1_B2_A2_A1_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007058, upper bound: 0.0007645
IS_A2_B1_B2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007204, upper bound: 0.0007742
IS_A2_B1_B2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007084, upper bound: 0.0007463
IS_A2_B1_B2_A2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007235, upper bound: 0.0007586
IS_A2_B1_B2_A2_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007084, upper bound: 0.0007463
IS_A2_B1_B2_A2_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0007235, upper bound: 0.0007586
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0006010, upper bound: 0.0006858
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0005557, upper bound: 0.0005698
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0006010, upper bound: 0.0006858
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.03
Output dim: 2, lower bound: -0.0005557, upper bound: 0.0005698

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041534, -0.0041214, -0.0041525, -0.0041201, -0.0000267, 0.0000254
1: -0.0082006, -0.0070032, -0.0081681, -0.0069554, -0.0009999, 0.0009493
2: 0.9666225, 0.9680593, 0.9666615, 0.9681166, -0.0011999, 0.0011392
3: 0.0001186, 0.0107168, 0.0004060, 0.0111400, -0.0088505, 0.0084027
4: -0.0015081, -0.0007021, -0.0015403, -0.0007239, -0.0006391, 0.0006731
5: 0.0157461, 0.0165608, 0.0157136, 0.0165387, -0.0006459, 0.0006803
6: 0.0038516, 0.0042479, 0.0038624, 0.0042637, -0.0003309, 0.0003142
7: -0.0105556, -0.0078090, -0.0106653, -0.0078835, -0.0021776, 0.0022937
8: 0.0083548, 0.0105339, 0.0082678, 0.0104748, -0.0017276, 0.0018197
9: 0.0127516, 0.0166708, 0.0125951, 0.0165645, -0.0031073, 0.0032729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005410, upper bound: 0.0004978
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007838, upper bound: 0.0007450
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007838, upper bound: 0.0007452
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041532, -0.0041213, -0.0041526, -0.0041201, -0.0000268, 0.0000252
1: -0.0081914, -0.0069980, -0.0081690, -0.0069527, -0.0010049, 0.0009435
2: 0.9666334, 0.9680656, 0.9666603, 0.9681199, -0.0012059, 0.0011323
3: 0.0001996, 0.0107629, 0.0003983, 0.0111636, -0.0088946, 0.0083515
4: -0.0015116, -0.0007082, -0.0015421, -0.0007233, -0.0006352, 0.0006765
5: 0.0157426, 0.0165546, 0.0157118, 0.0165393, -0.0006420, 0.0006837
6: 0.0038546, 0.0042496, 0.0038621, 0.0042646, -0.0003326, 0.0003123
7: -0.0105675, -0.0078300, -0.0106714, -0.0078815, -0.0021644, 0.0023051
8: 0.0083454, 0.0105172, 0.0082630, 0.0104763, -0.0017171, 0.0018288
9: 0.0127345, 0.0166408, 0.0125863, 0.0165673, -0.0030884, 0.0032892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005525, upper bound: 0.0005229
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007838, upper bound: 0.0007456
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007838, upper bound: 0.0007484
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041534, -0.0041214, -0.0041523, -0.0041191, -0.0000272, 0.0000246
1: -0.0082006, -0.0070032, -0.0081594, -0.0069148, -0.0010190, 0.0009219
2: 0.9666225, 0.9680593, 0.9666719, 0.9681653, -0.0012228, 0.0011064
3: 0.0001186, 0.0107168, 0.0004828, 0.0114991, -0.0090192, 0.0081603
4: -0.0015081, -0.0007021, -0.0015676, -0.0007297, -0.0006206, 0.0006860
5: 0.0157461, 0.0165608, 0.0156860, 0.0165328, -0.0006273, 0.0006933
6: 0.0038516, 0.0042479, 0.0038652, 0.0042771, -0.0003372, 0.0003051
7: -0.0105556, -0.0078090, -0.0107584, -0.0079034, -0.0021148, 0.0023374
8: 0.0083548, 0.0105339, 0.0081940, 0.0104590, -0.0016778, 0.0018544
9: 0.0127516, 0.0166708, 0.0124623, 0.0165361, -0.0030177, 0.0033353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004711, upper bound: 0.0004761
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007796, upper bound: 0.0007450
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007796, upper bound: 0.0007452
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041532, -0.0041213, -0.0041523, -0.0041190, -0.0000274, 0.0000245
1: -0.0081914, -0.0069980, -0.0081603, -0.0069121, -0.0010259, 0.0009171
2: 0.9666334, 0.9680656, 0.9666708, 0.9681687, -0.0012311, 0.0011005
3: 0.0001996, 0.0107629, 0.0004750, 0.0115237, -0.0090802, 0.0081171
4: -0.0015116, -0.0007082, -0.0015695, -0.0007292, -0.0006174, 0.0006906
5: 0.0157426, 0.0165546, 0.0156841, 0.0165334, -0.0006239, 0.0006980
6: 0.0038546, 0.0042496, 0.0038649, 0.0042780, -0.0003395, 0.0003035
7: -0.0105675, -0.0078300, -0.0107647, -0.0079014, -0.0021036, 0.0023532
8: 0.0083454, 0.0105172, 0.0081889, 0.0104606, -0.0016689, 0.0018669
9: 0.0127345, 0.0166408, 0.0124532, 0.0165390, -0.0030017, 0.0033578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004865, upper bound: 0.0005022
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007796, upper bound: 0.0007456
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007796, upper bound: 0.0007484
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041532, -0.0041213, -0.0041523, -0.0041196, -0.0000273, 0.0000250
1: -0.0081914, -0.0069980, -0.0081611, -0.0069349, -0.0010212, 0.0009366
2: 0.9666334, 0.9680656, 0.9666699, 0.9681413, -0.0012255, 0.0011239
3: 0.0001996, 0.0107629, 0.0004683, 0.0113214, -0.0090388, 0.0082899
4: -0.0015116, -0.0007082, -0.0015541, -0.0007286, -0.0006305, 0.0006875
5: 0.0157426, 0.0165546, 0.0156997, 0.0165339, -0.0006372, 0.0006948
6: 0.0038546, 0.0042496, 0.0038647, 0.0042705, -0.0003379, 0.0003099
7: -0.0105675, -0.0078300, -0.0107123, -0.0078996, -0.0021484, 0.0023425
8: 0.0083454, 0.0105172, 0.0082305, 0.0104620, -0.0017044, 0.0018584
9: 0.0127345, 0.0166408, 0.0125280, 0.0165415, -0.0030656, 0.0033426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005309, upper bound: 0.0004755
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007656, upper bound: 0.0007198
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B1_A1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007656, upper bound: 0.0007302
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0041530, -0.0041206, -0.0041523, -0.0041196, -0.0000272, 0.0000256
1: -0.0081861, -0.0069740, -0.0081611, -0.0069349, -0.0010183, 0.0009595
2: 0.9666398, 0.9680943, 0.9666699, 0.9681413, -0.0012220, 0.0011514
3: 0.0002467, 0.0109753, 0.0004683, 0.0113214, -0.0090135, 0.0084927
4: -0.0015278, -0.0007118, -0.0015541, -0.0007286, -0.0006459, 0.0006855
5: 0.0157263, 0.0165510, 0.0156997, 0.0165339, -0.0006528, 0.0006929
6: 0.0038564, 0.0042575, 0.0038647, 0.0042705, -0.0003370, 0.0003175
7: -0.0106226, -0.0078422, -0.0107123, -0.0078996, -0.0022010, 0.0023359
8: 0.0083017, 0.0105075, 0.0082305, 0.0104620, -0.0017461, 0.0018532
9: 0.0126560, 0.0166234, 0.0125280, 0.0165415, -0.0031406, 0.0033332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004928, upper bound: 0.0004407
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_A2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007602, upper bound: 0.0007070
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007602, upper bound: 0.0007204
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041525, -0.0041201, -0.0041534, -0.0041214, -0.0000254, 0.0000267
1: -0.0081681, -0.0069554, -0.0082006, -0.0070032, -0.0009493, 0.0009999
2: 0.9666615, 0.9681166, 0.9666225, 0.9680593, -0.0011392, 0.0011999
3: 0.0004060, 0.0111400, 0.0001186, 0.0107168, -0.0084027, 0.0088505
4: -0.0015403, -0.0007239, -0.0015081, -0.0007021, -0.0006731, 0.0006391
5: 0.0157136, 0.0165387, 0.0157461, 0.0165608, -0.0006803, 0.0006459
6: 0.0038624, 0.0042637, 0.0038516, 0.0042479, -0.0003142, 0.0003309
7: -0.0106653, -0.0078835, -0.0105556, -0.0078090, -0.0022937, 0.0021776
8: 0.0082678, 0.0104748, 0.0083548, 0.0105339, -0.0018197, 0.0017276
9: 0.0125951, 0.0165645, 0.0127516, 0.0166708, -0.0032729, 0.0031073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004978, upper bound: 0.0005411
time: 0.82 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007450, upper bound: 0.0007838
time: 0.93 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007450, upper bound: 0.0007850
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041526, -0.0041201, -0.0041532, -0.0041213, -0.0000252, 0.0000268
1: -0.0081690, -0.0069527, -0.0081914, -0.0069980, -0.0009435, 0.0010049
2: 0.9666603, 0.9681199, 0.9666334, 0.9680656, -0.0011323, 0.0012059
3: 0.0003983, 0.0111636, 0.0001996, 0.0107629, -0.0083515, 0.0088946
4: -0.0015421, -0.0007233, -0.0015116, -0.0007082, -0.0006765, 0.0006352
5: 0.0157118, 0.0165393, 0.0157426, 0.0165546, -0.0006837, 0.0006420
6: 0.0038621, 0.0042646, 0.0038546, 0.0042496, -0.0003123, 0.0003326
7: -0.0106714, -0.0078815, -0.0105675, -0.0078300, -0.0023051, 0.0021644
8: 0.0082630, 0.0104763, 0.0083454, 0.0105172, -0.0018288, 0.0017171
9: 0.0125863, 0.0165673, 0.0127345, 0.0166408, -0.0032892, 0.0030884

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005229, upper bound: 0.0005525
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007456, upper bound: 0.0007838
time: 0.81 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007456, upper bound: 0.0007867
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041523, -0.0041191, -0.0041534, -0.0041214, -0.0000246, 0.0000272
1: -0.0081594, -0.0069148, -0.0082006, -0.0070032, -0.0009219, 0.0010190
2: 0.9666719, 0.9681653, 0.9666225, 0.9680593, -0.0011064, 0.0012228
3: 0.0004828, 0.0114991, 0.0001186, 0.0107168, -0.0081603, 0.0090192
4: -0.0015676, -0.0007297, -0.0015081, -0.0007021, -0.0006860, 0.0006206
5: 0.0156860, 0.0165328, 0.0157461, 0.0165608, -0.0006933, 0.0006273
6: 0.0038652, 0.0042771, 0.0038516, 0.0042479, -0.0003051, 0.0003372
7: -0.0107584, -0.0079034, -0.0105556, -0.0078090, -0.0023374, 0.0021148
8: 0.0081940, 0.0104590, 0.0083548, 0.0105339, -0.0018544, 0.0016778
9: 0.0124623, 0.0165361, 0.0127516, 0.0166708, -0.0033353, 0.0030177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004761, upper bound: 0.0004711
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007546, upper bound: 0.0007696
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007546, upper bound: 0.0007720
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041523, -0.0041190, -0.0041532, -0.0041213, -0.0000245, 0.0000274
1: -0.0081603, -0.0069121, -0.0081914, -0.0069980, -0.0009171, 0.0010259
2: 0.9666708, 0.9681687, 0.9666334, 0.9680656, -0.0011005, 0.0012311
3: 0.0004750, 0.0115237, 0.0001996, 0.0107629, -0.0081171, 0.0090802
4: -0.0015695, -0.0007292, -0.0015116, -0.0007082, -0.0006906, 0.0006174
5: 0.0156841, 0.0165334, 0.0157426, 0.0165546, -0.0006980, 0.0006239
6: 0.0038649, 0.0042780, 0.0038546, 0.0042496, -0.0003035, 0.0003395
7: -0.0107647, -0.0079014, -0.0105675, -0.0078300, -0.0023532, 0.0021036
8: 0.0081889, 0.0104606, 0.0083454, 0.0105172, -0.0018669, 0.0016689
9: 0.0124532, 0.0165390, 0.0127345, 0.0166408, -0.0033578, 0.0030017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005022, upper bound: 0.0004865
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007550, upper bound: 0.0007696
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007550, upper bound: 0.0007735
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041523, -0.0041196, -0.0041532, -0.0041213, -0.0000250, 0.0000273
1: -0.0081611, -0.0069349, -0.0081914, -0.0069980, -0.0009366, 0.0010212
2: 0.9666699, 0.9681413, 0.9666334, 0.9680656, -0.0011239, 0.0012255
3: 0.0004683, 0.0113214, 0.0001996, 0.0107629, -0.0082899, 0.0090388
4: -0.0015541, -0.0007286, -0.0015116, -0.0007082, -0.0006875, 0.0006305
5: 0.0156997, 0.0165339, 0.0157426, 0.0165546, -0.0006948, 0.0006372
6: 0.0038647, 0.0042705, 0.0038546, 0.0042496, -0.0003099, 0.0003379
7: -0.0107123, -0.0078996, -0.0105675, -0.0078300, -0.0023425, 0.0021484
8: 0.0082305, 0.0104620, 0.0083454, 0.0105172, -0.0018584, 0.0017044
9: 0.0125280, 0.0165415, 0.0127345, 0.0166408, -0.0033426, 0.0030656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004755, upper bound: 0.0005309
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007198, upper bound: 0.0007656
time: 0.84 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007198, upper bound: 0.0007779
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0041523, -0.0041196, -0.0041530, -0.0041206, -0.0000256, 0.0000272
1: -0.0081611, -0.0069349, -0.0081861, -0.0069740, -0.0009595, 0.0010183
2: 0.9666699, 0.9681413, 0.9666398, 0.9680943, -0.0011514, 0.0012220
3: 0.0004683, 0.0113214, 0.0002467, 0.0109753, -0.0084927, 0.0090135
4: -0.0015541, -0.0007286, -0.0015278, -0.0007118, -0.0006855, 0.0006459
5: 0.0156997, 0.0165339, 0.0157263, 0.0165510, -0.0006929, 0.0006528
6: 0.0038647, 0.0042705, 0.0038564, 0.0042575, -0.0003175, 0.0003370
7: -0.0107123, -0.0078996, -0.0106226, -0.0078422, -0.0023359, 0.0022010
8: 0.0082305, 0.0104620, 0.0083017, 0.0105075, -0.0018532, 0.0017461
9: 0.0125280, 0.0165415, 0.0126560, 0.0166234, -0.0033332, 0.0031406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2_B2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004407, upper bound: 0.0004928
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007070, upper bound: 0.0007602
time: 0.96 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007070, upper bound: 0.0007742
time: 1.04 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 8.35 seconds
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007838, upper bound: 0.0007450
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007838, upper bound: 0.0007452
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007838, upper bound: 0.0007456
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007838, upper bound: 0.0007484
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007796, upper bound: 0.0007450
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007796, upper bound: 0.0007452
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007796, upper bound: 0.0007456
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007796, upper bound: 0.0007484
IS_A1_B2_A2_B2_B1_A1_A2_B1, status: Status.VERIFIED, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007656, upper bound: 0.0007198
IS_A1_B2_A2_B2_B1_A1_A2_B2, status: Status.VERIFIED, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007656, upper bound: 0.0007302
IS_A1_B2_A2_B2_B1_A2_A2_B1, status: Status.VERIFIED, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007602, upper bound: 0.0007070
IS_A1_B2_A2_B2_B1_A2_A2_B2, status: Status.VERIFIED, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007602, upper bound: 0.0007204
IS_A2_B1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007450, upper bound: 0.0007838
IS_A2_B1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007450, upper bound: 0.0007850
IS_A2_B1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007456, upper bound: 0.0007838
IS_A2_B1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007456, upper bound: 0.0007867
IS_A2_B1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007546, upper bound: 0.0007696
IS_A2_B1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007546, upper bound: 0.0007720
IS_A2_B1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007550, upper bound: 0.0007696
IS_A2_B1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007550, upper bound: 0.0007735
IS_A2_B1_B2_A2_A1_B1_B2_A1, status: Status.VERIFIED, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007198, upper bound: 0.0007656
IS_A2_B1_B2_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007198, upper bound: 0.0007779
IS_A2_B1_B2_A2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007070, upper bound: 0.0007602
IS_A2_B1_B2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 8.35
Output dim: 2, lower bound: -0.0007070, upper bound: 0.0007742

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041534, -0.0041214, -0.0041526, -0.0041205, -0.0000263, 0.0000251
1: -0.0082006, -0.0070032, -0.0081724, -0.0069669, -0.0009835, 0.0009405
2: 0.9666225, 0.9680593, 0.9666562, 0.9681028, -0.0011803, 0.0011286
3: 0.0001186, 0.0107168, 0.0003676, 0.0110380, -0.0087057, 0.0083245
4: -0.0015081, -0.0007021, -0.0015325, -0.0007210, -0.0006331, 0.0006621
5: 0.0157461, 0.0165608, 0.0157214, 0.0165417, -0.0006399, 0.0006692
6: 0.0038516, 0.0042479, 0.0038609, 0.0042599, -0.0003255, 0.0003112
7: -0.0105556, -0.0078090, -0.0106388, -0.0078735, -0.0021574, 0.0022561
8: 0.0083548, 0.0105339, 0.0082888, 0.0104827, -0.0017115, 0.0017899
9: 0.0127516, 0.0166708, 0.0126328, 0.0165787, -0.0030784, 0.0032193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005089, upper bound: 0.0004872
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005704, upper bound: 0.0005116
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007800, upper bound: 0.0007412
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041534, -0.0041214, -0.0041525, -0.0041203, -0.0000265, 0.0000252
1: -0.0082006, -0.0070032, -0.0081671, -0.0069599, -0.0009941, 0.0009435
2: 0.9666225, 0.9680593, 0.9666626, 0.9681113, -0.0011929, 0.0011322
3: 0.0001186, 0.0107168, 0.0004149, 0.0111001, -0.0087989, 0.0083511
4: -0.0015081, -0.0007021, -0.0015373, -0.0007246, -0.0006352, 0.0006692
5: 0.0157461, 0.0165608, 0.0157167, 0.0165380, -0.0006419, 0.0006764
6: 0.0038516, 0.0042479, 0.0038627, 0.0042622, -0.0003290, 0.0003122
7: -0.0105556, -0.0078090, -0.0106550, -0.0078858, -0.0021643, 0.0022803
8: 0.0083548, 0.0105339, 0.0082760, 0.0104729, -0.0017170, 0.0018091
9: 0.0127516, 0.0166708, 0.0126098, 0.0165612, -0.0030882, 0.0032538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005089, upper bound: 0.0004978
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005704, upper bound: 0.0005188
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007800, upper bound: 0.0007414
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041532, -0.0041213, -0.0041526, -0.0041205, -0.0000262, 0.0000253
1: -0.0081914, -0.0069980, -0.0081724, -0.0069669, -0.0009824, 0.0009475
2: 0.9666334, 0.9680656, 0.9666562, 0.9681028, -0.0011789, 0.0011370
3: 0.0001996, 0.0107629, 0.0003676, 0.0110380, -0.0086952, 0.0083865
4: -0.0015116, -0.0007082, -0.0015325, -0.0007210, -0.0006378, 0.0006613
5: 0.0157426, 0.0165546, 0.0157214, 0.0165417, -0.0006447, 0.0006684
6: 0.0038546, 0.0042496, 0.0038609, 0.0042599, -0.0003251, 0.0003136
7: -0.0105675, -0.0078300, -0.0106388, -0.0078735, -0.0021734, 0.0022534
8: 0.0083454, 0.0105172, 0.0082888, 0.0104827, -0.0017243, 0.0017878
9: 0.0127345, 0.0166408, 0.0126328, 0.0165787, -0.0031013, 0.0032155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005090, upper bound: 0.0004910
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005711, upper bound: 0.0005147
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007800, upper bound: 0.0007417
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041532, -0.0041213, -0.0041525, -0.0041203, -0.0000261, 0.0000250
1: -0.0081914, -0.0069980, -0.0081671, -0.0069599, -0.0009783, 0.0009372
2: 0.9666334, 0.9680656, 0.9666626, 0.9681113, -0.0011741, 0.0011247
3: 0.0001996, 0.0107629, 0.0004149, 0.0111001, -0.0086597, 0.0082959
4: -0.0015116, -0.0007082, -0.0015373, -0.0007246, -0.0006310, 0.0006586
5: 0.0157426, 0.0165546, 0.0157167, 0.0165380, -0.0006377, 0.0006657
6: 0.0038546, 0.0042496, 0.0038627, 0.0042622, -0.0003238, 0.0003102
7: -0.0105675, -0.0078300, -0.0106550, -0.0078858, -0.0021500, 0.0022442
8: 0.0083454, 0.0105172, 0.0082760, 0.0104729, -0.0017057, 0.0017805
9: 0.0127345, 0.0166408, 0.0126098, 0.0165612, -0.0030678, 0.0032023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005090, upper bound: 0.0005229
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005711, upper bound: 0.0005221
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007800, upper bound: 0.0007446
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041534, -0.0041214, -0.0041524, -0.0041194, -0.0000268, 0.0000244
1: -0.0082006, -0.0070032, -0.0081637, -0.0069268, -0.0010026, 0.0009142
2: 0.9666225, 0.9680593, 0.9666668, 0.9681512, -0.0012031, 0.0010971
3: 0.0001186, 0.0107168, 0.0004451, 0.0113935, -0.0088741, 0.0080923
4: -0.0015081, -0.0007021, -0.0015596, -0.0007269, -0.0006155, 0.0006749
5: 0.0157461, 0.0165608, 0.0156941, 0.0165357, -0.0006220, 0.0006821
6: 0.0038516, 0.0042479, 0.0038638, 0.0042732, -0.0003318, 0.0003026
7: -0.0105556, -0.0078090, -0.0107310, -0.0078936, -0.0020972, 0.0022998
8: 0.0083548, 0.0105339, 0.0082157, 0.0104667, -0.0016638, 0.0018246
9: 0.0127516, 0.0166708, 0.0125013, 0.0165500, -0.0029925, 0.0032816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004103, upper bound: 0.0004347
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005199, upper bound: 0.0005074
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007757, upper bound: 0.0007412
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041534, -0.0041214, -0.0041523, -0.0041192, -0.0000271, 0.0000245
1: -0.0082006, -0.0070032, -0.0081584, -0.0069190, -0.0010130, 0.0009162
2: 0.9666225, 0.9680593, 0.9666731, 0.9681603, -0.0012156, 0.0010995
3: 0.0001186, 0.0107168, 0.0004919, 0.0114618, -0.0089661, 0.0081097
4: -0.0015081, -0.0007021, -0.0015648, -0.0007304, -0.0006168, 0.0006819
5: 0.0157461, 0.0165608, 0.0156889, 0.0165321, -0.0006234, 0.0006892
6: 0.0038516, 0.0042479, 0.0038656, 0.0042757, -0.0003352, 0.0003032
7: -0.0105556, -0.0078090, -0.0107487, -0.0079057, -0.0021017, 0.0023236
8: 0.0083548, 0.0105339, 0.0082017, 0.0104571, -0.0016674, 0.0018435
9: 0.0127516, 0.0166708, 0.0124761, 0.0165327, -0.0029990, 0.0033157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004103, upper bound: 0.0004761
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005199, upper bound: 0.0005121
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007757, upper bound: 0.0007414
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041532, -0.0041213, -0.0041524, -0.0041194, -0.0000268, 0.0000246
1: -0.0081914, -0.0069980, -0.0081637, -0.0069268, -0.0010033, 0.0009213
2: 0.9666334, 0.9680656, 0.9666668, 0.9681512, -0.0012040, 0.0011055
3: 0.0001996, 0.0107629, 0.0004451, 0.0113935, -0.0088807, 0.0081543
4: -0.0015116, -0.0007082, -0.0015596, -0.0007269, -0.0006202, 0.0006754
5: 0.0157426, 0.0165546, 0.0156941, 0.0165357, -0.0006268, 0.0006826
6: 0.0038546, 0.0042496, 0.0038638, 0.0042732, -0.0003320, 0.0003049
7: -0.0105675, -0.0078300, -0.0107310, -0.0078936, -0.0021133, 0.0023015
8: 0.0083454, 0.0105172, 0.0082157, 0.0104667, -0.0016766, 0.0018259
9: 0.0127345, 0.0166408, 0.0125013, 0.0165500, -0.0030154, 0.0032841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004106, upper bound: 0.0004361
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005475, upper bound: 0.0004963
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007757, upper bound: 0.0007417
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041532, -0.0041213, -0.0041523, -0.0041192, -0.0000266, 0.0000243
1: -0.0081914, -0.0069980, -0.0081584, -0.0069190, -0.0009965, 0.0009108
2: 0.9666334, 0.9680656, 0.9666731, 0.9681603, -0.0011959, 0.0010930
3: 0.0001996, 0.0107629, 0.0004919, 0.0114618, -0.0088207, 0.0080620
4: -0.0015116, -0.0007082, -0.0015648, -0.0007304, -0.0006132, 0.0006709
5: 0.0157426, 0.0165546, 0.0156889, 0.0165321, -0.0006197, 0.0006780
6: 0.0038546, 0.0042496, 0.0038656, 0.0042757, -0.0003298, 0.0003014
7: -0.0105675, -0.0078300, -0.0107487, -0.0079057, -0.0020893, 0.0022860
8: 0.0083454, 0.0105172, 0.0082017, 0.0104571, -0.0016576, 0.0018136
9: 0.0127345, 0.0166408, 0.0124761, 0.0165327, -0.0029813, 0.0032619

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.14 + 597.10 = 600.24 seconds
