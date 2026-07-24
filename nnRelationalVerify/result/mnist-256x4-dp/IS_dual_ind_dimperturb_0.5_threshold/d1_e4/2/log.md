## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00066248


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0006548, 0.0122371, -0.0006548, 0.0122371, -0.0122943, 0.0122943)
1: (-0.0036243, 0.0034959, -0.0036243, 0.0034959, -0.0071203, 0.0071203)
2: (0.0044464, 0.0170466, 0.0044464, 0.0170466, -0.0126002, 0.0126002)
3: (1.0056938, 1.0071729, 1.0056938, 1.0071729, -0.0014791, 0.0014791)
4: (-0.0044103, -0.0001325, -0.0044103, -0.0001325, -0.0042778, 0.0042778)
5: (0.0034811, 0.0196071, 0.0034811, 0.0196071, -0.0156474, 0.0156474)
6: (-0.0148070, -0.0025298, -0.0148070, -0.0025298, -0.0122772, 0.0122772)
7: (-0.0184874, -0.0101309, -0.0184874, -0.0101309, -0.0082882, 0.0082882)
8: (-0.0153831, -0.0060751, -0.0153831, -0.0060751, -0.0093079, 0.0093079)
9: (-0.0071178, 0.0038722, -0.0071178, 0.0038722, -0.0109900, 0.0109900)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.40 + 1.73 = 3.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0010156, upper bound: 0.0010156

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 183

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 183

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009631, upper bound: 0.0009432
time: 0.97 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009399, upper bound: 0.0009399
time: 0.85 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.95 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.95
Output dim: 3, lower bound: -0.0009631, upper bound: 0.0009432
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.95
Output dim: 3, lower bound: -0.0009399, upper bound: 0.0009399

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0006528, 0.0117925, -0.0006548, 0.0122371, -0.0122822, 0.0118476
1: -0.0035917, 0.0032300, -0.0036243, 0.0034959, -0.0069784, 0.0068544
2: 0.0052129, 0.0170196, 0.0044464, 0.0170466, -0.0118337, 0.0125732
3: 1.0057517, 1.0071558, 1.0056938, 1.0071729, -0.0014212, 0.0014620
4: -0.0044069, -0.0006408, -0.0044103, -0.0001325, -0.0042744, 0.0037695
5: 0.0034825, 0.0185246, 0.0034811, 0.0196071, -0.0156460, 0.0145631
6: -0.0141238, -0.0025332, -0.0148070, -0.0025298, -0.0115940, 0.0122738
7: -0.0180430, -0.0101350, -0.0184874, -0.0101309, -0.0078435, 0.0082840
8: -0.0153404, -0.0064147, -0.0153831, -0.0060751, -0.0092653, 0.0089684
9: -0.0067099, 0.0034125, -0.0071178, 0.0038722, -0.0105821, 0.0105304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009399, upper bound: 0.0009399
time: 1.01 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009399, upper bound: 0.0009399
time: 0.82 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0008209, 0.0099003, -0.0006508, 0.0113304, -0.0115932, 0.0099815
1: -0.0035921, 0.0020921, -0.0035914, 0.0029520, -0.0064369, 0.0055731
2: 0.0071509, 0.0168926, 0.0056831, 0.0169874, -0.0098365, 0.0112095
3: 1.0057809, 1.0071299, 1.0057719, 1.0071559, -0.0013750, 0.0013580
4: -0.0043900, -0.0012828, -0.0044025, -0.0007966, -0.0035933, 0.0031198
5: 0.0033570, 0.0161074, 0.0034840, 0.0179344, -0.0141370, 0.0121740
6: -0.0120977, -0.0025378, -0.0136292, -0.0025332, -0.0095644, 0.0110914
7: -0.0171660, -0.0096178, -0.0178350, -0.0101394, -0.0069636, 0.0081569
8: -0.0151875, -0.0077474, -0.0153005, -0.0067309, -0.0084566, 0.0075531
9: -0.0050262, 0.0033586, -0.0063030, 0.0033971, -0.0084233, 0.0096616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 183

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 183

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009399, upper bound: 0.0009399
time: 0.83 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009399, upper bound: 0.0009399
time: 0.86 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.11 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 3, lower bound: -0.0009399, upper bound: 0.0009399
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 3, lower bound: -0.0009399, upper bound: 0.0009399
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 3, lower bound: -0.0009399, upper bound: 0.0009399
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.11
Output dim: 3, lower bound: -0.0009399, upper bound: 0.0009399

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006528, 0.0117925, -0.0006528, 0.0117925, -0.0118355, 0.0118355
1: -0.0035917, 0.0032300, -0.0035917, 0.0032300, -0.0067080, 0.0067080
2: 0.0052129, 0.0170196, 0.0052129, 0.0170196, -0.0118068, 0.0118068
3: 1.0057517, 1.0071558, 1.0057517, 1.0071558, -0.0014040, 0.0014040
4: -0.0044069, -0.0006408, -0.0044069, -0.0006408, -0.0037661, 0.0037661
5: 0.0034825, 0.0185246, 0.0034825, 0.0185246, -0.0145616, 0.0145616
6: -0.0141238, -0.0025332, -0.0141238, -0.0025332, -0.0115906, 0.0115906
7: -0.0180430, -0.0101350, -0.0180430, -0.0101350, -0.0078393, 0.0078393
8: -0.0153404, -0.0064147, -0.0153404, -0.0064147, -0.0089257, 0.0089257
9: -0.0067099, 0.0034125, -0.0067099, 0.0034125, -0.0101224, 0.0101224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 237

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008829
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009180, upper bound: 0.0008807
time: 1.28 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006528, 0.0117925, -0.0008209, 0.0099003, -0.0099698, 0.0120466
1: -0.0035917, 0.0032300, -0.0035921, 0.0020921, -0.0055683, 0.0067148
2: 0.0052129, 0.0170196, 0.0071509, 0.0168926, -0.0116797, 0.0098688
3: 1.0057517, 1.0071558, 1.0057809, 1.0071299, -0.0013782, 0.0013748
4: -0.0044069, -0.0006408, -0.0043900, -0.0012828, -0.0031241, 0.0037492
5: 0.0034825, 0.0185246, 0.0033570, 0.0161074, -0.0121646, 0.0147208
6: -0.0141238, -0.0025332, -0.0120977, -0.0025378, -0.0115860, 0.0095645
7: -0.0180430, -0.0101350, -0.0171660, -0.0096178, -0.0083630, 0.0069665
8: -0.0153404, -0.0064147, -0.0151875, -0.0077474, -0.0075930, 0.0087729
9: -0.0067099, 0.0034125, -0.0050262, 0.0033586, -0.0100685, 0.0084387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 237

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008829
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009180, upper bound: 0.0008807
time: 1.35 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0008209, 0.0099003, -0.0006528, 0.0117925, -0.0120466, 0.0099698
1: -0.0035921, 0.0020921, -0.0035917, 0.0032300, -0.0067148, 0.0055683
2: 0.0071509, 0.0168926, 0.0052129, 0.0170196, -0.0098688, 0.0116797
3: 1.0057809, 1.0071299, 1.0057517, 1.0071558, -0.0013748, 0.0013782
4: -0.0043900, -0.0012828, -0.0044069, -0.0006408, -0.0037492, 0.0031241
5: 0.0033570, 0.0161074, 0.0034825, 0.0185246, -0.0147208, 0.0121646
6: -0.0120977, -0.0025378, -0.0141238, -0.0025332, -0.0095645, 0.0115860
7: -0.0171660, -0.0096178, -0.0180430, -0.0101350, -0.0069665, 0.0083630
8: -0.0151875, -0.0077474, -0.0153404, -0.0064147, -0.0087729, 0.0075930
9: -0.0050262, 0.0033586, -0.0067099, 0.0034125, -0.0084387, 0.0100685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008693, upper bound: 0.0008788
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008770, upper bound: 0.0008770
time: 0.87 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0008209, 0.0099003, -0.0008209, 0.0099003, -0.0101606, 0.0101606
1: -0.0035921, 0.0020921, -0.0035921, 0.0020921, -0.0055729, 0.0055729
2: 0.0071509, 0.0168926, 0.0071509, 0.0168926, -0.0097417, 0.0097417
3: 1.0057809, 1.0071299, 1.0057809, 1.0071299, -0.0013490, 0.0013490
4: -0.0043900, -0.0012828, -0.0043900, -0.0012828, -0.0031072, 0.0031072
5: 0.0033570, 0.0161074, 0.0033570, 0.0161074, -0.0123078, 0.0123078
6: -0.0120977, -0.0025378, -0.0120977, -0.0025378, -0.0095599, 0.0095599
7: -0.0171660, -0.0096178, -0.0171660, -0.0096178, -0.0074877, 0.0074877
8: -0.0151875, -0.0077474, -0.0151875, -0.0077474, -0.0074402, 0.0074402
9: -0.0050262, 0.0033586, -0.0050262, 0.0033586, -0.0083848, 0.0083848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008693, upper bound: 0.0008788
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008770, upper bound: 0.0008770
time: 0.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.61 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008829
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 3, lower bound: -0.0009180, upper bound: 0.0008807
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008829
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 3, lower bound: -0.0009180, upper bound: 0.0008807
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 3, lower bound: -0.0008693, upper bound: 0.0008788
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 3, lower bound: -0.0008770, upper bound: 0.0008770
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 3, lower bound: -0.0008693, upper bound: 0.0008788
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.61
Output dim: 3, lower bound: -0.0008770, upper bound: 0.0008770

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0006526, 0.0113464, -0.0113917, 0.0100521
1: -0.0035917, 0.0021417, -0.0035917, 0.0029598, -0.0064238, 0.0056104
2: 0.0070180, 0.0168748, 0.0056615, 0.0169842, -0.0099662, 0.0112133
3: 1.0058836, 1.0071558, 1.0057843, 1.0071558, -0.0012722, 0.0013715
4: -0.0043845, -0.0012404, -0.0044014, -0.0007899, -0.0035946, 0.0031610
5: 0.0034829, 0.0162374, 0.0034826, 0.0179555, -0.0139945, 0.0122822
6: -0.0122079, -0.0025360, -0.0136470, -0.0025339, -0.0096741, 0.0111110
7: -0.0172598, -0.0101350, -0.0178495, -0.0101350, -0.0070568, 0.0076459
8: -0.0151080, -0.0076373, -0.0152838, -0.0067195, -0.0083885, 0.0076465
9: -0.0051578, 0.0033023, -0.0063246, 0.0033856, -0.0085434, 0.0096269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009880
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0006525, 0.0109627, -0.0112040, 0.0100366
1: -0.0036292, 0.0021306, -0.0035917, 0.0027272, -0.0062051, 0.0056016
2: 0.0070511, 0.0168794, 0.0060498, 0.0169537, -0.0099026, 0.0108296
3: 1.0058714, 1.0071830, 1.0058128, 1.0071558, -0.0012844, 0.0013702
4: -0.0043853, -0.0012508, -0.0043967, -0.0009186, -0.0034667, 0.0031459
5: 0.0033241, 0.0162060, 0.0034827, 0.0174660, -0.0136551, 0.0122567
6: -0.0121811, -0.0025370, -0.0132369, -0.0025349, -0.0096463, 0.0106999
7: -0.0172352, -0.0098209, -0.0176750, -0.0101350, -0.0070339, 0.0077848
8: -0.0151180, -0.0076559, -0.0152349, -0.0069874, -0.0081305, 0.0075790
9: -0.0051244, 0.0033073, -0.0059869, 0.0033628, -0.0084871, 0.0092942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009879, upper bound: 0.0009651
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009879, upper bound: 0.0009880
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0008208, 0.0094634, -0.0095342, 0.0102631
1: -0.0035917, 0.0021417, -0.0035921, 0.0018266, -0.0052897, 0.0056172
2: 0.0070180, 0.0168748, 0.0075913, 0.0168577, -0.0098397, 0.0092835
3: 1.0058836, 1.0071558, 1.0058115, 1.0071299, -0.0012463, 0.0013443
4: -0.0043845, -0.0012404, -0.0043846, -0.0014291, -0.0029554, 0.0031442
5: 0.0034829, 0.0162374, 0.0033571, 0.0155502, -0.0116077, 0.0124412
6: -0.0122079, -0.0025360, -0.0116309, -0.0025385, -0.0096694, 0.0090949
7: -0.0172598, -0.0101350, -0.0169744, -0.0096178, -0.0075805, 0.0067750
8: -0.0151080, -0.0076373, -0.0151318, -0.0080359, -0.0070721, 0.0074945
9: -0.0051578, 0.0033023, -0.0046478, 0.0033319, -0.0084898, 0.0079501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008733
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008807
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0008206, 0.0091390, -0.0093997, 0.0102486
1: -0.0036292, 0.0021306, -0.0035921, 0.0016317, -0.0051064, 0.0056087
2: 0.0070511, 0.0168794, 0.0079252, 0.0168329, -0.0097819, 0.0089542
3: 1.0058714, 1.0071830, 1.0058367, 1.0071299, -0.0012585, 0.0013462
4: -0.0043853, -0.0012508, -0.0043808, -0.0015397, -0.0028456, 0.0031300
5: 0.0033241, 0.0162060, 0.0033572, 0.0151356, -0.0113397, 0.0124163
6: -0.0121811, -0.0025370, -0.0112832, -0.0025395, -0.0096417, 0.0087462
7: -0.0172352, -0.0098209, -0.0168193, -0.0096178, -0.0075578, 0.0069329
8: -0.0151180, -0.0076559, -0.0150928, -0.0082562, -0.0068617, 0.0074369
9: -0.0051244, 0.0033073, -0.0043551, 0.0033136, -0.0084379, 0.0076625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008968, upper bound: 0.0008423
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008845, upper bound: 0.0008401
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0006527, 0.0113765, -0.0116326, 0.0081552
1: -0.0035921, 0.0009910, -0.0035917, 0.0029780, -0.0064494, 0.0044548
2: 0.0089864, 0.0167475, 0.0056313, 0.0169866, -0.0080002, 0.0111161
3: 1.0059075, 1.0071299, 1.0057822, 1.0071558, -0.0012482, 0.0013477
4: -0.0043676, -0.0018928, -0.0044018, -0.0007799, -0.0035877, 0.0025090
5: 0.0033575, 0.0137868, 0.0034826, 0.0179938, -0.0141919, 0.0098463
6: -0.0101534, -0.0025408, -0.0136792, -0.0025338, -0.0076196, 0.0111384
7: -0.0163678, -0.0096178, -0.0178626, -0.0101350, -0.0061688, 0.0081827
8: -0.0149543, -0.0089254, -0.0152876, -0.0066990, -0.0082553, 0.0063622
9: -0.0034489, 0.0032479, -0.0063505, 0.0033874, -0.0068363, 0.0095984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009180
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010320, 0.0081786, -0.0006525, 0.0109656, -0.0114205, 0.0082558
1: -0.0036297, 0.0010522, -0.0035917, 0.0027289, -0.0062171, 0.0045180
2: 0.0089004, 0.0167598, 0.0060469, 0.0169540, -0.0080535, 0.0107129
3: 1.0058941, 1.0071546, 1.0058126, 1.0071558, -0.0012617, 0.0013419
4: -0.0043695, -0.0018634, -0.0043967, -0.0009177, -0.0034519, 0.0025333
5: 0.0031950, 0.0139095, 0.0034827, 0.0174696, -0.0138197, 0.0099724
6: -0.0102558, -0.0025416, -0.0132399, -0.0025348, -0.0077210, 0.0106984
7: -0.0163867, -0.0092872, -0.0176763, -0.0101350, -0.0061888, 0.0083245
8: -0.0149751, -0.0088170, -0.0152352, -0.0069855, -0.0079896, 0.0064182
9: -0.0035149, 0.0032580, -0.0059894, 0.0033630, -0.0068779, 0.0092473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008807, upper bound: 0.0009053
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008807, upper bound: 0.0009180
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0008208, 0.0094599, -0.0097213, 0.0083445
1: -0.0035921, 0.0009910, -0.0035921, 0.0018244, -0.0052923, 0.0044592
2: 0.0089864, 0.0167475, 0.0075949, 0.0168574, -0.0078710, 0.0091526
3: 1.0059075, 1.0071299, 1.0058116, 1.0071299, -0.0012224, 0.0013183
4: -0.0043676, -0.0018928, -0.0043845, -0.0014303, -0.0029372, 0.0024917
5: 0.0033575, 0.0137868, 0.0033571, 0.0155457, -0.0117464, 0.0099890
6: -0.0101534, -0.0025408, -0.0116271, -0.0025385, -0.0076149, 0.0090863
7: -0.0163678, -0.0096178, -0.0169729, -0.0096178, -0.0066899, 0.0072946
8: -0.0149543, -0.0089254, -0.0151313, -0.0080383, -0.0069160, 0.0062059
9: -0.0034489, 0.0032479, -0.0046448, 0.0033317, -0.0067806, 0.0078927

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008692, upper bound: 0.0008692
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008692, upper bound: 0.0008770
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010320, 0.0081786, -0.0008206, 0.0091225, -0.0095787, 0.0084502
1: -0.0036297, 0.0010522, -0.0035921, 0.0016217, -0.0051041, 0.0045241
2: 0.0089004, 0.0167598, 0.0079419, 0.0168317, -0.0079313, 0.0088179
3: 1.0058941, 1.0071546, 1.0058380, 1.0071299, -0.0012358, 0.0013165
4: -0.0043695, -0.0018634, -0.0043806, -0.0015453, -0.0028243, 0.0025172
5: 0.0031950, 0.0139095, 0.0033572, 0.0151145, -0.0114657, 0.0101184
6: -0.0102558, -0.0025416, -0.0112656, -0.0025395, -0.0077163, 0.0087240
7: -0.0163867, -0.0092872, -0.0168119, -0.0096178, -0.0067103, 0.0074615
8: -0.0149751, -0.0088170, -0.0150907, -0.0082672, -0.0067079, 0.0062737
9: -0.0035149, 0.0032580, -0.0043407, 0.0033126, -0.0068276, 0.0075986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008454, upper bound: 0.0008349
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008338, upper bound: 0.0008338
time: 0.96 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.62 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009880
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0009879, upper bound: 0.0009651
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0009879, upper bound: 0.0009880
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008733
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008807
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0008968, upper bound: 0.0008423
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0008845, upper bound: 0.0008401
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009180
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0008807, upper bound: 0.0009053
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0008807, upper bound: 0.0009180
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0008692, upper bound: 0.0008692
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0008692, upper bound: 0.0008770
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0008454, upper bound: 0.0008349
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.62
Output dim: 3, lower bound: -0.0008338, upper bound: 0.0008338

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0006523, 0.0099991, -0.0100518, 0.0100518
1: -0.0035917, 0.0021417, -0.0035917, 0.0021417, -0.0056016, 0.0056016
2: 0.0070180, 0.0168748, 0.0070180, 0.0168748, -0.0098568, 0.0098568
3: 1.0058836, 1.0071558, 1.0058836, 1.0071558, -0.0012722, 0.0012722
4: -0.0043845, -0.0012404, -0.0043845, -0.0012404, -0.0031441, 0.0031441
5: 0.0034829, 0.0162374, 0.0034829, 0.0162374, -0.0122819, 0.0122819
6: -0.0122079, -0.0025360, -0.0122079, -0.0025360, -0.0096719, 0.0096719
7: -0.0172598, -0.0101350, -0.0172598, -0.0101350, -0.0070568, 0.0070568
8: -0.0151080, -0.0076373, -0.0151080, -0.0076373, -0.0074707, 0.0074707
9: -0.0051578, 0.0033023, -0.0051578, 0.0033023, -0.0084601, 0.0084601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009695
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0008592, 0.0099755, -0.0100297, 0.0102475
1: -0.0035917, 0.0021417, -0.0036292, 0.0021306, -0.0055916, 0.0056173
2: 0.0070180, 0.0168748, 0.0070511, 0.0168794, -0.0098614, 0.0098237
3: 1.0058836, 1.0071558, 1.0058714, 1.0071830, -0.0012994, 0.0012844
4: -0.0043845, -0.0012404, -0.0043853, -0.0012508, -0.0031337, 0.0031449
5: 0.0034829, 0.0162374, 0.0033241, 0.0162060, -0.0122511, 0.0124320
6: -0.0122079, -0.0025360, -0.0121811, -0.0025370, -0.0096709, 0.0096451
7: -0.0172598, -0.0101350, -0.0172352, -0.0098209, -0.0073705, 0.0070330
8: -0.0151080, -0.0076373, -0.0151180, -0.0076559, -0.0074521, 0.0074807
9: -0.0051578, 0.0033023, -0.0051244, 0.0033073, -0.0084652, 0.0084266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009746
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009699
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0006523, 0.0099991, -0.0102475, 0.0100297
1: -0.0036292, 0.0021306, -0.0035917, 0.0021417, -0.0056173, 0.0055916
2: 0.0070511, 0.0168794, 0.0070180, 0.0168748, -0.0098237, 0.0098614
3: 1.0058714, 1.0071830, 1.0058836, 1.0071558, -0.0012844, 0.0012994
4: -0.0043853, -0.0012508, -0.0043845, -0.0012404, -0.0031449, 0.0031337
5: 0.0033241, 0.0162060, 0.0034829, 0.0162374, -0.0124320, 0.0122511
6: -0.0121811, -0.0025370, -0.0122079, -0.0025360, -0.0096451, 0.0096709
7: -0.0172352, -0.0098209, -0.0172598, -0.0101350, -0.0070330, 0.0073705
8: -0.0151180, -0.0076559, -0.0151080, -0.0076373, -0.0074807, 0.0074521
9: -0.0051244, 0.0033073, -0.0051578, 0.0033023, -0.0084266, 0.0084652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009879, upper bound: 0.0009651
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0008592, 0.0099755, -0.0101064, 0.0101064
1: -0.0036292, 0.0021306, -0.0036292, 0.0021306, -0.0055980, 0.0055980
2: 0.0070511, 0.0168794, 0.0070511, 0.0168794, -0.0098283, 0.0098283
3: 1.0058714, 1.0071830, 1.0058714, 1.0071830, -0.0013115, 0.0013115
4: -0.0043853, -0.0012508, -0.0043853, -0.0012508, -0.0031345, 0.0031345
5: 0.0033241, 0.0162060, 0.0033241, 0.0162060, -0.0123078, 0.0123078
6: -0.0121811, -0.0025370, -0.0121811, -0.0025370, -0.0096441, 0.0096441
7: -0.0172352, -0.0098209, -0.0172352, -0.0098209, -0.0073326, 0.0073326
8: -0.0151180, -0.0076559, -0.0151180, -0.0076559, -0.0074621, 0.0074621
9: -0.0051244, 0.0033073, -0.0051244, 0.0033073, -0.0084317, 0.0084317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009487
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009490
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0008203, 0.0080815, -0.0081549, 0.0102628
1: -0.0035917, 0.0021417, -0.0035921, 0.0009910, -0.0044458, 0.0056085
2: 0.0070180, 0.0168748, 0.0089864, 0.0167475, -0.0097295, 0.0078884
3: 1.0058836, 1.0071558, 1.0059075, 1.0071299, -0.0012463, 0.0012482
4: -0.0043845, -0.0012404, -0.0043676, -0.0018928, -0.0024917, 0.0031272
5: 0.0034829, 0.0162374, 0.0033575, 0.0137868, -0.0098461, 0.0124410
6: -0.0122079, -0.0025360, -0.0101534, -0.0025408, -0.0096671, 0.0076173
7: -0.0172598, -0.0101350, -0.0163678, -0.0096178, -0.0075805, 0.0061688
8: -0.0151080, -0.0076373, -0.0149543, -0.0089254, -0.0061826, 0.0073170
9: -0.0051578, 0.0033023, -0.0034489, 0.0032479, -0.0084058, 0.0067511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008823
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008734
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0010320, 0.0081786, -0.0082542, 0.0104603
1: -0.0035917, 0.0021417, -0.0036297, 0.0010522, -0.0045092, 0.0056270
2: 0.0070180, 0.0168748, 0.0089004, 0.0167598, -0.0097418, 0.0079744
3: 1.0058836, 1.0071558, 1.0058941, 1.0071546, -0.0012710, 0.0012617
4: -0.0043845, -0.0012404, -0.0043695, -0.0018634, -0.0025211, 0.0031292
5: 0.0034829, 0.0162374, 0.0031950, 0.0139095, -0.0099703, 0.0125925
6: -0.0122079, -0.0025360, -0.0102558, -0.0025416, -0.0096664, 0.0077198
7: -0.0172598, -0.0101350, -0.0163867, -0.0092872, -0.0079089, 0.0061876
8: -0.0151080, -0.0076373, -0.0149751, -0.0088170, -0.0062910, 0.0073378
9: -0.0051578, 0.0033023, -0.0035149, 0.0032580, -0.0084158, 0.0068172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008829
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008807
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0008512, 0.0099756, -0.0007888, 0.0099003, -0.0101559, 0.0102121
1: -0.0036267, 0.0021305, -0.0035823, 0.0020921, -0.0055687, 0.0055972
2: 0.0070511, 0.0168794, 0.0071510, 0.0168926, -0.0098415, 0.0097284
3: 1.0058719, 1.0071783, 1.0057830, 1.0071132, -0.0012413, 0.0013953
4: -0.0043853, -0.0012508, -0.0043900, -0.0012828, -0.0031026, 0.0031391
5: 0.0033303, 0.0162060, 0.0033820, 0.0161074, -0.0123071, 0.0123879
6: -0.0121811, -0.0025373, -0.0120976, -0.0025387, -0.0096424, 0.0095604
7: -0.0172352, -0.0098281, -0.0171659, -0.0096454, -0.0075301, 0.0072727
8: -0.0151180, -0.0076571, -0.0151875, -0.0077523, -0.0073657, 0.0075305
9: -0.0051243, 0.0033073, -0.0050260, 0.0033586, -0.0084829, 0.0083333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008423
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008968, upper bound: 0.0008423
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0008425, 0.0099755, -0.0007837, 0.0102666, -0.0105271, 0.0102107
1: -0.0036229, 0.0021305, -0.0035771, 0.0023161, -0.0057973, 0.0055958
2: 0.0070511, 0.0168794, 0.0067873, 0.0169267, -0.0098756, 0.0100921
3: 1.0058724, 1.0071681, 1.0057590, 1.0070903, -0.0012180, 0.0014091
4: -0.0043853, -0.0012508, -0.0043960, -0.0011615, -0.0032239, 0.0031451
5: 0.0033371, 0.0162060, 0.0033860, 0.0165740, -0.0127778, 0.0123869
6: -0.0121811, -0.0025379, -0.0124883, -0.0025403, -0.0096408, 0.0099504
7: -0.0172352, -0.0098329, -0.0173095, -0.0096460, -0.0075293, 0.0074118
8: -0.0151180, -0.0076584, -0.0152551, -0.0075303, -0.0075876, 0.0075968
9: -0.0051243, 0.0033073, -0.0053291, 0.0033921, -0.0085163, 0.0086364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008401
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008845, upper bound: 0.0008401
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0006523, 0.0099991, -0.0102628, 0.0081549
1: -0.0035921, 0.0009910, -0.0035917, 0.0021417, -0.0056085, 0.0044458
2: 0.0089864, 0.0167475, 0.0070180, 0.0168748, -0.0078884, 0.0097295
3: 1.0059075, 1.0071299, 1.0058836, 1.0071558, -0.0012482, 0.0012463
4: -0.0043676, -0.0018928, -0.0043845, -0.0012404, -0.0031272, 0.0024917
5: 0.0033575, 0.0137868, 0.0034829, 0.0162374, -0.0124410, 0.0098461
6: -0.0101534, -0.0025408, -0.0122079, -0.0025360, -0.0076173, 0.0096671
7: -0.0163678, -0.0096178, -0.0172598, -0.0101350, -0.0061688, 0.0075805
8: -0.0149543, -0.0089254, -0.0151080, -0.0076373, -0.0073170, 0.0061826
9: -0.0034489, 0.0032479, -0.0051578, 0.0033023, -0.0067511, 0.0084058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 70

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008735, upper bound: 0.0009107
time: 0.89 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0008592, 0.0099755, -0.0102411, 0.0083505
1: -0.0035921, 0.0009910, -0.0036292, 0.0021306, -0.0055990, 0.0044615
2: 0.0089864, 0.0167475, 0.0070511, 0.0168794, -0.0078930, 0.0096964
3: 1.0059075, 1.0071299, 1.0058714, 1.0071830, -0.0012754, 0.0012585
4: -0.0043676, -0.0018928, -0.0043853, -0.0012508, -0.0031167, 0.0024925
5: 0.0033575, 0.0137868, 0.0033241, 0.0162060, -0.0124104, 0.0099961
6: -0.0101534, -0.0025408, -0.0121811, -0.0025370, -0.0076164, 0.0096403
7: -0.0163678, -0.0096178, -0.0172352, -0.0098209, -0.0064824, 0.0075570
8: -0.0149543, -0.0089254, -0.0151180, -0.0076559, -0.0072984, 0.0061925
9: -0.0034489, 0.0032479, -0.0051244, 0.0033073, -0.0067562, 0.0083723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 70

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008385, upper bound: 0.0008968
time: 0.96 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008846
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010320, 0.0081786, -0.0006523, 0.0099991, -0.0104603, 0.0082542
1: -0.0036297, 0.0010522, -0.0035917, 0.0021417, -0.0056270, 0.0045092
2: 0.0089004, 0.0167598, 0.0070180, 0.0168748, -0.0079744, 0.0097418
3: 1.0058941, 1.0071546, 1.0058836, 1.0071558, -0.0012617, 0.0012710
4: -0.0043695, -0.0018634, -0.0043845, -0.0012404, -0.0031292, 0.0025211
5: 0.0031950, 0.0139095, 0.0034829, 0.0162374, -0.0125925, 0.0099703
6: -0.0102558, -0.0025416, -0.0122079, -0.0025360, -0.0077198, 0.0096664
7: -0.0163867, -0.0092872, -0.0172598, -0.0101350, -0.0061876, 0.0079089
8: -0.0149751, -0.0088170, -0.0151080, -0.0076373, -0.0073378, 0.0062910
9: -0.0035149, 0.0032580, -0.0051578, 0.0033023, -0.0068172, 0.0084158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008807, upper bound: 0.0009053
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010320, 0.0081786, -0.0008592, 0.0099755, -0.0103213, 0.0083256
1: -0.0036297, 0.0010522, -0.0036292, 0.0021306, -0.0056046, 0.0045144
2: 0.0089004, 0.0167598, 0.0070511, 0.0168794, -0.0079789, 0.0097088
3: 1.0058941, 1.0071546, 1.0058714, 1.0071830, -0.0012889, 0.0012832
4: -0.0043695, -0.0018634, -0.0043853, -0.0012508, -0.0031187, 0.0025219
5: 0.0031950, 0.0139095, 0.0033241, 0.0162060, -0.0124695, 0.0100235
6: -0.0102558, -0.0025416, -0.0121811, -0.0025370, -0.0077188, 0.0096396
7: -0.0163867, -0.0092872, -0.0172352, -0.0098209, -0.0064876, 0.0078719
8: -0.0149751, -0.0088170, -0.0151180, -0.0076559, -0.0073192, 0.0063010
9: -0.0035149, 0.0032580, -0.0051244, 0.0033073, -0.0068223, 0.0083823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008423, upper bound: 0.0008810
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008401, upper bound: 0.0008754
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0008203, 0.0080815, -0.0083442, 0.0083442
1: -0.0035921, 0.0009910, -0.0035921, 0.0009910, -0.0044504, 0.0044504
2: 0.0089864, 0.0167475, 0.0089864, 0.0167475, -0.0077611, 0.0077611
3: 1.0059075, 1.0071299, 1.0059075, 1.0071299, -0.0012224, 0.0012224
4: -0.0043676, -0.0018928, -0.0043676, -0.0018928, -0.0024747, 0.0024747
5: 0.0033575, 0.0137868, 0.0033575, 0.0137868, -0.0099889, 0.0099889
6: -0.0101534, -0.0025408, -0.0101534, -0.0025408, -0.0076126, 0.0076126
7: -0.0163678, -0.0096178, -0.0163678, -0.0096178, -0.0066899, 0.0066899
8: -0.0149543, -0.0089254, -0.0149543, -0.0089254, -0.0060289, 0.0060289
9: -0.0034489, 0.0032479, -0.0034489, 0.0032479, -0.0066968, 0.0066968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 70

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008693, upper bound: 0.0008783
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008692, upper bound: 0.0008692
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0010320, 0.0081786, -0.0084473, 0.0085414
1: -0.0035921, 0.0009910, -0.0036297, 0.0010522, -0.0045148, 0.0044678
2: 0.0089864, 0.0167475, 0.0089004, 0.0167598, -0.0077734, 0.0078471
3: 1.0059075, 1.0071299, 1.0058941, 1.0071546, -0.0012470, 0.0012358
4: -0.0043676, -0.0018928, -0.0043695, -0.0018634, -0.0025042, 0.0024767
5: 0.0033575, 0.0137868, 0.0031950, 0.0139095, -0.0101156, 0.0101402
6: -0.0101534, -0.0025408, -0.0102558, -0.0025416, -0.0076118, 0.0077150
7: -0.0163678, -0.0096178, -0.0163867, -0.0092872, -0.0070183, 0.0067089
8: -0.0149543, -0.0089254, -0.0149751, -0.0088170, -0.0061373, 0.0060496
9: -0.0034489, 0.0032479, -0.0035149, 0.0032580, -0.0067068, 0.0067629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 70

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008307, upper bound: 0.0008465
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008278, upper bound: 0.0008345
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010229, 0.0081786, -0.0007888, 0.0099003, -0.0103469, 0.0084131
1: -0.0036268, 0.0010522, -0.0035823, 0.0020921, -0.0055747, 0.0045126
2: 0.0089005, 0.0167598, 0.0071510, 0.0168926, -0.0079921, 0.0096089
3: 1.0058944, 1.0071493, 1.0057830, 1.0071132, -0.0012188, 0.0013664
4: -0.0043695, -0.0018634, -0.0043900, -0.0012828, -0.0030868, 0.0025266
5: 0.0032020, 0.0139095, 0.0033820, 0.0161074, -0.0124507, 0.0100894
6: -0.0102558, -0.0025419, -0.0120976, -0.0025387, -0.0077170, 0.0095558
7: -0.0163866, -0.0092956, -0.0171659, -0.0096454, -0.0066826, 0.0078078
8: -0.0149751, -0.0088245, -0.0151875, -0.0077523, -0.0072228, 0.0063630
9: -0.0035149, 0.0032580, -0.0050260, 0.0033586, -0.0068735, 0.0082839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008349
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008454, upper bound: 0.0008349
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010133, 0.0081786, -0.0007837, 0.0102666, -0.0107187, 0.0084093
1: -0.0036228, 0.0010522, -0.0035771, 0.0023161, -0.0058033, 0.0045101
2: 0.0089005, 0.0167598, 0.0067873, 0.0169267, -0.0080262, 0.0099726
3: 1.0058949, 1.0071378, 1.0057590, 1.0070903, -0.0011954, 0.0013788
4: -0.0043695, -0.0018634, -0.0043960, -0.0011615, -0.0032081, 0.0025326
5: 0.0032095, 0.0139095, 0.0033860, 0.0165740, -0.0129219, 0.0100865
6: -0.0102558, -0.0025426, -0.0124883, -0.0025403, -0.0077155, 0.0099457
7: -0.0163866, -0.0093005, -0.0173095, -0.0096460, -0.0066816, 0.0079463
8: -0.0149751, -0.0088389, -0.0152551, -0.0075303, -0.0074447, 0.0064162
9: -0.0035148, 0.0032580, -0.0053291, 0.0033921, -0.0069069, 0.0085870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008338, upper bound: 0.0008338
time: 1.00 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.48 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009695
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009746
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009699
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0009879, upper bound: 0.0009651
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009487
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009490
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008823
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008734
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008829
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008807
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008423
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008968, upper bound: 0.0008423
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008401
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008845, upper bound: 0.0008401
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008735, upper bound: 0.0009107
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008385, upper bound: 0.0008968
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008846
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008807, upper bound: 0.0009053
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008423, upper bound: 0.0008810
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008401, upper bound: 0.0008754
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008693, upper bound: 0.0008783
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008692, upper bound: 0.0008692
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008307, upper bound: 0.0008465
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008278, upper bound: 0.0008345
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008349
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008454, upper bound: 0.0008349
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.48
Output dim: 3, lower bound: -0.0008338, upper bound: 0.0008338

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0006523, 0.0099991, -0.0100518, 0.0100518
1: -0.0035917, 0.0021417, -0.0035917, 0.0021417, -0.0056016, 0.0056016
2: 0.0070180, 0.0168748, 0.0070180, 0.0168748, -0.0098568, 0.0098568
3: 1.0058836, 1.0071558, 1.0058836, 1.0071558, -0.0012722, 0.0012722
4: -0.0043845, -0.0012404, -0.0043845, -0.0012404, -0.0031441, 0.0031441
5: 0.0034829, 0.0162374, 0.0034829, 0.0162374, -0.0122819, 0.0122819
6: -0.0122079, -0.0025360, -0.0122079, -0.0025360, -0.0096719, 0.0096719
7: -0.0172598, -0.0101350, -0.0172598, -0.0101350, -0.0070568, 0.0070568
8: -0.0151080, -0.0076373, -0.0151080, -0.0076373, -0.0074707, 0.0074707
9: -0.0051578, 0.0033023, -0.0051578, 0.0033023, -0.0084601, 0.0084601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009696, upper bound: 0.0009651
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0006523, 0.0099991, -0.0102475, 0.0100297
1: -0.0036292, 0.0021306, -0.0035917, 0.0021417, -0.0056173, 0.0055916
2: 0.0070511, 0.0168794, 0.0070180, 0.0168748, -0.0098237, 0.0098614
3: 1.0058714, 1.0071830, 1.0058836, 1.0071558, -0.0012844, 0.0012994
4: -0.0043853, -0.0012508, -0.0043845, -0.0012404, -0.0031449, 0.0031337
5: 0.0033241, 0.0162060, 0.0034829, 0.0162374, -0.0124320, 0.0122511
6: -0.0121811, -0.0025370, -0.0122079, -0.0025360, -0.0096451, 0.0096709
7: -0.0172352, -0.0098209, -0.0172598, -0.0101350, -0.0070330, 0.0073705
8: -0.0151180, -0.0076559, -0.0151080, -0.0076373, -0.0074807, 0.0074521
9: -0.0051244, 0.0033073, -0.0051578, 0.0033023, -0.0084266, 0.0084652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009519, upper bound: 0.0009468
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009468
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006213, 0.0099991, -0.0008509, 0.0099755, -0.0099928, 0.0102380
1: -0.0035823, 0.0021416, -0.0036266, 0.0021305, -0.0055781, 0.0056135
2: 0.0070180, 0.0168748, 0.0070511, 0.0168794, -0.0098613, 0.0098237
3: 1.0058852, 1.0071396, 1.0058719, 1.0071779, -0.0012927, 0.0012677
4: -0.0043845, -0.0012404, -0.0043853, -0.0012508, -0.0031337, 0.0031449
5: 0.0035069, 0.0162374, 0.0033305, 0.0162060, -0.0122223, 0.0124245
6: -0.0122079, -0.0025368, -0.0121811, -0.0025373, -0.0096706, 0.0096443
7: -0.0172597, -0.0101618, -0.0172352, -0.0098284, -0.0073628, 0.0070060
8: -0.0151080, -0.0076387, -0.0151180, -0.0076571, -0.0074509, 0.0074792
9: -0.0051578, 0.0033023, -0.0051243, 0.0033073, -0.0084651, 0.0084266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009747
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0006148, 0.0104695, -0.0008401, 0.0099756, -0.0099934, 0.0107065
1: -0.0035763, 0.0024303, -0.0036220, 0.0021305, -0.0055781, 0.0059050
2: 0.0065562, 0.0169198, 0.0070511, 0.0168794, -0.0103232, 0.0098687
3: 1.0058498, 1.0071174, 1.0058727, 1.0071660, -0.0013162, 0.0012447
4: -0.0043918, -0.0010863, -0.0043853, -0.0012508, -0.0031409, 0.0032990
5: 0.0035121, 0.0168360, 0.0033390, 0.0162060, -0.0122228, 0.0130223
6: -0.0127089, -0.0025387, -0.0121811, -0.0025380, -0.0101708, 0.0096425
7: -0.0174431, -0.0101635, -0.0172352, -0.0098346, -0.0075406, 0.0070048
8: -0.0151858, -0.0073504, -0.0151180, -0.0076587, -0.0075271, 0.0077676
9: -0.0055453, 0.0033407, -0.0051243, 0.0033073, -0.0088527, 0.0084650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009467
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009698
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0006523, 0.0099991, -0.0100518, 0.0100518
1: -0.0035917, 0.0021417, -0.0035917, 0.0021417, -0.0056016, 0.0056016
2: 0.0070180, 0.0168748, 0.0070180, 0.0168748, -0.0098568, 0.0098568
3: 1.0058836, 1.0071558, 1.0058836, 1.0071558, -0.0012722, 0.0012722
4: -0.0043845, -0.0012404, -0.0043845, -0.0012404, -0.0031441, 0.0031441
5: 0.0034829, 0.0162374, 0.0034829, 0.0162374, -0.0122819, 0.0122819
6: -0.0122079, -0.0025360, -0.0122079, -0.0025360, -0.0096719, 0.0096719
7: -0.0172598, -0.0101350, -0.0172598, -0.0101350, -0.0070568, 0.0070568
8: -0.0151080, -0.0076373, -0.0151080, -0.0076373, -0.0074707, 0.0074707
9: -0.0051578, 0.0033023, -0.0051578, 0.0033023, -0.0084601, 0.0084601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009696, upper bound: 0.0009651
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0006523, 0.0099991, -0.0102475, 0.0100297
1: -0.0036292, 0.0021306, -0.0035917, 0.0021417, -0.0056173, 0.0055916
2: 0.0070511, 0.0168794, 0.0070180, 0.0168748, -0.0098237, 0.0098614
3: 1.0058714, 1.0071830, 1.0058836, 1.0071558, -0.0012844, 0.0012994
4: -0.0043853, -0.0012508, -0.0043845, -0.0012404, -0.0031449, 0.0031337
5: 0.0033241, 0.0162060, 0.0034829, 0.0162374, -0.0124320, 0.0122511
6: -0.0121811, -0.0025370, -0.0122079, -0.0025360, -0.0096451, 0.0096709
7: -0.0172352, -0.0098209, -0.0172598, -0.0101350, -0.0070330, 0.0073705
8: -0.0151180, -0.0076559, -0.0151080, -0.0076373, -0.0074807, 0.0074521
9: -0.0051244, 0.0033073, -0.0051578, 0.0033023, -0.0084266, 0.0084652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009519, upper bound: 0.0009468
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009468
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0008325, 0.0099755, -0.0008509, 0.0099755, -0.0100717, 0.0100955
1: -0.0036210, 0.0021305, -0.0036266, 0.0021305, -0.0055845, 0.0055937
2: 0.0070511, 0.0168794, 0.0070511, 0.0168794, -0.0098283, 0.0098283
3: 1.0058732, 1.0071677, 1.0058719, 1.0071779, -0.0013047, 0.0012958
4: -0.0043853, -0.0012508, -0.0043853, -0.0012508, -0.0031345, 0.0031345
5: 0.0033448, 0.0162060, 0.0033305, 0.0162060, -0.0122807, 0.0122993
6: -0.0121811, -0.0025379, -0.0121811, -0.0025373, -0.0096438, 0.0096432
7: -0.0172352, -0.0098452, -0.0172352, -0.0098284, -0.0073249, 0.0073079
8: -0.0151180, -0.0076594, -0.0151180, -0.0076571, -0.0074609, 0.0074586
9: -0.0051242, 0.0033073, -0.0051243, 0.0033073, -0.0084316, 0.0084317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009468
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009701, upper bound: 0.0009747
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0008134, 0.0102705, -0.0008401, 0.0099756, -0.0100674, 0.0103934
1: -0.0036125, 0.0023114, -0.0036220, 0.0021305, -0.0055827, 0.0057797
2: 0.0067613, 0.0169073, 0.0070511, 0.0168794, -0.0101181, 0.0098562
3: 1.0058534, 1.0071440, 1.0058727, 1.0071660, -0.0013126, 0.0012712
4: -0.0043902, -0.0011543, -0.0043853, -0.0012508, -0.0031394, 0.0032311
5: 0.0033597, 0.0165816, 0.0033390, 0.0162060, -0.0122773, 0.0126776
6: -0.0124955, -0.0025395, -0.0121811, -0.0025380, -0.0099575, 0.0096416
7: -0.0173510, -0.0098539, -0.0172352, -0.0098346, -0.0074349, 0.0073005
8: -0.0151701, -0.0074798, -0.0151180, -0.0076587, -0.0075114, 0.0076381
9: -0.0053681, 0.0033333, -0.0051243, 0.0033073, -0.0086754, 0.0084576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009468
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009701, upper bound: 0.0009701
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0008203, 0.0080815, -0.0081549, 0.0102628
1: -0.0035917, 0.0021417, -0.0035921, 0.0009910, -0.0044458, 0.0056085
2: 0.0070180, 0.0168748, 0.0089864, 0.0167475, -0.0097295, 0.0078884
3: 1.0058836, 1.0071558, 1.0059075, 1.0071299, -0.0012463, 0.0012482
4: -0.0043845, -0.0012404, -0.0043676, -0.0018928, -0.0024917, 0.0031272
5: 0.0034829, 0.0162374, 0.0033575, 0.0137868, -0.0098461, 0.0124410
6: -0.0122079, -0.0025360, -0.0101534, -0.0025408, -0.0096671, 0.0076173
7: -0.0172598, -0.0101350, -0.0163678, -0.0096178, -0.0075805, 0.0061688
8: -0.0151080, -0.0076373, -0.0149543, -0.0089254, -0.0061826, 0.0073170
9: -0.0051578, 0.0033023, -0.0034489, 0.0032479, -0.0084058, 0.0067511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 70

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009107, upper bound: 0.0008735
time: 1.03 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008734
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0008203, 0.0080815, -0.0083506, 0.0102411
1: -0.0036292, 0.0021306, -0.0035921, 0.0009910, -0.0044615, 0.0055990
2: 0.0070511, 0.0168794, 0.0089864, 0.0167475, -0.0096964, 0.0078930
3: 1.0058714, 1.0071830, 1.0059075, 1.0071299, -0.0012585, 0.0012754
4: -0.0043853, -0.0012508, -0.0043676, -0.0018928, -0.0024925, 0.0031167
5: 0.0033241, 0.0162060, 0.0033575, 0.0137868, -0.0099961, 0.0124104
6: -0.0121811, -0.0025370, -0.0101534, -0.0025408, -0.0096403, 0.0076164
7: -0.0172352, -0.0098209, -0.0163678, -0.0096178, -0.0075570, 0.0064824
8: -0.0151180, -0.0076559, -0.0149543, -0.0089254, -0.0061925, 0.0072984
9: -0.0051244, 0.0033073, -0.0034489, 0.0032479, -0.0083723, 0.0067562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 70

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008876, upper bound: 0.0008385
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008781, upper bound: 0.0008334
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0010320, 0.0081786, -0.0082542, 0.0104603
1: -0.0035917, 0.0021417, -0.0036297, 0.0010522, -0.0045092, 0.0056270
2: 0.0070180, 0.0168748, 0.0089004, 0.0167598, -0.0097418, 0.0079744
3: 1.0058836, 1.0071558, 1.0058941, 1.0071546, -0.0012710, 0.0012617
4: -0.0043845, -0.0012404, -0.0043695, -0.0018634, -0.0025211, 0.0031292
5: 0.0034829, 0.0162374, 0.0031950, 0.0139095, -0.0099703, 0.0125925
6: -0.0122079, -0.0025360, -0.0102558, -0.0025416, -0.0096664, 0.0077198
7: -0.0172598, -0.0101350, -0.0163867, -0.0092872, -0.0079089, 0.0061876
8: -0.0151080, -0.0076373, -0.0149751, -0.0088170, -0.0062910, 0.0073378
9: -0.0051578, 0.0033023, -0.0035149, 0.0032580, -0.0084158, 0.0068172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008734
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008807
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0010320, 0.0081786, -0.0083256, 0.0103213
1: -0.0036292, 0.0021306, -0.0036297, 0.0010522, -0.0045144, 0.0056046
2: 0.0070511, 0.0168794, 0.0089004, 0.0167598, -0.0097088, 0.0079789
3: 1.0058714, 1.0071830, 1.0058941, 1.0071546, -0.0012832, 0.0012889
4: -0.0043853, -0.0012508, -0.0043695, -0.0018634, -0.0025219, 0.0031187
5: 0.0033241, 0.0162060, 0.0031950, 0.0139095, -0.0100235, 0.0124695
6: -0.0121811, -0.0025370, -0.0102558, -0.0025416, -0.0096396, 0.0077188
7: -0.0172352, -0.0098209, -0.0163867, -0.0092872, -0.0078719, 0.0064876
8: -0.0151180, -0.0076559, -0.0149751, -0.0088170, -0.0063010, 0.0073192
9: -0.0051244, 0.0033073, -0.0035149, 0.0032580, -0.0083823, 0.0068223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008384
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008334
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0007886, 0.0094634, -0.0095341, 0.0102264
1: -0.0035917, 0.0021417, -0.0035823, 0.0018265, -0.0052897, 0.0056026
2: 0.0070180, 0.0168748, 0.0075914, 0.0168577, -0.0098397, 0.0092834
3: 1.0058836, 1.0071558, 1.0058134, 1.0071132, -0.0012296, 0.0013424
4: -0.0043845, -0.0012404, -0.0043846, -0.0014291, -0.0029554, 0.0031442
5: 0.0034829, 0.0162374, 0.0033821, 0.0155502, -0.0116077, 0.0124126
6: -0.0122079, -0.0025360, -0.0116309, -0.0025395, -0.0096685, 0.0090948
7: -0.0172598, -0.0101350, -0.0169744, -0.0096454, -0.0075528, 0.0067749
8: -0.0151080, -0.0076373, -0.0151318, -0.0080445, -0.0070635, 0.0074945
9: -0.0051578, 0.0033023, -0.0046476, 0.0033319, -0.0084898, 0.0079499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008384
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008423
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0007885, 0.0091390, -0.0093997, 0.0102120
1: -0.0036292, 0.0021306, -0.0035823, 0.0016317, -0.0051064, 0.0055940
2: 0.0070511, 0.0168794, 0.0079253, 0.0168329, -0.0097819, 0.0089541
3: 1.0058714, 1.0071830, 1.0058388, 1.0071132, -0.0012418, 0.0013442
4: -0.0043853, -0.0012508, -0.0043808, -0.0015398, -0.0028456, 0.0031300
5: 0.0033241, 0.0162060, 0.0033822, 0.0151356, -0.0113396, 0.0123878
6: -0.0121811, -0.0025370, -0.0112832, -0.0025404, -0.0096407, 0.0087462
7: -0.0172352, -0.0098209, -0.0168192, -0.0096454, -0.0075301, 0.0069328
8: -0.0151180, -0.0076559, -0.0150928, -0.0082672, -0.0068508, 0.0074369
9: -0.0051244, 0.0033073, -0.0043549, 0.0033136, -0.0084379, 0.0076622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008968, upper bound: 0.0008424
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008845, upper bound: 0.0008401
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0007836, 0.0098395, -0.0099170, 0.0102292
1: -0.0035917, 0.0021417, -0.0035771, 0.0020572, -0.0055256, 0.0056032
2: 0.0070180, 0.0168748, 0.0072204, 0.0168929, -0.0098749, 0.0096544
3: 1.0058836, 1.0071558, 1.0057887, 1.0070903, -0.0012068, 0.0013671
4: -0.0043845, -0.0012404, -0.0043909, -0.0013052, -0.0030793, 0.0031505
5: 0.0034829, 0.0162374, 0.0033861, 0.0160290, -0.0120928, 0.0124149
6: -0.0122079, -0.0025360, -0.0120316, -0.0025410, -0.0096669, 0.0094956
7: -0.0172598, -0.0101350, -0.0171210, -0.0096460, -0.0075525, 0.0069221
8: -0.0151080, -0.0076373, -0.0152018, -0.0078199, -0.0072881, 0.0075645
9: -0.0051578, 0.0033023, -0.0049573, 0.0033665, -0.0085243, 0.0082596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008334
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008401
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0007834, 0.0094519, -0.0097251, 0.0102106
1: -0.0036292, 0.0021306, -0.0035771, 0.0018241, -0.0053043, 0.0055926
2: 0.0070511, 0.0168794, 0.0076144, 0.0168644, -0.0098133, 0.0092649
3: 1.0058714, 1.0071830, 1.0058186, 1.0070903, -0.0012189, 0.0013644
4: -0.0043853, -0.0012508, -0.0043859, -0.0014359, -0.0029494, 0.0031351
5: 0.0033241, 0.0162060, 0.0033862, 0.0155341, -0.0117476, 0.0123868
6: -0.0121811, -0.0025370, -0.0116169, -0.0025419, -0.0096392, 0.0090799
7: -0.0172352, -0.0098209, -0.0169441, -0.0096460, -0.0075293, 0.0070581
8: -0.0151180, -0.0076559, -0.0151478, -0.0080828, -0.0070351, 0.0074919
9: -0.0051244, 0.0033073, -0.0046153, 0.0033412, -0.0084656, 0.0079226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008845, upper bound: 0.0008401
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008845, upper bound: 0.0008401
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0006523, 0.0099991, -0.0102628, 0.0081549
1: -0.0035921, 0.0009910, -0.0035917, 0.0021417, -0.0056085, 0.0044458
2: 0.0089864, 0.0167475, 0.0070180, 0.0168748, -0.0078884, 0.0097295
3: 1.0059075, 1.0071299, 1.0058836, 1.0071558, -0.0012482, 0.0012463
4: -0.0043676, -0.0018928, -0.0043845, -0.0012404, -0.0031272, 0.0024917
5: 0.0033575, 0.0137868, 0.0034829, 0.0162374, -0.0124410, 0.0098461
6: -0.0101534, -0.0025408, -0.0122079, -0.0025360, -0.0076173, 0.0096671
7: -0.0163678, -0.0096178, -0.0172598, -0.0101350, -0.0061688, 0.0075805
8: -0.0149543, -0.0089254, -0.0151080, -0.0076373, -0.0073170, 0.0061826
9: -0.0034489, 0.0032479, -0.0051578, 0.0033023, -0.0067511, 0.0084058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008823, upper bound: 0.0009053
time: 1.29 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010320, 0.0081786, -0.0006523, 0.0099991, -0.0104603, 0.0082542
1: -0.0036297, 0.0010522, -0.0035917, 0.0021417, -0.0056270, 0.0045092
2: 0.0089004, 0.0167598, 0.0070180, 0.0168748, -0.0079744, 0.0097418
3: 1.0058941, 1.0071546, 1.0058836, 1.0071558, -0.0012617, 0.0012710
4: -0.0043695, -0.0018634, -0.0043845, -0.0012404, -0.0031292, 0.0025211
5: 0.0031950, 0.0139095, 0.0034829, 0.0162374, -0.0125925, 0.0099703
6: -0.0102558, -0.0025416, -0.0122079, -0.0025360, -0.0077198, 0.0096664
7: -0.0163867, -0.0092872, -0.0172598, -0.0101350, -0.0061876, 0.0079089
8: -0.0149751, -0.0088170, -0.0151080, -0.0076373, -0.0073378, 0.0062910
9: -0.0035149, 0.0032580, -0.0051578, 0.0033023, -0.0068172, 0.0084158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008823, upper bound: 0.0009053
time: 1.29 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0007882, 0.0080815, -0.0008512, 0.0099756, -0.0102044, 0.0083413
1: -0.0035823, 0.0009909, -0.0036267, 0.0021305, -0.0055843, 0.0044579
2: 0.0089866, 0.0167475, 0.0070511, 0.0168794, -0.0078928, 0.0096964
3: 1.0059092, 1.0071132, 1.0058719, 1.0071783, -0.0012691, 0.0012413
4: -0.0043676, -0.0018929, -0.0043853, -0.0012508, -0.0031167, 0.0024925
5: 0.0033824, 0.0137868, 0.0033303, 0.0162060, -0.0123818, 0.0099889
6: -0.0101534, -0.0025417, -0.0121811, -0.0025373, -0.0076161, 0.0096394
7: -0.0163677, -0.0096455, -0.0172352, -0.0098281, -0.0064750, 0.0075293
8: -0.0149543, -0.0089442, -0.0151180, -0.0076571, -0.0072972, 0.0061738
9: -0.0034487, 0.0032479, -0.0051243, 0.0033073, -0.0067560, 0.0083722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008810
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008968
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0007831, 0.0084850, -0.0008425, 0.0099755, -0.0102073, 0.0087490
1: -0.0035771, 0.0012392, -0.0036229, 0.0021305, -0.0055849, 0.0047098
2: 0.0085887, 0.0167879, 0.0070511, 0.0168794, -0.0082907, 0.0097368
3: 1.0058836, 1.0070903, 1.0058724, 1.0071681, -0.0012845, 0.0012180
4: -0.0043745, -0.0017598, -0.0043853, -0.0012508, -0.0031237, 0.0026255
5: 0.0033864, 0.0143007, 0.0033371, 0.0162060, -0.0123841, 0.0105067
6: -0.0105835, -0.0025433, -0.0121811, -0.0025379, -0.0080456, 0.0096378
7: -0.0165253, -0.0096461, -0.0172352, -0.0098329, -0.0066285, 0.0075289
8: -0.0150299, -0.0087157, -0.0151180, -0.0076584, -0.0073715, 0.0064023
9: -0.0037815, 0.0032854, -0.0051243, 0.0033073, -0.0070888, 0.0084097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008753
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008846
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0006523, 0.0099991, -0.0102628, 0.0081549
1: -0.0035921, 0.0009910, -0.0035917, 0.0021417, -0.0056085, 0.0044458
2: 0.0089864, 0.0167475, 0.0070180, 0.0168748, -0.0078884, 0.0097295
3: 1.0059075, 1.0071299, 1.0058836, 1.0071558, -0.0012482, 0.0012463
4: -0.0043676, -0.0018928, -0.0043845, -0.0012404, -0.0031272, 0.0024917
5: 0.0033575, 0.0137868, 0.0034829, 0.0162374, -0.0124410, 0.0098461
6: -0.0101534, -0.0025408, -0.0122079, -0.0025360, -0.0076173, 0.0096671
7: -0.0163678, -0.0096178, -0.0172598, -0.0101350, -0.0061688, 0.0075805
8: -0.0149543, -0.0089254, -0.0151080, -0.0076373, -0.0073170, 0.0061826
9: -0.0034489, 0.0032479, -0.0051578, 0.0033023, -0.0067511, 0.0084058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008823, upper bound: 0.0009053
time: 1.30 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010320, 0.0081786, -0.0006523, 0.0099991, -0.0104603, 0.0082542
1: -0.0036297, 0.0010522, -0.0035917, 0.0021417, -0.0056270, 0.0045092
2: 0.0089004, 0.0167598, 0.0070180, 0.0168748, -0.0079744, 0.0097418
3: 1.0058941, 1.0071546, 1.0058836, 1.0071558, -0.0012617, 0.0012710
4: -0.0043695, -0.0018634, -0.0043845, -0.0012404, -0.0031292, 0.0025211
5: 0.0031950, 0.0139095, 0.0034829, 0.0162374, -0.0125925, 0.0099703
6: -0.0102558, -0.0025416, -0.0122079, -0.0025360, -0.0077198, 0.0096664
7: -0.0163867, -0.0092872, -0.0172598, -0.0101350, -0.0061876, 0.0079089
8: -0.0149751, -0.0088170, -0.0151080, -0.0076373, -0.0073378, 0.0062910
9: -0.0035149, 0.0032580, -0.0051578, 0.0033023, -0.0068172, 0.0084158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008829, upper bound: 0.0009053
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008807, upper bound: 0.0009053
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0010040, 0.0081786, -0.0008512, 0.0099756, -0.0102862, 0.0083152
1: -0.0036211, 0.0010522, -0.0036267, 0.0021305, -0.0055900, 0.0045102
2: 0.0089005, 0.0167598, 0.0070511, 0.0168794, -0.0079788, 0.0097087
3: 1.0058955, 1.0071385, 1.0058719, 1.0071783, -0.0012828, 0.0012666
4: -0.0043695, -0.0018634, -0.0043853, -0.0012508, -0.0031187, 0.0025219
5: 0.0032166, 0.0139095, 0.0033303, 0.0162060, -0.0124422, 0.0100153
6: -0.0102558, -0.0025425, -0.0121811, -0.0025373, -0.0077185, 0.0096386
7: -0.0163866, -0.0093129, -0.0172352, -0.0098281, -0.0064801, 0.0078462
8: -0.0149751, -0.0088398, -0.0151180, -0.0076571, -0.0073180, 0.0062782
9: -0.0035147, 0.0032580, -0.0051243, 0.0033073, -0.0068220, 0.0083823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008423, upper bound: 0.0008810
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008423, upper bound: 0.0008969
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0009874, 0.0084447, -0.0008425, 0.0099755, -0.0102841, 0.0085918
1: -0.0036136, 0.0012150, -0.0036229, 0.0021305, -0.0055884, 0.0046798
2: 0.0086357, 0.0167854, 0.0070511, 0.0168794, -0.0082437, 0.0097343
3: 1.0058812, 1.0071148, 1.0058724, 1.0071681, -0.0012869, 0.0012424
4: -0.0043740, -0.0017749, -0.0043853, -0.0012508, -0.0031232, 0.0026105
5: 0.0032297, 0.0142486, 0.0033371, 0.0162060, -0.0124405, 0.0103632
6: -0.0105399, -0.0025439, -0.0121811, -0.0025379, -0.0080019, 0.0096372
7: -0.0164950, -0.0093188, -0.0172352, -0.0098329, -0.0065844, 0.0078406
8: -0.0150245, -0.0087088, -0.0151180, -0.0076584, -0.0073661, 0.0064092
9: -0.0037369, 0.0032809, -0.0051243, 0.0033073, -0.0070443, 0.0084052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008401, upper bound: 0.0008753
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008401, upper bound: 0.0008846
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0008203, 0.0080815, -0.0083442, 0.0083442
1: -0.0035921, 0.0009910, -0.0035921, 0.0009910, -0.0044504, 0.0044504
2: 0.0089864, 0.0167475, 0.0089864, 0.0167475, -0.0077611, 0.0077611
3: 1.0059075, 1.0071299, 1.0059075, 1.0071299, -0.0012224, 0.0012224
4: -0.0043676, -0.0018928, -0.0043676, -0.0018928, -0.0024747, 0.0024747
5: 0.0033575, 0.0137868, 0.0033575, 0.0137868, -0.0099889, 0.0099889
6: -0.0101534, -0.0025408, -0.0101534, -0.0025408, -0.0076126, 0.0076126
7: -0.0163678, -0.0096178, -0.0163678, -0.0096178, -0.0066899, 0.0066899
8: -0.0149543, -0.0089254, -0.0149543, -0.0089254, -0.0060289, 0.0060289
9: -0.0034489, 0.0032479, -0.0034489, 0.0032479, -0.0066968, 0.0066968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 70

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008783, upper bound: 0.0008693
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008692, upper bound: 0.0008692
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010320, 0.0081786, -0.0008203, 0.0080815, -0.0085414, 0.0084473
1: -0.0036297, 0.0010522, -0.0035921, 0.0009910, -0.0044678, 0.0045148
2: 0.0089004, 0.0167598, 0.0089864, 0.0167475, -0.0078471, 0.0077734
3: 1.0058941, 1.0071546, 1.0059075, 1.0071299, -0.0012358, 0.0012470
4: -0.0043695, -0.0018634, -0.0043676, -0.0018928, -0.0024767, 0.0025042
5: 0.0031950, 0.0139095, 0.0033575, 0.0137868, -0.0101402, 0.0101156
6: -0.0102558, -0.0025416, -0.0101534, -0.0025408, -0.0077150, 0.0076118
7: -0.0163867, -0.0092872, -0.0163678, -0.0096178, -0.0067089, 0.0070183
8: -0.0149751, -0.0088170, -0.0149543, -0.0089254, -0.0060496, 0.0061373
9: -0.0035149, 0.0032580, -0.0034489, 0.0032479, -0.0067629, 0.0067068

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 70

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008452, upper bound: 0.0008307
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008330, upper bound: 0.0008278
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0007882, 0.0080815, -0.0010229, 0.0081786, -0.0084100, 0.0085307
1: -0.0035823, 0.0009909, -0.0036268, 0.0010522, -0.0045003, 0.0044637
2: 0.0089866, 0.0167475, 0.0089005, 0.0167598, -0.0077733, 0.0078470
3: 1.0059092, 1.0071132, 1.0058944, 1.0071493, -0.0012401, 0.0012188
4: -0.0043676, -0.0018929, -0.0043695, -0.0018634, -0.0025042, 0.0024767
5: 0.0033824, 0.0137868, 0.0032020, 0.0139095, -0.0100865, 0.0101320
6: -0.0101534, -0.0025417, -0.0102558, -0.0025419, -0.0076115, 0.0077141
7: -0.0163677, -0.0096455, -0.0163866, -0.0092956, -0.0070100, 0.0066812
8: -0.0149543, -0.0089442, -0.0149751, -0.0088245, -0.0061298, 0.0060309
9: -0.0034487, 0.0032479, -0.0035149, 0.0032580, -0.0067066, 0.0067628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008304, upper bound: 0.0008357
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008304, upper bound: 0.0008454
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0007831, 0.0084850, -0.0010133, 0.0081786, -0.0084106, 0.0089403
1: -0.0035771, 0.0012392, -0.0036228, 0.0010522, -0.0045001, 0.0047157
2: 0.0085887, 0.0167879, 0.0089005, 0.0167598, -0.0081712, 0.0078874
3: 1.0058836, 1.0070903, 1.0058949, 1.0071378, -0.0012542, 0.0011954
4: -0.0043745, -0.0017598, -0.0043695, -0.0018634, -0.0025111, 0.0026097
5: 0.0033864, 0.0143007, 0.0032095, 0.0139095, -0.0100870, 0.0106509
6: -0.0105835, -0.0025433, -0.0102558, -0.0025426, -0.0080410, 0.0077125
7: -0.0165253, -0.0096461, -0.0163866, -0.0093005, -0.0071629, 0.0066808
8: -0.0150299, -0.0087157, -0.0149751, -0.0088389, -0.0061909, 0.0062594
9: -0.0037815, 0.0032854, -0.0035148, 0.0032580, -0.0070394, 0.0068002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008277
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0007886, 0.0094599, -0.0097212, 0.0083072
1: -0.0035921, 0.0009910, -0.0035823, 0.0018244, -0.0052923, 0.0044445
2: 0.0089864, 0.0167475, 0.0075950, 0.0168574, -0.0078710, 0.0091525
3: 1.0059075, 1.0071299, 1.0058137, 1.0071132, -0.0012057, 0.0013162
4: -0.0043676, -0.0018928, -0.0043845, -0.0014303, -0.0029372, 0.0024917
5: 0.0033575, 0.0137868, 0.0033821, 0.0155457, -0.0117464, 0.0099600
6: -0.0101534, -0.0025408, -0.0116271, -0.0025395, -0.0076139, 0.0090863
7: -0.0163678, -0.0096178, -0.0169729, -0.0096454, -0.0066622, 0.0072945
8: -0.0149543, -0.0089254, -0.0151313, -0.0080468, -0.0069075, 0.0062059
9: -0.0034489, 0.0032479, -0.0046446, 0.0033317, -0.0067806, 0.0078925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008305
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008349
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0010320, 0.0081786, -0.0007885, 0.0091225, -0.0095787, 0.0084130
1: -0.0036297, 0.0010522, -0.0035823, 0.0016217, -0.0051041, 0.0045094
2: 0.0089004, 0.0167598, 0.0079420, 0.0168317, -0.0079313, 0.0088178
3: 1.0058941, 1.0071546, 1.0058401, 1.0071132, -0.0012192, 0.0013145
4: -0.0043695, -0.0018634, -0.0043806, -0.0015453, -0.0028242, 0.0025172
5: 0.0031950, 0.0139095, 0.0033822, 0.0151145, -0.0114656, 0.0100894
6: -0.0102558, -0.0025416, -0.0112656, -0.0025404, -0.0077154, 0.0087241
7: -0.0163867, -0.0092872, -0.0168118, -0.0096454, -0.0066826, 0.0074614
8: -0.0149751, -0.0088170, -0.0150907, -0.0082782, -0.0066969, 0.0062737
9: -0.0035149, 0.0032580, -0.0043405, 0.0033126, -0.0068276, 0.0075984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008454, upper bound: 0.0008349
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008338, upper bound: 0.0008338
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0007836, 0.0098360, -0.0101045, 0.0083078
1: -0.0035921, 0.0009910, -0.0035771, 0.0020550, -0.0055282, 0.0044443
2: 0.0089864, 0.0167475, 0.0072239, 0.0168926, -0.0079062, 0.0095236
3: 1.0059075, 1.0071299, 1.0057890, 1.0070903, -0.0011828, 0.0013409
4: -0.0043676, -0.0018928, -0.0043908, -0.0013064, -0.0030611, 0.0024980
5: 0.0033575, 0.0137868, 0.0033861, 0.0160245, -0.0122318, 0.0099605
6: -0.0101534, -0.0025408, -0.0120279, -0.0025410, -0.0076124, 0.0094871
7: -0.0163678, -0.0096178, -0.0171195, -0.0096460, -0.0066618, 0.0074418
8: -0.0149543, -0.0089254, -0.0152013, -0.0078223, -0.0071320, 0.0062759
9: -0.0034489, 0.0032479, -0.0049543, 0.0033663, -0.0068151, 0.0082023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008278
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
time: 1.04 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0010320, 0.0081786, -0.0007834, 0.0094349, -0.0099014, 0.0084092
1: -0.0036297, 0.0010522, -0.0035771, 0.0018138, -0.0053014, 0.0045069
2: 0.0089004, 0.0167598, 0.0076318, 0.0168631, -0.0079626, 0.0091281
3: 1.0058941, 1.0071546, 1.0058198, 1.0070903, -0.0011963, 0.0013348
4: -0.0043695, -0.0018634, -0.0043857, -0.0014417, -0.0029279, 0.0025223
5: 0.0031950, 0.0139095, 0.0033862, 0.0155124, -0.0118712, 0.0100864
6: -0.0102558, -0.0025416, -0.0115987, -0.0025419, -0.0077139, 0.0090572
7: -0.0163867, -0.0092872, -0.0169364, -0.0096461, -0.0066817, 0.0075865
8: -0.0149751, -0.0088170, -0.0151457, -0.0080943, -0.0068808, 0.0063287
9: -0.0035149, 0.0032580, -0.0046004, 0.0033402, -0.0068551, 0.0078583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008338, upper bound: 0.0008338
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008338, upper bound: 0.0008338
time: 0.97 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.59 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009696, upper bound: 0.0009651
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009519, upper bound: 0.0009468
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009468
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009747
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009467
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009698
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009696, upper bound: 0.0009651
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009519, upper bound: 0.0009468
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009468
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009468
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009701, upper bound: 0.0009747
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009468
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009701, upper bound: 0.0009701
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009107, upper bound: 0.0008735
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008734
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008876, upper bound: 0.0008385
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008781, upper bound: 0.0008334
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008734
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008807
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008384
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008334
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008384
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008423
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008968, upper bound: 0.0008424
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008845, upper bound: 0.0008401
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008334
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008401
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008845, upper bound: 0.0008401
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008845, upper bound: 0.0008401
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008823, upper bound: 0.0009053
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008823, upper bound: 0.0009053
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008810
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008968
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008753
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008846
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008823, upper bound: 0.0009053
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008829, upper bound: 0.0009053
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008807, upper bound: 0.0009053
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008423, upper bound: 0.0008810
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008423, upper bound: 0.0008969
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008401, upper bound: 0.0008753
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008401, upper bound: 0.0008846
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008783, upper bound: 0.0008693
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008692, upper bound: 0.0008692
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008452, upper bound: 0.0008307
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008330, upper bound: 0.0008278
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008304, upper bound: 0.0008357
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008304, upper bound: 0.0008454
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008277
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008305
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008349
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008454, upper bound: 0.0008349
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008338, upper bound: 0.0008338
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008278
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008338, upper bound: 0.0008338
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.59
Output dim: 3, lower bound: -0.0008338, upper bound: 0.0008338

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0006523, 0.0099991, -0.0100518, 0.0100518
1: -0.0035917, 0.0021417, -0.0035917, 0.0021417, -0.0056016, 0.0056016
2: 0.0070180, 0.0168748, 0.0070180, 0.0168748, -0.0098568, 0.0098568
3: 1.0058836, 1.0071558, 1.0058836, 1.0071558, -0.0012722, 0.0012722
4: -0.0043845, -0.0012404, -0.0043845, -0.0012404, -0.0031441, 0.0031441
5: 0.0034829, 0.0162374, 0.0034829, 0.0162374, -0.0122819, 0.0122819
6: -0.0122079, -0.0025360, -0.0122079, -0.0025360, -0.0096719, 0.0096719
7: -0.0172598, -0.0101350, -0.0172598, -0.0101350, -0.0070568, 0.0070568
8: -0.0151080, -0.0076373, -0.0151080, -0.0076373, -0.0074707, 0.0074707
9: -0.0051578, 0.0033023, -0.0051578, 0.0033023, -0.0084601, 0.0084601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009696
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0008592, 0.0099755, -0.0100297, 0.0102475
1: -0.0035917, 0.0021417, -0.0036292, 0.0021306, -0.0055916, 0.0056173
2: 0.0070180, 0.0168748, 0.0070511, 0.0168794, -0.0098614, 0.0098237
3: 1.0058836, 1.0071558, 1.0058714, 1.0071830, -0.0012994, 0.0012844
4: -0.0043845, -0.0012404, -0.0043853, -0.0012508, -0.0031337, 0.0031449
5: 0.0034829, 0.0162374, 0.0033241, 0.0162060, -0.0122511, 0.0124320
6: -0.0122079, -0.0025360, -0.0121811, -0.0025370, -0.0096709, 0.0096451
7: -0.0172598, -0.0101350, -0.0172352, -0.0098209, -0.0073705, 0.0070330
8: -0.0151080, -0.0076373, -0.0151180, -0.0076559, -0.0074521, 0.0074807
9: -0.0051578, 0.0033023, -0.0051244, 0.0033073, -0.0084652, 0.0084266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009519
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009504
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0008509, 0.0099755, -0.0006213, 0.0099991, -0.0102380, 0.0099928
1: -0.0036266, 0.0021305, -0.0035823, 0.0021416, -0.0056135, 0.0055781
2: 0.0070511, 0.0168794, 0.0070180, 0.0168748, -0.0098237, 0.0098613
3: 1.0058719, 1.0071779, 1.0058852, 1.0071396, -0.0012677, 0.0012927
4: -0.0043853, -0.0012508, -0.0043845, -0.0012404, -0.0031449, 0.0031337
5: 0.0033305, 0.0162060, 0.0035069, 0.0162374, -0.0124245, 0.0122223
6: -0.0121811, -0.0025373, -0.0122079, -0.0025368, -0.0096443, 0.0096706
7: -0.0172352, -0.0098284, -0.0172597, -0.0101618, -0.0070060, 0.0073628
8: -0.0151180, -0.0076571, -0.0151080, -0.0076387, -0.0074792, 0.0074509
9: -0.0051243, 0.0033073, -0.0051578, 0.0033023, -0.0084266, 0.0084651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009469, upper bound: 0.0009468
time: 1.08 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009469, upper bound: 0.0009468
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0008401, 0.0099756, -0.0006148, 0.0104695, -0.0107065, 0.0099934
1: -0.0036220, 0.0021305, -0.0035763, 0.0024303, -0.0059050, 0.0055781
2: 0.0070511, 0.0168794, 0.0065562, 0.0169198, -0.0098687, 0.0103232
3: 1.0058727, 1.0071660, 1.0058498, 1.0071174, -0.0012447, 0.0013162
4: -0.0043853, -0.0012508, -0.0043918, -0.0010863, -0.0032990, 0.0031409
5: 0.0033390, 0.0162060, 0.0035121, 0.0168360, -0.0130223, 0.0122228
6: -0.0121811, -0.0025380, -0.0127089, -0.0025387, -0.0096425, 0.0101708
7: -0.0172352, -0.0098346, -0.0174431, -0.0101635, -0.0070048, 0.0075406
8: -0.0151180, -0.0076587, -0.0151858, -0.0073504, -0.0077676, 0.0075271
9: -0.0051243, 0.0033073, -0.0055453, 0.0033407, -0.0084650, 0.0088527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009468, upper bound: 0.0009468
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006213, 0.0099991, -0.0006523, 0.0099991, -0.0100149, 0.0100518
1: -0.0035823, 0.0021416, -0.0035917, 0.0021417, -0.0055880, 0.0056016
2: 0.0070180, 0.0168748, 0.0070180, 0.0168748, -0.0098567, 0.0098568
3: 1.0058852, 1.0071396, 1.0058836, 1.0071558, -0.0012705, 0.0012560
4: -0.0043845, -0.0012404, -0.0043845, -0.0012404, -0.0031441, 0.0031441
5: 0.0035069, 0.0162374, 0.0034829, 0.0162374, -0.0122531, 0.0122819
6: -0.0122079, -0.0025368, -0.0122079, -0.0025360, -0.0096719, 0.0096711
7: -0.0172597, -0.0101618, -0.0172598, -0.0101350, -0.0070568, 0.0070299
8: -0.0151080, -0.0076387, -0.0151080, -0.0076373, -0.0074707, 0.0074693
9: -0.0051578, 0.0033023, -0.0051578, 0.0033023, -0.0084600, 0.0084601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006213, 0.0099991, -0.0008592, 0.0099755, -0.0099928, 0.0102475
1: -0.0035823, 0.0021416, -0.0036292, 0.0021306, -0.0055781, 0.0056173
2: 0.0070180, 0.0168748, 0.0070511, 0.0168794, -0.0098613, 0.0098237
3: 1.0058852, 1.0071396, 1.0058714, 1.0071830, -0.0012977, 0.0012681
4: -0.0043845, -0.0012404, -0.0043853, -0.0012508, -0.0031337, 0.0031449
5: 0.0035069, 0.0162374, 0.0033241, 0.0162060, -0.0122223, 0.0124320
6: -0.0122079, -0.0025368, -0.0121811, -0.0025370, -0.0096709, 0.0096443
7: -0.0172597, -0.0101618, -0.0172352, -0.0098209, -0.0073704, 0.0070060
8: -0.0151080, -0.0076387, -0.0151180, -0.0076559, -0.0074521, 0.0074792
9: -0.0051578, 0.0033023, -0.0051244, 0.0033073, -0.0084651, 0.0084266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009747
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009468, upper bound: 0.0009747
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0006148, 0.0104695, -0.0006523, 0.0099991, -0.0100156, 0.0105267
1: -0.0035763, 0.0024303, -0.0035917, 0.0021417, -0.0055881, 0.0058952
2: 0.0065562, 0.0169198, 0.0070180, 0.0168748, -0.0103186, 0.0099018
3: 1.0058498, 1.0071174, 1.0058836, 1.0071558, -0.0013059, 0.0012338
4: -0.0043918, -0.0010863, -0.0043845, -0.0012404, -0.0031514, 0.0032982
5: 0.0035121, 0.0168360, 0.0034829, 0.0162374, -0.0122537, 0.0128847
6: -0.0127089, -0.0025387, -0.0122079, -0.0025360, -0.0101729, 0.0096693
7: -0.0174431, -0.0101635, -0.0172598, -0.0101350, -0.0072405, 0.0070287
8: -0.0151858, -0.0073504, -0.0151080, -0.0076373, -0.0075486, 0.0077576
9: -0.0055453, 0.0033407, -0.0051578, 0.0033023, -0.0088476, 0.0084986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009467
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0006148, 0.0104695, -0.0008592, 0.0099755, -0.0099935, 0.0107224
1: -0.0035763, 0.0024303, -0.0036292, 0.0021306, -0.0055782, 0.0059109
2: 0.0065562, 0.0169198, 0.0070511, 0.0168794, -0.0103232, 0.0098687
3: 1.0058498, 1.0071174, 1.0058714, 1.0071830, -0.0013331, 0.0012460
4: -0.0043918, -0.0010863, -0.0043853, -0.0012508, -0.0031410, 0.0032990
5: 0.0035121, 0.0168360, 0.0033241, 0.0162060, -0.0122229, 0.0130347
6: -0.0127089, -0.0025387, -0.0121811, -0.0025370, -0.0101719, 0.0096425
7: -0.0174431, -0.0101635, -0.0172352, -0.0098209, -0.0075542, 0.0070048
8: -0.0151858, -0.0073504, -0.0151180, -0.0076559, -0.0075300, 0.0077676
9: -0.0055453, 0.0033407, -0.0051244, 0.0033073, -0.0088527, 0.0084651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009468, upper bound: 0.0009699
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009699
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0006523, 0.0099991, -0.0100518, 0.0100518
1: -0.0035917, 0.0021417, -0.0035917, 0.0021417, -0.0056016, 0.0056016
2: 0.0070180, 0.0168748, 0.0070180, 0.0168748, -0.0098568, 0.0098568
3: 1.0058836, 1.0071558, 1.0058836, 1.0071558, -0.0012722, 0.0012722
4: -0.0043845, -0.0012404, -0.0043845, -0.0012404, -0.0031441, 0.0031441
5: 0.0034829, 0.0162374, 0.0034829, 0.0162374, -0.0122819, 0.0122819
6: -0.0122079, -0.0025360, -0.0122079, -0.0025360, -0.0096719, 0.0096719
7: -0.0172598, -0.0101350, -0.0172598, -0.0101350, -0.0070568, 0.0070568
8: -0.0151080, -0.0076373, -0.0151080, -0.0076373, -0.0074707, 0.0074707
9: -0.0051578, 0.0033023, -0.0051578, 0.0033023, -0.0084601, 0.0084601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009696
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0008592, 0.0099755, -0.0100297, 0.0102475
1: -0.0035917, 0.0021417, -0.0036292, 0.0021306, -0.0055916, 0.0056173
2: 0.0070180, 0.0168748, 0.0070511, 0.0168794, -0.0098614, 0.0098237
3: 1.0058836, 1.0071558, 1.0058714, 1.0071830, -0.0012994, 0.0012844
4: -0.0043845, -0.0012404, -0.0043853, -0.0012508, -0.0031337, 0.0031449
5: 0.0034829, 0.0162374, 0.0033241, 0.0162060, -0.0122511, 0.0124320
6: -0.0122079, -0.0025360, -0.0121811, -0.0025370, -0.0096709, 0.0096451
7: -0.0172598, -0.0101350, -0.0172352, -0.0098209, -0.0073705, 0.0070330
8: -0.0151080, -0.0076373, -0.0151180, -0.0076559, -0.0074521, 0.0074807
9: -0.0051578, 0.0033023, -0.0051244, 0.0033073, -0.0084652, 0.0084266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009519
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009504
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0008509, 0.0099755, -0.0006213, 0.0099991, -0.0102380, 0.0099928
1: -0.0036266, 0.0021305, -0.0035823, 0.0021416, -0.0056135, 0.0055781
2: 0.0070511, 0.0168794, 0.0070180, 0.0168748, -0.0098237, 0.0098613
3: 1.0058719, 1.0071779, 1.0058852, 1.0071396, -0.0012677, 0.0012927
4: -0.0043853, -0.0012508, -0.0043845, -0.0012404, -0.0031449, 0.0031337
5: 0.0033305, 0.0162060, 0.0035069, 0.0162374, -0.0124245, 0.0122223
6: -0.0121811, -0.0025373, -0.0122079, -0.0025368, -0.0096443, 0.0096706
7: -0.0172352, -0.0098284, -0.0172597, -0.0101618, -0.0070060, 0.0073628
8: -0.0151180, -0.0076571, -0.0151080, -0.0076387, -0.0074792, 0.0074509
9: -0.0051243, 0.0033073, -0.0051578, 0.0033023, -0.0084266, 0.0084651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009469, upper bound: 0.0009468
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009745, upper bound: 0.0009468
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0008401, 0.0099756, -0.0006148, 0.0104695, -0.0107065, 0.0099934
1: -0.0036220, 0.0021305, -0.0035763, 0.0024303, -0.0059050, 0.0055781
2: 0.0070511, 0.0168794, 0.0065562, 0.0169198, -0.0098687, 0.0103232
3: 1.0058727, 1.0071660, 1.0058498, 1.0071174, -0.0012447, 0.0013162
4: -0.0043853, -0.0012508, -0.0043918, -0.0010863, -0.0032990, 0.0031409
5: 0.0033390, 0.0162060, 0.0035121, 0.0168360, -0.0130223, 0.0122228
6: -0.0121811, -0.0025380, -0.0127089, -0.0025387, -0.0096425, 0.0101708
7: -0.0172352, -0.0098346, -0.0174431, -0.0101635, -0.0070048, 0.0075406
8: -0.0151180, -0.0076587, -0.0151858, -0.0073504, -0.0077676, 0.0075271
9: -0.0051243, 0.0033073, -0.0055453, 0.0033407, -0.0084650, 0.0088527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009468, upper bound: 0.0009468
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009468
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008325, 0.0099755, -0.0006523, 0.0099991, -0.0102171, 0.0100297
1: -0.0036210, 0.0021305, -0.0035917, 0.0021417, -0.0056056, 0.0055916
2: 0.0070511, 0.0168794, 0.0070180, 0.0168748, -0.0098237, 0.0098614
3: 1.0058732, 1.0071677, 1.0058836, 1.0071558, -0.0012826, 0.0012841
4: -0.0043853, -0.0012508, -0.0043845, -0.0012404, -0.0031449, 0.0031337
5: 0.0033448, 0.0162060, 0.0034829, 0.0162374, -0.0124083, 0.0122511
6: -0.0121811, -0.0025379, -0.0122079, -0.0025360, -0.0096451, 0.0096700
7: -0.0172352, -0.0098452, -0.0172598, -0.0101350, -0.0070329, 0.0073458
8: -0.0151180, -0.0076594, -0.0151080, -0.0076373, -0.0074807, 0.0074486
9: -0.0051242, 0.0033073, -0.0051578, 0.0033023, -0.0084265, 0.0084652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0008325, 0.0099755, -0.0008592, 0.0099755, -0.0100717, 0.0101063
1: -0.0036210, 0.0021305, -0.0036292, 0.0021306, -0.0055845, 0.0055980
2: 0.0070511, 0.0168794, 0.0070511, 0.0168794, -0.0098283, 0.0098283
3: 1.0058732, 1.0071677, 1.0058714, 1.0071830, -0.0013098, 0.0012963
4: -0.0043853, -0.0012508, -0.0043853, -0.0012508, -0.0031345, 0.0031345
5: 0.0033448, 0.0162060, 0.0033241, 0.0162060, -0.0122807, 0.0123078
6: -0.0121811, -0.0025379, -0.0121811, -0.0025370, -0.0096441, 0.0096432
7: -0.0172352, -0.0098452, -0.0172352, -0.0098209, -0.0073326, 0.0073079
8: -0.0151180, -0.0076594, -0.0151180, -0.0076559, -0.0074621, 0.0074586
9: -0.0051242, 0.0033073, -0.0051244, 0.0033073, -0.0084316, 0.0084317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009747
time: 1.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009701
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0008134, 0.0102705, -0.0006523, 0.0099991, -0.0102147, 0.0103217
1: -0.0036125, 0.0023114, -0.0035917, 0.0021417, -0.0056063, 0.0057718
2: 0.0067613, 0.0169073, 0.0070180, 0.0168748, -0.0101135, 0.0098893
3: 1.0058534, 1.0071440, 1.0058836, 1.0071558, -0.0013024, 0.0012604
4: -0.0043902, -0.0011543, -0.0043845, -0.0012404, -0.0031498, 0.0032303
5: 0.0033597, 0.0165816, 0.0034829, 0.0162374, -0.0124065, 0.0126253
6: -0.0124955, -0.0025395, -0.0122079, -0.0025360, -0.0099595, 0.0096684
7: -0.0173510, -0.0098539, -0.0172598, -0.0101350, -0.0071483, 0.0073387
8: -0.0151701, -0.0074798, -0.0151080, -0.0076373, -0.0075329, 0.0076282
9: -0.0053681, 0.0033333, -0.0051578, 0.0033023, -0.0086703, 0.0084911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 0.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0008134, 0.0102705, -0.0008592, 0.0099755, -0.0100674, 0.0104064
1: -0.0036125, 0.0023114, -0.0036292, 0.0021306, -0.0055827, 0.0057845
2: 0.0067613, 0.0169073, 0.0070511, 0.0168794, -0.0101181, 0.0098562
3: 1.0058534, 1.0071440, 1.0058714, 1.0071830, -0.0013295, 0.0012726
4: -0.0043902, -0.0011543, -0.0043853, -0.0012508, -0.0031394, 0.0032311
5: 0.0033597, 0.0165816, 0.0033241, 0.0162060, -0.0122773, 0.0126876
6: -0.0124955, -0.0025395, -0.0121811, -0.0025370, -0.0099585, 0.0096416
7: -0.0173510, -0.0098539, -0.0172352, -0.0098209, -0.0074485, 0.0073005
8: -0.0151701, -0.0074798, -0.0151180, -0.0076559, -0.0075143, 0.0076381
9: -0.0053681, 0.0033333, -0.0051244, 0.0033073, -0.0086754, 0.0084576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009701
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009467
time: 1.07 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0008203, 0.0080815, -0.0081549, 0.0102628
1: -0.0035917, 0.0021417, -0.0035921, 0.0009910, -0.0044458, 0.0056085
2: 0.0070180, 0.0168748, 0.0089864, 0.0167475, -0.0097295, 0.0078884
3: 1.0058836, 1.0071558, 1.0059075, 1.0071299, -0.0012463, 0.0012482
4: -0.0043845, -0.0012404, -0.0043676, -0.0018928, -0.0024917, 0.0031272
5: 0.0034829, 0.0162374, 0.0033575, 0.0137868, -0.0098461, 0.0124410
6: -0.0122079, -0.0025360, -0.0101534, -0.0025408, -0.0096671, 0.0076173
7: -0.0172598, -0.0101350, -0.0163678, -0.0096178, -0.0075805, 0.0061688
8: -0.0151080, -0.0076373, -0.0149543, -0.0089254, -0.0061826, 0.0073170
9: -0.0051578, 0.0033023, -0.0034489, 0.0032479, -0.0084058, 0.0067511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008823
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008733
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0010320, 0.0081786, -0.0082542, 0.0104603
1: -0.0035917, 0.0021417, -0.0036297, 0.0010522, -0.0045092, 0.0056270
2: 0.0070180, 0.0168748, 0.0089004, 0.0167598, -0.0097418, 0.0079744
3: 1.0058836, 1.0071558, 1.0058941, 1.0071546, -0.0012710, 0.0012617
4: -0.0043845, -0.0012404, -0.0043695, -0.0018634, -0.0025211, 0.0031292
5: 0.0034829, 0.0162374, 0.0031950, 0.0139095, -0.0099703, 0.0125925
6: -0.0122079, -0.0025360, -0.0102558, -0.0025416, -0.0096664, 0.0077198
7: -0.0172598, -0.0101350, -0.0163867, -0.0092872, -0.0079089, 0.0061876
8: -0.0151080, -0.0076373, -0.0149751, -0.0088170, -0.0062910, 0.0073378
9: -0.0051578, 0.0033023, -0.0035149, 0.0032580, -0.0084158, 0.0068172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008823
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008734
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0008512, 0.0099756, -0.0007882, 0.0080815, -0.0083413, 0.0102045
1: -0.0036267, 0.0021305, -0.0035823, 0.0009909, -0.0044579, 0.0055843
2: 0.0070511, 0.0168794, 0.0089866, 0.0167475, -0.0096964, 0.0078928
3: 1.0058719, 1.0071783, 1.0059092, 1.0071132, -0.0012413, 0.0012691
4: -0.0043853, -0.0012508, -0.0043676, -0.0018929, -0.0024925, 0.0031167
5: 0.0033303, 0.0162060, 0.0033824, 0.0137868, -0.0099889, 0.0123818
6: -0.0121811, -0.0025373, -0.0101534, -0.0025417, -0.0096394, 0.0076161
7: -0.0172352, -0.0098281, -0.0163677, -0.0096455, -0.0075293, 0.0064750
8: -0.0151180, -0.0076571, -0.0149543, -0.0089442, -0.0061738, 0.0072972
9: -0.0051243, 0.0033073, -0.0034487, 0.0032479, -0.0083722, 0.0067560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008384
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008384
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0008425, 0.0099755, -0.0007831, 0.0084850, -0.0087490, 0.0102073
1: -0.0036229, 0.0021305, -0.0035771, 0.0012392, -0.0047098, 0.0055849
2: 0.0070511, 0.0168794, 0.0085887, 0.0167879, -0.0097368, 0.0082907
3: 1.0058724, 1.0071681, 1.0058836, 1.0070903, -0.0012180, 0.0012845
4: -0.0043853, -0.0012508, -0.0043745, -0.0017598, -0.0026255, 0.0031237
5: 0.0033371, 0.0162060, 0.0033864, 0.0143007, -0.0105067, 0.0123841
6: -0.0121811, -0.0025379, -0.0105835, -0.0025433, -0.0096378, 0.0080456
7: -0.0172352, -0.0098329, -0.0165253, -0.0096461, -0.0075289, 0.0066285
8: -0.0151180, -0.0076584, -0.0150299, -0.0087157, -0.0064023, 0.0073715
9: -0.0051243, 0.0033073, -0.0037815, 0.0032854, -0.0084097, 0.0070888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008334
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008334
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0008203, 0.0080815, -0.0081549, 0.0102628
1: -0.0035917, 0.0021417, -0.0035921, 0.0009910, -0.0044458, 0.0056085
2: 0.0070180, 0.0168748, 0.0089864, 0.0167475, -0.0097295, 0.0078884
3: 1.0058836, 1.0071558, 1.0059075, 1.0071299, -0.0012463, 0.0012482
4: -0.0043845, -0.0012404, -0.0043676, -0.0018928, -0.0024917, 0.0031272
5: 0.0034829, 0.0162374, 0.0033575, 0.0137868, -0.0098461, 0.0124410
6: -0.0122079, -0.0025360, -0.0101534, -0.0025408, -0.0096671, 0.0076173
7: -0.0172598, -0.0101350, -0.0163678, -0.0096178, -0.0075805, 0.0061688
8: -0.0151080, -0.0076373, -0.0149543, -0.0089254, -0.0061826, 0.0073170
9: -0.0051578, 0.0033023, -0.0034489, 0.0032479, -0.0084058, 0.0067511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008823
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008734
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0010320, 0.0081786, -0.0082542, 0.0104603
1: -0.0035917, 0.0021417, -0.0036297, 0.0010522, -0.0045092, 0.0056270
2: 0.0070180, 0.0168748, 0.0089004, 0.0167598, -0.0097418, 0.0079744
3: 1.0058836, 1.0071558, 1.0058941, 1.0071546, -0.0012710, 0.0012617
4: -0.0043845, -0.0012404, -0.0043695, -0.0018634, -0.0025211, 0.0031292
5: 0.0034829, 0.0162374, 0.0031950, 0.0139095, -0.0099703, 0.0125925
6: -0.0122079, -0.0025360, -0.0102558, -0.0025416, -0.0096664, 0.0077198
7: -0.0172598, -0.0101350, -0.0163867, -0.0092872, -0.0079089, 0.0061876
8: -0.0151080, -0.0076373, -0.0149751, -0.0088170, -0.0062910, 0.0073378
9: -0.0051578, 0.0033023, -0.0035149, 0.0032580, -0.0084158, 0.0068172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008829
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008807
time: 1.04 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0008512, 0.0099756, -0.0010040, 0.0081786, -0.0083152, 0.0102862
1: -0.0036267, 0.0021305, -0.0036211, 0.0010522, -0.0045102, 0.0055900
2: 0.0070511, 0.0168794, 0.0089005, 0.0167598, -0.0097087, 0.0079788
3: 1.0058719, 1.0071783, 1.0058955, 1.0071385, -0.0012666, 0.0012828
4: -0.0043853, -0.0012508, -0.0043695, -0.0018634, -0.0025219, 0.0031187
5: 0.0033303, 0.0162060, 0.0032166, 0.0139095, -0.0100153, 0.0124422
6: -0.0121811, -0.0025373, -0.0102558, -0.0025425, -0.0096386, 0.0077185
7: -0.0172352, -0.0098281, -0.0163866, -0.0093129, -0.0078462, 0.0064801
8: -0.0151180, -0.0076571, -0.0149751, -0.0088398, -0.0062782, 0.0073180
9: -0.0051243, 0.0033073, -0.0035147, 0.0032580, -0.0083823, 0.0068220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008423
time: 1.07 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008423
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0008425, 0.0099755, -0.0009874, 0.0084447, -0.0085918, 0.0102841
1: -0.0036229, 0.0021305, -0.0036136, 0.0012150, -0.0046798, 0.0055884
2: 0.0070511, 0.0168794, 0.0086357, 0.0167854, -0.0097343, 0.0082437
3: 1.0058724, 1.0071681, 1.0058812, 1.0071148, -0.0012424, 0.0012869
4: -0.0043853, -0.0012508, -0.0043740, -0.0017749, -0.0026105, 0.0031232
5: 0.0033371, 0.0162060, 0.0032297, 0.0142486, -0.0103632, 0.0124405
6: -0.0121811, -0.0025379, -0.0105399, -0.0025439, -0.0096372, 0.0080019
7: -0.0172352, -0.0098329, -0.0164950, -0.0093188, -0.0078406, 0.0065844
8: -0.0151180, -0.0076584, -0.0150245, -0.0087088, -0.0064092, 0.0073661
9: -0.0051243, 0.0033073, -0.0037369, 0.0032809, -0.0084052, 0.0070443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008401
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008401
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0007882, 0.0080815, -0.0081548, 0.0102262
1: -0.0035917, 0.0021417, -0.0035823, 0.0009909, -0.0044458, 0.0055938
2: 0.0070180, 0.0168748, 0.0089866, 0.0167475, -0.0097295, 0.0078882
3: 1.0058836, 1.0071558, 1.0059092, 1.0071132, -0.0012296, 0.0012466
4: -0.0043845, -0.0012404, -0.0043676, -0.0018929, -0.0024917, 0.0031272
5: 0.0034829, 0.0162374, 0.0033824, 0.0137868, -0.0098461, 0.0124124
6: -0.0122079, -0.0025360, -0.0101534, -0.0025417, -0.0096662, 0.0076173
7: -0.0172598, -0.0101350, -0.0163677, -0.0096455, -0.0075528, 0.0061687
8: -0.0151080, -0.0076373, -0.0149543, -0.0089442, -0.0061638, 0.0073170
9: -0.0051578, 0.0033023, -0.0034487, 0.0032479, -0.0084058, 0.0067509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008413
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008384
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0010040, 0.0081786, -0.0082541, 0.0104275
1: -0.0035917, 0.0021417, -0.0036211, 0.0010522, -0.0045092, 0.0056144
2: 0.0070180, 0.0168748, 0.0089005, 0.0167598, -0.0097418, 0.0079742
3: 1.0058836, 1.0071558, 1.0058955, 1.0071385, -0.0012549, 0.0012603
4: -0.0043845, -0.0012404, -0.0043695, -0.0018634, -0.0025211, 0.0031292
5: 0.0034829, 0.0162374, 0.0032166, 0.0139095, -0.0099703, 0.0125670
6: -0.0122079, -0.0025360, -0.0102558, -0.0025425, -0.0096654, 0.0077198
7: -0.0172598, -0.0101350, -0.0163866, -0.0093129, -0.0078832, 0.0061875
8: -0.0151080, -0.0076373, -0.0149751, -0.0088398, -0.0062682, 0.0073378
9: -0.0051578, 0.0033023, -0.0035147, 0.0032580, -0.0084158, 0.0068170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008428
time: 0.90 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008423
time: 1.12 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0008512, 0.0099756, -0.0007888, 0.0099003, -0.0101559, 0.0102121
1: -0.0036267, 0.0021305, -0.0035823, 0.0020921, -0.0055687, 0.0055972
2: 0.0070511, 0.0168794, 0.0071510, 0.0168926, -0.0098415, 0.0097284
3: 1.0058719, 1.0071783, 1.0057830, 1.0071132, -0.0012413, 0.0013953
4: -0.0043853, -0.0012508, -0.0043900, -0.0012828, -0.0031026, 0.0031391
5: 0.0033303, 0.0162060, 0.0033820, 0.0161074, -0.0123071, 0.0123879
6: -0.0121811, -0.0025373, -0.0120976, -0.0025387, -0.0096424, 0.0095604
7: -0.0172352, -0.0098281, -0.0171659, -0.0096454, -0.0075301, 0.0072727
8: -0.0151180, -0.0076571, -0.0151875, -0.0077523, -0.0073657, 0.0075305
9: -0.0051243, 0.0033073, -0.0050260, 0.0033586, -0.0084829, 0.0083333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008423
time: 1.06 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008968, upper bound: 0.0008424
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0008425, 0.0099755, -0.0007837, 0.0102666, -0.0105271, 0.0102107
1: -0.0036229, 0.0021305, -0.0035771, 0.0023161, -0.0057973, 0.0055958
2: 0.0070511, 0.0168794, 0.0067873, 0.0169267, -0.0098756, 0.0100921
3: 1.0058724, 1.0071681, 1.0057590, 1.0070903, -0.0012180, 0.0014091
4: -0.0043853, -0.0012508, -0.0043960, -0.0011615, -0.0032239, 0.0031451
5: 0.0033371, 0.0162060, 0.0033860, 0.0165740, -0.0127778, 0.0123869
6: -0.0121811, -0.0025379, -0.0124883, -0.0025403, -0.0096408, 0.0099504
7: -0.0172352, -0.0098329, -0.0173095, -0.0096460, -0.0075293, 0.0074118
8: -0.0151180, -0.0076584, -0.0152551, -0.0075303, -0.0075876, 0.0075968
9: -0.0051243, 0.0033073, -0.0053291, 0.0033921, -0.0085163, 0.0086364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008401
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008845, upper bound: 0.0008401
time: 0.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0007831, 0.0084850, -0.0085672, 0.0102290
1: -0.0035917, 0.0021417, -0.0035771, 0.0012392, -0.0046993, 0.0055944
2: 0.0070180, 0.0168748, 0.0085887, 0.0167879, -0.0097698, 0.0082861
3: 1.0058836, 1.0071558, 1.0058836, 1.0070903, -0.0012068, 0.0012722
4: -0.0043845, -0.0012404, -0.0043745, -0.0017598, -0.0026247, 0.0031341
5: 0.0034829, 0.0162374, 0.0033864, 0.0143007, -0.0103675, 0.0124147
6: -0.0122079, -0.0025360, -0.0105835, -0.0025433, -0.0096646, 0.0080475
7: -0.0172598, -0.0101350, -0.0165253, -0.0096461, -0.0075525, 0.0063267
8: -0.0151080, -0.0076373, -0.0150299, -0.0087157, -0.0063923, 0.0073926
9: -0.0051578, 0.0033023, -0.0037815, 0.0032854, -0.0084432, 0.0070837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008391
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008334
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0009874, 0.0084447, -0.0085263, 0.0104285
1: -0.0035917, 0.0021417, -0.0036136, 0.0012150, -0.0046755, 0.0056158
2: 0.0070180, 0.0168748, 0.0086357, 0.0167854, -0.0097674, 0.0082391
3: 1.0058836, 1.0071558, 1.0058812, 1.0071148, -0.0012312, 0.0012746
4: -0.0043845, -0.0012404, -0.0043740, -0.0017749, -0.0026097, 0.0031336
5: 0.0034829, 0.0162374, 0.0032297, 0.0142486, -0.0103159, 0.0125678
6: -0.0122079, -0.0025360, -0.0105399, -0.0025439, -0.0096640, 0.0080038
7: -0.0172598, -0.0101350, -0.0164950, -0.0093188, -0.0078780, 0.0062963
8: -0.0151080, -0.0076373, -0.0150245, -0.0087088, -0.0063992, 0.0073872
9: -0.0051578, 0.0033023, -0.0037369, 0.0032809, -0.0084388, 0.0070392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008409
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008401
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0008512, 0.0099756, -0.0007888, 0.0099003, -0.0101559, 0.0102121
1: -0.0036267, 0.0021305, -0.0035823, 0.0020921, -0.0055687, 0.0055972
2: 0.0070511, 0.0168794, 0.0071510, 0.0168926, -0.0098415, 0.0097284
3: 1.0058719, 1.0071783, 1.0057830, 1.0071132, -0.0012413, 0.0013953
4: -0.0043853, -0.0012508, -0.0043900, -0.0012828, -0.0031026, 0.0031391
5: 0.0033303, 0.0162060, 0.0033820, 0.0161074, -0.0123071, 0.0123879
6: -0.0121811, -0.0025373, -0.0120976, -0.0025387, -0.0096424, 0.0095604
7: -0.0172352, -0.0098281, -0.0171659, -0.0096454, -0.0075301, 0.0072727
8: -0.0151180, -0.0076571, -0.0151875, -0.0077523, -0.0073657, 0.0075305
9: -0.0051243, 0.0033073, -0.0050260, 0.0033586, -0.0084829, 0.0083333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008423
time: 1.04 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008968, upper bound: 0.0008424
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0008425, 0.0099755, -0.0007837, 0.0102666, -0.0105271, 0.0102107
1: -0.0036229, 0.0021305, -0.0035771, 0.0023161, -0.0057973, 0.0055958
2: 0.0070511, 0.0168794, 0.0067873, 0.0169267, -0.0098756, 0.0100921
3: 1.0058724, 1.0071681, 1.0057590, 1.0070903, -0.0012180, 0.0014091
4: -0.0043853, -0.0012508, -0.0043960, -0.0011615, -0.0032239, 0.0031451
5: 0.0033371, 0.0162060, 0.0033860, 0.0165740, -0.0127778, 0.0123869
6: -0.0121811, -0.0025379, -0.0124883, -0.0025403, -0.0096408, 0.0099504
7: -0.0172352, -0.0098329, -0.0173095, -0.0096460, -0.0075293, 0.0074118
8: -0.0151180, -0.0076584, -0.0152551, -0.0075303, -0.0075876, 0.0075968
9: -0.0051243, 0.0033073, -0.0053291, 0.0033921, -0.0085163, 0.0086364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 64
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 237

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008401
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008845, upper bound: 0.0008401
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0006523, 0.0099991, -0.0102628, 0.0081549
1: -0.0035921, 0.0009910, -0.0035917, 0.0021417, -0.0056085, 0.0044458
2: 0.0089864, 0.0167475, 0.0070180, 0.0168748, -0.0078884, 0.0097295
3: 1.0059075, 1.0071299, 1.0058836, 1.0071558, -0.0012482, 0.0012463
4: -0.0043676, -0.0018928, -0.0043845, -0.0012404, -0.0031272, 0.0024917
5: 0.0033575, 0.0137868, 0.0034829, 0.0162374, -0.0124410, 0.0098461
6: -0.0101534, -0.0025408, -0.0122079, -0.0025360, -0.0076173, 0.0096671
7: -0.0163678, -0.0096178, -0.0172598, -0.0101350, -0.0061688, 0.0075805
8: -0.0149543, -0.0089254, -0.0151080, -0.0076373, -0.0073170, 0.0061826
9: -0.0034489, 0.0032479, -0.0051578, 0.0033023, -0.0067511, 0.0084058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 70

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008735, upper bound: 0.0009107
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0008592, 0.0099755, -0.0102411, 0.0083505
1: -0.0035921, 0.0009910, -0.0036292, 0.0021306, -0.0055990, 0.0044615
2: 0.0089864, 0.0167475, 0.0070511, 0.0168794, -0.0078930, 0.0096964
3: 1.0059075, 1.0071299, 1.0058714, 1.0071830, -0.0012754, 0.0012585
4: -0.0043676, -0.0018928, -0.0043853, -0.0012508, -0.0031167, 0.0024925
5: 0.0033575, 0.0137868, 0.0033241, 0.0162060, -0.0124104, 0.0099961
6: -0.0101534, -0.0025408, -0.0121811, -0.0025370, -0.0076164, 0.0096403
7: -0.0163678, -0.0096178, -0.0172352, -0.0098209, -0.0064824, 0.0075570
8: -0.0149543, -0.0089254, -0.0151180, -0.0076559, -0.0072984, 0.0061925
9: -0.0034489, 0.0032479, -0.0051244, 0.0033073, -0.0067562, 0.0083723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 70

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008385, upper bound: 0.0008876
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008781
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010320, 0.0081786, -0.0006523, 0.0099991, -0.0104603, 0.0082542
1: -0.0036297, 0.0010522, -0.0035917, 0.0021417, -0.0056270, 0.0045092
2: 0.0089004, 0.0167598, 0.0070180, 0.0168748, -0.0079744, 0.0097418
3: 1.0058941, 1.0071546, 1.0058836, 1.0071558, -0.0012617, 0.0012710
4: -0.0043695, -0.0018634, -0.0043845, -0.0012404, -0.0031292, 0.0025211
5: 0.0031950, 0.0139095, 0.0034829, 0.0162374, -0.0125925, 0.0099703
6: -0.0102558, -0.0025416, -0.0122079, -0.0025360, -0.0077198, 0.0096664
7: -0.0163867, -0.0092872, -0.0172598, -0.0101350, -0.0061876, 0.0079089
8: -0.0149751, -0.0088170, -0.0151080, -0.0076373, -0.0073378, 0.0062910
9: -0.0035149, 0.0032580, -0.0051578, 0.0033023, -0.0068172, 0.0084158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010320, 0.0081786, -0.0008592, 0.0099755, -0.0103213, 0.0083256
1: -0.0036297, 0.0010522, -0.0036292, 0.0021306, -0.0056046, 0.0045144
2: 0.0089004, 0.0167598, 0.0070511, 0.0168794, -0.0079789, 0.0097088
3: 1.0058941, 1.0071546, 1.0058714, 1.0071830, -0.0012889, 0.0012832
4: -0.0043695, -0.0018634, -0.0043853, -0.0012508, -0.0031187, 0.0025219
5: 0.0031950, 0.0139095, 0.0033241, 0.0162060, -0.0124695, 0.0100235
6: -0.0102558, -0.0025416, -0.0121811, -0.0025370, -0.0077188, 0.0096396
7: -0.0163867, -0.0092872, -0.0172352, -0.0098209, -0.0064876, 0.0078719
8: -0.0149751, -0.0088170, -0.0151180, -0.0076559, -0.0073192, 0.0063010
9: -0.0035149, 0.0032580, -0.0051244, 0.0033073, -0.0068223, 0.0083823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008810
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008753
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0007882, 0.0080815, -0.0006523, 0.0099991, -0.0102262, 0.0081548
1: -0.0035823, 0.0009909, -0.0035917, 0.0021417, -0.0055938, 0.0044458
2: 0.0089866, 0.0167475, 0.0070180, 0.0168748, -0.0078882, 0.0097295
3: 1.0059092, 1.0071132, 1.0058836, 1.0071558, -0.0012466, 0.0012296
4: -0.0043676, -0.0018929, -0.0043845, -0.0012404, -0.0031272, 0.0024917
5: 0.0033824, 0.0137868, 0.0034829, 0.0162374, -0.0124124, 0.0098461
6: -0.0101534, -0.0025417, -0.0122079, -0.0025360, -0.0076173, 0.0096662
7: -0.0163677, -0.0096455, -0.0172598, -0.0101350, -0.0061687, 0.0075528
8: -0.0149543, -0.0089442, -0.0151080, -0.0076373, -0.0073170, 0.0061638
9: -0.0034487, 0.0032479, -0.0051578, 0.0033023, -0.0067509, 0.0084058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008810
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008810
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0007882, 0.0080815, -0.0008592, 0.0099755, -0.0102045, 0.0083505
1: -0.0035823, 0.0009909, -0.0036292, 0.0021306, -0.0055843, 0.0044615
2: 0.0089866, 0.0167475, 0.0070511, 0.0168794, -0.0078928, 0.0096964
3: 1.0059092, 1.0071132, 1.0058714, 1.0071830, -0.0012738, 0.0012418
4: -0.0043676, -0.0018929, -0.0043853, -0.0012508, -0.0031167, 0.0024925
5: 0.0033824, 0.0137868, 0.0033241, 0.0162060, -0.0123818, 0.0099961
6: -0.0101534, -0.0025417, -0.0121811, -0.0025370, -0.0076164, 0.0096394
7: -0.0163677, -0.0096455, -0.0172352, -0.0098209, -0.0064824, 0.0075293
8: -0.0149543, -0.0089442, -0.0151180, -0.0076559, -0.0072984, 0.0061738
9: -0.0034487, 0.0032479, -0.0051244, 0.0033073, -0.0067560, 0.0083723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008969
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008846
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0007831, 0.0084850, -0.0006523, 0.0099991, -0.0102290, 0.0085672
1: -0.0035771, 0.0012392, -0.0035917, 0.0021417, -0.0055944, 0.0046993
2: 0.0085887, 0.0167879, 0.0070180, 0.0168748, -0.0082861, 0.0097698
3: 1.0058836, 1.0070903, 1.0058836, 1.0071558, -0.0012722, 0.0012068
4: -0.0043745, -0.0017598, -0.0043845, -0.0012404, -0.0031341, 0.0026247
5: 0.0033864, 0.0143007, 0.0034829, 0.0162374, -0.0124147, 0.0103675
6: -0.0105835, -0.0025433, -0.0122079, -0.0025360, -0.0080475, 0.0096646
7: -0.0165253, -0.0096461, -0.0172598, -0.0101350, -0.0063267, 0.0075525
8: -0.0150299, -0.0087157, -0.0151080, -0.0076373, -0.0073926, 0.0063923
9: -0.0037815, 0.0032854, -0.0051578, 0.0033023, -0.0070837, 0.0084432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008753
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008753
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0007831, 0.0084850, -0.0008592, 0.0099755, -0.0102073, 0.0087629
1: -0.0035771, 0.0012392, -0.0036292, 0.0021306, -0.0055850, 0.0047150
2: 0.0085887, 0.0167879, 0.0070511, 0.0168794, -0.0082907, 0.0097368
3: 1.0058836, 1.0070903, 1.0058714, 1.0071830, -0.0012994, 0.0012189
4: -0.0043745, -0.0017598, -0.0043853, -0.0012508, -0.0031237, 0.0026255
5: 0.0033864, 0.0143007, 0.0033241, 0.0162060, -0.0123841, 0.0105176
6: -0.0105835, -0.0025433, -0.0121811, -0.0025370, -0.0080465, 0.0096378
7: -0.0165253, -0.0096461, -0.0172352, -0.0098209, -0.0066403, 0.0075289
8: -0.0150299, -0.0087157, -0.0151180, -0.0076559, -0.0073740, 0.0064023
9: -0.0037815, 0.0032854, -0.0051244, 0.0033073, -0.0070888, 0.0084097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008846
time: 1.02 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008846
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0006523, 0.0099991, -0.0102628, 0.0081549
1: -0.0035921, 0.0009910, -0.0035917, 0.0021417, -0.0056085, 0.0044458
2: 0.0089864, 0.0167475, 0.0070180, 0.0168748, -0.0078884, 0.0097295
3: 1.0059075, 1.0071299, 1.0058836, 1.0071558, -0.0012482, 0.0012463
4: -0.0043676, -0.0018928, -0.0043845, -0.0012404, -0.0031272, 0.0024917
5: 0.0033575, 0.0137868, 0.0034829, 0.0162374, -0.0124410, 0.0098461
6: -0.0101534, -0.0025408, -0.0122079, -0.0025360, -0.0076173, 0.0096671
7: -0.0163678, -0.0096178, -0.0172598, -0.0101350, -0.0061688, 0.0075805
8: -0.0149543, -0.0089254, -0.0151080, -0.0076373, -0.0073170, 0.0061826
9: -0.0034489, 0.0032479, -0.0051578, 0.0033023, -0.0067511, 0.0084058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 70

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008735, upper bound: 0.0009107
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0008592, 0.0099755, -0.0102411, 0.0083505
1: -0.0035921, 0.0009910, -0.0036292, 0.0021306, -0.0055990, 0.0044615
2: 0.0089864, 0.0167475, 0.0070511, 0.0168794, -0.0078930, 0.0096964
3: 1.0059075, 1.0071299, 1.0058714, 1.0071830, -0.0012754, 0.0012585
4: -0.0043676, -0.0018928, -0.0043853, -0.0012508, -0.0031167, 0.0024925
5: 0.0033575, 0.0137868, 0.0033241, 0.0162060, -0.0124104, 0.0099961
6: -0.0101534, -0.0025408, -0.0121811, -0.0025370, -0.0076164, 0.0096403
7: -0.0163678, -0.0096178, -0.0172352, -0.0098209, -0.0064824, 0.0075570
8: -0.0149543, -0.0089254, -0.0151180, -0.0076559, -0.0072984, 0.0061925
9: -0.0034489, 0.0032479, -0.0051244, 0.0033073, -0.0067562, 0.0083723

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 70

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008385, upper bound: 0.0008876
time: 1.00 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008781
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010320, 0.0081786, -0.0006523, 0.0099991, -0.0104603, 0.0082542
1: -0.0036297, 0.0010522, -0.0035917, 0.0021417, -0.0056270, 0.0045092
2: 0.0089004, 0.0167598, 0.0070180, 0.0168748, -0.0079744, 0.0097418
3: 1.0058941, 1.0071546, 1.0058836, 1.0071558, -0.0012617, 0.0012710
4: -0.0043695, -0.0018634, -0.0043845, -0.0012404, -0.0031292, 0.0025211
5: 0.0031950, 0.0139095, 0.0034829, 0.0162374, -0.0125925, 0.0099703
6: -0.0102558, -0.0025416, -0.0122079, -0.0025360, -0.0077198, 0.0096664
7: -0.0163867, -0.0092872, -0.0172598, -0.0101350, -0.0061876, 0.0079089
8: -0.0149751, -0.0088170, -0.0151080, -0.0076373, -0.0073378, 0.0062910
9: -0.0035149, 0.0032580, -0.0051578, 0.0033023, -0.0068172, 0.0084158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008807, upper bound: 0.0009053
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010320, 0.0081786, -0.0008592, 0.0099755, -0.0103213, 0.0083256
1: -0.0036297, 0.0010522, -0.0036292, 0.0021306, -0.0056046, 0.0045144
2: 0.0089004, 0.0167598, 0.0070511, 0.0168794, -0.0079789, 0.0097088
3: 1.0058941, 1.0071546, 1.0058714, 1.0071830, -0.0012889, 0.0012832
4: -0.0043695, -0.0018634, -0.0043853, -0.0012508, -0.0031187, 0.0025219
5: 0.0031950, 0.0139095, 0.0033241, 0.0162060, -0.0124695, 0.0100235
6: -0.0102558, -0.0025416, -0.0121811, -0.0025370, -0.0077188, 0.0096396
7: -0.0163867, -0.0092872, -0.0172352, -0.0098209, -0.0064876, 0.0078719
8: -0.0149751, -0.0088170, -0.0151180, -0.0076559, -0.0073192, 0.0063010
9: -0.0035149, 0.0032580, -0.0051244, 0.0033073, -0.0068223, 0.0083823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008810
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008753
time: 1.24 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0010040, 0.0081786, -0.0006523, 0.0099991, -0.0104275, 0.0082541
1: -0.0036211, 0.0010522, -0.0035917, 0.0021417, -0.0056144, 0.0045092
2: 0.0089005, 0.0167598, 0.0070180, 0.0168748, -0.0079742, 0.0097418
3: 1.0058955, 1.0071385, 1.0058836, 1.0071558, -0.0012603, 0.0012549
4: -0.0043695, -0.0018634, -0.0043845, -0.0012404, -0.0031292, 0.0025211
5: 0.0032166, 0.0139095, 0.0034829, 0.0162374, -0.0125670, 0.0099703
6: -0.0102558, -0.0025425, -0.0122079, -0.0025360, -0.0077198, 0.0096654
7: -0.0163866, -0.0093129, -0.0172598, -0.0101350, -0.0061875, 0.0078832
8: -0.0149751, -0.0088398, -0.0151080, -0.0076373, -0.0073378, 0.0062682
9: -0.0035147, 0.0032580, -0.0051578, 0.0033023, -0.0068170, 0.0084158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008810
time: 1.04 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008810
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0010040, 0.0081786, -0.0008592, 0.0099755, -0.0102862, 0.0083256
1: -0.0036211, 0.0010522, -0.0036292, 0.0021306, -0.0055900, 0.0045143
2: 0.0089005, 0.0167598, 0.0070511, 0.0168794, -0.0079788, 0.0097088
3: 1.0058955, 1.0071385, 1.0058714, 1.0071830, -0.0012875, 0.0012671
4: -0.0043695, -0.0018634, -0.0043853, -0.0012508, -0.0031187, 0.0025219
5: 0.0032166, 0.0139095, 0.0033241, 0.0162060, -0.0124422, 0.0100235
6: -0.0102558, -0.0025425, -0.0121811, -0.0025370, -0.0077188, 0.0096386
7: -0.0163866, -0.0093129, -0.0172352, -0.0098209, -0.0064875, 0.0078462
8: -0.0149751, -0.0088398, -0.0151180, -0.0076559, -0.0073192, 0.0062782
9: -0.0035147, 0.0032580, -0.0051244, 0.0033073, -0.0068220, 0.0083823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008423, upper bound: 0.0008968
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008401, upper bound: 0.0008846
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0009874, 0.0084447, -0.0006523, 0.0099991, -0.0104285, 0.0085263
1: -0.0036136, 0.0012150, -0.0035917, 0.0021417, -0.0056158, 0.0046755
2: 0.0086357, 0.0167854, 0.0070180, 0.0168748, -0.0082391, 0.0097674
3: 1.0058812, 1.0071148, 1.0058836, 1.0071558, -0.0012746, 0.0012312
4: -0.0043740, -0.0017749, -0.0043845, -0.0012404, -0.0031336, 0.0026097
5: 0.0032297, 0.0142486, 0.0034829, 0.0162374, -0.0125678, 0.0103159
6: -0.0105399, -0.0025439, -0.0122079, -0.0025360, -0.0080038, 0.0096640
7: -0.0164950, -0.0093188, -0.0172598, -0.0101350, -0.0062963, 0.0078779
8: -0.0150245, -0.0087088, -0.0151080, -0.0076373, -0.0073872, 0.0063992
9: -0.0037369, 0.0032809, -0.0051578, 0.0033023, -0.0070392, 0.0084388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008753
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008753
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0009874, 0.0084447, -0.0008592, 0.0099755, -0.0102841, 0.0086032
1: -0.0036136, 0.0012150, -0.0036292, 0.0021306, -0.0055884, 0.0046841
2: 0.0086357, 0.0167854, 0.0070511, 0.0168794, -0.0082437, 0.0097343
3: 1.0058812, 1.0071148, 1.0058714, 1.0071830, -0.0013018, 0.0012434
4: -0.0043740, -0.0017749, -0.0043853, -0.0012508, -0.0031232, 0.0026105
5: 0.0032297, 0.0142486, 0.0033241, 0.0162060, -0.0124406, 0.0103720
6: -0.0105399, -0.0025439, -0.0121811, -0.0025370, -0.0080028, 0.0096372
7: -0.0164950, -0.0093188, -0.0172352, -0.0098209, -0.0065962, 0.0078406
8: -0.0150245, -0.0087088, -0.0151180, -0.0076559, -0.0073686, 0.0064092
9: -0.0037369, 0.0032809, -0.0051244, 0.0033073, -0.0070443, 0.0084053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008401, upper bound: 0.0008846
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008401, upper bound: 0.0008846
time: 1.20 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0008203, 0.0080815, -0.0083442, 0.0083442
1: -0.0035921, 0.0009910, -0.0035921, 0.0009910, -0.0044504, 0.0044504
2: 0.0089864, 0.0167475, 0.0089864, 0.0167475, -0.0077611, 0.0077611
3: 1.0059075, 1.0071299, 1.0059075, 1.0071299, -0.0012224, 0.0012224
4: -0.0043676, -0.0018928, -0.0043676, -0.0018928, -0.0024747, 0.0024747
5: 0.0033575, 0.0137868, 0.0033575, 0.0137868, -0.0099889, 0.0099889
6: -0.0101534, -0.0025408, -0.0101534, -0.0025408, -0.0076126, 0.0076126
7: -0.0163678, -0.0096178, -0.0163678, -0.0096178, -0.0066899, 0.0066899
8: -0.0149543, -0.0089254, -0.0149543, -0.0089254, -0.0060289, 0.0060289
9: -0.0034489, 0.0032479, -0.0034489, 0.0032479, -0.0066968, 0.0066968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 70

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008693, upper bound: 0.0008783
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008692, upper bound: 0.0008692
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0010320, 0.0081786, -0.0084473, 0.0085414
1: -0.0035921, 0.0009910, -0.0036297, 0.0010522, -0.0045148, 0.0044678
2: 0.0089864, 0.0167475, 0.0089004, 0.0167598, -0.0077734, 0.0078471
3: 1.0059075, 1.0071299, 1.0058941, 1.0071546, -0.0012470, 0.0012358
4: -0.0043676, -0.0018928, -0.0043695, -0.0018634, -0.0025042, 0.0024767
5: 0.0033575, 0.0137868, 0.0031950, 0.0139095, -0.0101156, 0.0101402
6: -0.0101534, -0.0025408, -0.0102558, -0.0025416, -0.0076118, 0.0077150
7: -0.0163678, -0.0096178, -0.0163867, -0.0092872, -0.0070183, 0.0067089
8: -0.0149543, -0.0089254, -0.0149751, -0.0088170, -0.0061373, 0.0060496
9: -0.0034489, 0.0032479, -0.0035149, 0.0032580, -0.0067068, 0.0067629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 70

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008307, upper bound: 0.0008452
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008278, upper bound: 0.0008330
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010229, 0.0081786, -0.0007882, 0.0080815, -0.0085307, 0.0084100
1: -0.0036268, 0.0010522, -0.0035823, 0.0009909, -0.0044637, 0.0045003
2: 0.0089005, 0.0167598, 0.0089866, 0.0167475, -0.0078470, 0.0077733
3: 1.0058944, 1.0071493, 1.0059092, 1.0071132, -0.0012188, 0.0012401
4: -0.0043695, -0.0018634, -0.0043676, -0.0018929, -0.0024767, 0.0025042
5: 0.0032020, 0.0139095, 0.0033824, 0.0137868, -0.0101320, 0.0100865
6: -0.0102558, -0.0025419, -0.0101534, -0.0025417, -0.0077141, 0.0076115
7: -0.0163866, -0.0092956, -0.0163677, -0.0096455, -0.0066812, 0.0070100
8: -0.0149751, -0.0088245, -0.0149543, -0.0089442, -0.0060309, 0.0061298
9: -0.0035149, 0.0032580, -0.0034487, 0.0032479, -0.0067628, 0.0067066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008305
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008305
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010133, 0.0081786, -0.0007831, 0.0084850, -0.0089404, 0.0084106
1: -0.0036228, 0.0010522, -0.0035771, 0.0012392, -0.0047157, 0.0045001
2: 0.0089005, 0.0167598, 0.0085887, 0.0167879, -0.0078874, 0.0081712
3: 1.0058949, 1.0071378, 1.0058836, 1.0070903, -0.0011954, 0.0012542
4: -0.0043695, -0.0018634, -0.0043745, -0.0017598, -0.0026097, 0.0025111
5: 0.0032095, 0.0139095, 0.0033864, 0.0143007, -0.0106509, 0.0100870
6: -0.0102558, -0.0025426, -0.0105835, -0.0025433, -0.0077125, 0.0080410
7: -0.0163866, -0.0093005, -0.0165253, -0.0096461, -0.0066808, 0.0071629
8: -0.0149751, -0.0088389, -0.0150299, -0.0087157, -0.0062594, 0.0061909
9: -0.0035148, 0.0032580, -0.0037815, 0.0032854, -0.0068002, 0.0070394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008278
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008278
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0007882, 0.0080815, -0.0008203, 0.0080815, -0.0083070, 0.0083442
1: -0.0035823, 0.0009909, -0.0035921, 0.0009910, -0.0044359, 0.0044504
2: 0.0089866, 0.0167475, 0.0089864, 0.0167475, -0.0077609, 0.0077611
3: 1.0059092, 1.0071132, 1.0059075, 1.0071299, -0.0012207, 0.0012057
4: -0.0043676, -0.0018929, -0.0043676, -0.0018928, -0.0024747, 0.0024747
5: 0.0033824, 0.0137868, 0.0033575, 0.0137868, -0.0099598, 0.0099889
6: -0.0101534, -0.0025417, -0.0101534, -0.0025408, -0.0076126, 0.0076117
7: -0.0163677, -0.0096455, -0.0163678, -0.0096178, -0.0066898, 0.0066622
8: -0.0149543, -0.0089442, -0.0149543, -0.0089254, -0.0060289, 0.0060101
9: -0.0034487, 0.0032479, -0.0034489, 0.0032479, -0.0066966, 0.0066968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008304, upper bound: 0.0008357
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008304, upper bound: 0.0008357
time: 0.99 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0007882, 0.0080815, -0.0010320, 0.0081786, -0.0084100, 0.0085414
1: -0.0035823, 0.0009909, -0.0036297, 0.0010522, -0.0045003, 0.0044678
2: 0.0089866, 0.0167475, 0.0089004, 0.0167598, -0.0077733, 0.0078471
3: 1.0059092, 1.0071132, 1.0058941, 1.0071546, -0.0012454, 0.0012192
4: -0.0043676, -0.0018929, -0.0043695, -0.0018634, -0.0025042, 0.0024767
5: 0.0033824, 0.0137868, 0.0031950, 0.0139095, -0.0100865, 0.0101402
6: -0.0101534, -0.0025417, -0.0102558, -0.0025416, -0.0076118, 0.0077141
7: -0.0163677, -0.0096455, -0.0163867, -0.0092872, -0.0070183, 0.0066813
8: -0.0149543, -0.0089442, -0.0149751, -0.0088170, -0.0061373, 0.0060309
9: -0.0034487, 0.0032479, -0.0035149, 0.0032580, -0.0067066, 0.0067629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008304, upper bound: 0.0008454
time: 1.05 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0007831, 0.0084850, -0.0008203, 0.0080815, -0.0083076, 0.0087583
1: -0.0035771, 0.0012392, -0.0035921, 0.0009910, -0.0044357, 0.0047041
2: 0.0085887, 0.0167879, 0.0089864, 0.0167475, -0.0081588, 0.0078015
3: 1.0058836, 1.0070903, 1.0059075, 1.0071299, -0.0012463, 0.0011828
4: -0.0043745, -0.0017598, -0.0043676, -0.0018928, -0.0024817, 0.0026077
5: 0.0033864, 0.0143007, 0.0033575, 0.0137868, -0.0099603, 0.0105113
6: -0.0105835, -0.0025433, -0.0101534, -0.0025408, -0.0080428, 0.0076101
7: -0.0165253, -0.0096461, -0.0163678, -0.0096178, -0.0068478, 0.0066618
8: -0.0150299, -0.0087157, -0.0149543, -0.0089254, -0.0061045, 0.0062386
9: -0.0037815, 0.0032854, -0.0034489, 0.0032479, -0.0070294, 0.0067342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008277
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008278
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0007831, 0.0084850, -0.0010320, 0.0081786, -0.0084106, 0.0089555
1: -0.0035771, 0.0012392, -0.0036297, 0.0010522, -0.0045001, 0.0047215
2: 0.0085887, 0.0167879, 0.0089004, 0.0167598, -0.0081712, 0.0078874
3: 1.0058836, 1.0070903, 1.0058941, 1.0071546, -0.0012710, 0.0011963
4: -0.0043745, -0.0017598, -0.0043695, -0.0018634, -0.0025111, 0.0026097
5: 0.0033864, 0.0143007, 0.0031950, 0.0139095, -0.0100870, 0.0106627
6: -0.0105835, -0.0025433, -0.0102558, -0.0025416, -0.0080420, 0.0077125
7: -0.0165253, -0.0096461, -0.0163867, -0.0092872, -0.0071762, 0.0066808
8: -0.0150299, -0.0087157, -0.0149751, -0.0088170, -0.0062129, 0.0062594
9: -0.0037815, 0.0032854, -0.0035149, 0.0032580, -0.0070394, 0.0068003

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0007882, 0.0080815, -0.0083442, 0.0083070
1: -0.0035921, 0.0009910, -0.0035823, 0.0009909, -0.0044504, 0.0044359
2: 0.0089864, 0.0167475, 0.0089866, 0.0167475, -0.0077611, 0.0077609
3: 1.0059075, 1.0071299, 1.0059092, 1.0071132, -0.0012057, 0.0012207
4: -0.0043676, -0.0018928, -0.0043676, -0.0018929, -0.0024747, 0.0024747
5: 0.0033575, 0.0137868, 0.0033824, 0.0137868, -0.0099889, 0.0099598
6: -0.0101534, -0.0025408, -0.0101534, -0.0025417, -0.0076117, 0.0076126
7: -0.0163678, -0.0096178, -0.0163677, -0.0096455, -0.0066622, 0.0066898
8: -0.0149543, -0.0089254, -0.0149543, -0.0089442, -0.0060101, 0.0060289
9: -0.0034489, 0.0032479, -0.0034487, 0.0032479, -0.0066968, 0.0066966

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 70

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008339
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008305
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0010040, 0.0081786, -0.0084472, 0.0085088
1: -0.0035921, 0.0009910, -0.0036211, 0.0010522, -0.0045148, 0.0044556
2: 0.0089864, 0.0167475, 0.0089005, 0.0167598, -0.0077734, 0.0078469
3: 1.0059075, 1.0071299, 1.0058955, 1.0071385, -0.0012310, 0.0012344
4: -0.0043676, -0.0018928, -0.0043695, -0.0018634, -0.0025042, 0.0024767
5: 0.0033575, 0.0137868, 0.0032166, 0.0139095, -0.0101155, 0.0101148
6: -0.0101534, -0.0025408, -0.0102558, -0.0025425, -0.0076109, 0.0077150
7: -0.0163678, -0.0096178, -0.0163866, -0.0093129, -0.0069926, 0.0067089
8: -0.0149543, -0.0089254, -0.0149751, -0.0088398, -0.0061145, 0.0060496
9: -0.0034489, 0.0032479, -0.0035147, 0.0032580, -0.0067068, 0.0067626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 70

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008278, upper bound: 0.0008345
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008278, upper bound: 0.0008345
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010229, 0.0081786, -0.0007888, 0.0099003, -0.0103469, 0.0084131
1: -0.0036268, 0.0010522, -0.0035823, 0.0020921, -0.0055747, 0.0045126
2: 0.0089005, 0.0167598, 0.0071510, 0.0168926, -0.0079921, 0.0096089
3: 1.0058944, 1.0071493, 1.0057830, 1.0071132, -0.0012188, 0.0013664
4: -0.0043695, -0.0018634, -0.0043900, -0.0012828, -0.0030868, 0.0025266
5: 0.0032020, 0.0139095, 0.0033820, 0.0161074, -0.0124507, 0.0100894
6: -0.0102558, -0.0025419, -0.0120976, -0.0025387, -0.0077170, 0.0095558
7: -0.0163866, -0.0092956, -0.0171659, -0.0096454, -0.0066826, 0.0078078
8: -0.0149751, -0.0088245, -0.0151875, -0.0077523, -0.0072228, 0.0063630
9: -0.0035149, 0.0032580, -0.0050260, 0.0033586, -0.0068735, 0.0082839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008349
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008454, upper bound: 0.0008349
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010133, 0.0081786, -0.0007837, 0.0102666, -0.0107187, 0.0084093
1: -0.0036228, 0.0010522, -0.0035771, 0.0023161, -0.0058033, 0.0045101
2: 0.0089005, 0.0167598, 0.0067873, 0.0169267, -0.0080262, 0.0099726
3: 1.0058949, 1.0071378, 1.0057590, 1.0070903, -0.0011954, 0.0013788
4: -0.0043695, -0.0018634, -0.0043960, -0.0011615, -0.0032081, 0.0025326
5: 0.0032095, 0.0139095, 0.0033860, 0.0165740, -0.0129219, 0.0100865
6: -0.0102558, -0.0025426, -0.0124883, -0.0025403, -0.0077155, 0.0099457
7: -0.0163866, -0.0093005, -0.0173095, -0.0096460, -0.0066816, 0.0079463
8: -0.0149751, -0.0088389, -0.0152551, -0.0075303, -0.0074447, 0.0064162
9: -0.0035148, 0.0032580, -0.0053291, 0.0033921, -0.0069069, 0.0085870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008338, upper bound: 0.0008338
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0007831, 0.0084850, -0.0087583, 0.0083076
1: -0.0035921, 0.0009910, -0.0035771, 0.0012392, -0.0047041, 0.0044357
2: 0.0089864, 0.0167475, 0.0085887, 0.0167879, -0.0078015, 0.0081588
3: 1.0059075, 1.0071299, 1.0058836, 1.0070903, -0.0011828, 0.0012463
4: -0.0043676, -0.0018928, -0.0043745, -0.0017598, -0.0026077, 0.0024817
5: 0.0033575, 0.0137868, 0.0033864, 0.0143007, -0.0105113, 0.0099603
6: -0.0101534, -0.0025408, -0.0105835, -0.0025433, -0.0076101, 0.0080428
7: -0.0163678, -0.0096178, -0.0165253, -0.0096461, -0.0066618, 0.0068478
8: -0.0149543, -0.0089254, -0.0150299, -0.0087157, -0.0062386, 0.0061045
9: -0.0034489, 0.0032479, -0.0037815, 0.0032854, -0.0067342, 0.0070294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 70

Time for candidate selection: 0.35 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008278, upper bound: 0.0008330
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008278
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0008203, 0.0080815, -0.0009874, 0.0084447, -0.0087195, 0.0085072
1: -0.0035921, 0.0009910, -0.0036136, 0.0012150, -0.0046812, 0.0044559
2: 0.0089864, 0.0167475, 0.0086357, 0.0167854, -0.0077990, 0.0081118
3: 1.0059075, 1.0071299, 1.0058812, 1.0071148, -0.0012072, 0.0012487
4: -0.0043676, -0.0018928, -0.0043740, -0.0017749, -0.0025927, 0.0024812
5: 0.0033575, 0.0137868, 0.0032297, 0.0142486, -0.0104610, 0.0101137
6: -0.0101534, -0.0025408, -0.0105399, -0.0025439, -0.0076094, 0.0079991
7: -0.0163678, -0.0096178, -0.0164950, -0.0093188, -0.0069871, 0.0068177
8: -0.0149543, -0.0089254, -0.0150245, -0.0087088, -0.0062455, 0.0060991
9: -0.0034489, 0.0032479, -0.0037369, 0.0032809, -0.0067298, 0.0069849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 237
type: A, layer: 3, pos: 70

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 165

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008278, upper bound: 0.0008345
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008278, upper bound: 0.0008345
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0010229, 0.0081786, -0.0007888, 0.0099003, -0.0103469, 0.0084131
1: -0.0036268, 0.0010522, -0.0035823, 0.0020921, -0.0055747, 0.0045126
2: 0.0089005, 0.0167598, 0.0071510, 0.0168926, -0.0079921, 0.0096089
3: 1.0058944, 1.0071493, 1.0057830, 1.0071132, -0.0012188, 0.0013664
4: -0.0043695, -0.0018634, -0.0043900, -0.0012828, -0.0030868, 0.0025266
5: 0.0032020, 0.0139095, 0.0033820, 0.0161074, -0.0124507, 0.0100894
6: -0.0102558, -0.0025419, -0.0120976, -0.0025387, -0.0077170, 0.0095558
7: -0.0163866, -0.0092956, -0.0171659, -0.0096454, -0.0066826, 0.0078078
8: -0.0149751, -0.0088245, -0.0151875, -0.0077523, -0.0072228, 0.0063630
9: -0.0035149, 0.0032580, -0.0050260, 0.0033586, -0.0068735, 0.0082839

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008349
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008454, upper bound: 0.0008349
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0010133, 0.0081786, -0.0007837, 0.0102666, -0.0107187, 0.0084093
1: -0.0036228, 0.0010522, -0.0035771, 0.0023161, -0.0058033, 0.0045101
2: 0.0089005, 0.0167598, 0.0067873, 0.0169267, -0.0080262, 0.0099726
3: 1.0058949, 1.0071378, 1.0057590, 1.0070903, -0.0011954, 0.0013788
4: -0.0043695, -0.0018634, -0.0043960, -0.0011615, -0.0032081, 0.0025326
5: 0.0032095, 0.0139095, 0.0033860, 0.0165740, -0.0129219, 0.0100865
6: -0.0102558, -0.0025426, -0.0124883, -0.0025403, -0.0077155, 0.0099457
7: -0.0163866, -0.0093005, -0.0173095, -0.0096460, -0.0066816, 0.0079463
8: -0.0149751, -0.0088389, -0.0152551, -0.0075303, -0.0074447, 0.0064162
9: -0.0035148, 0.0032580, -0.0053291, 0.0033921, -0.0069069, 0.0085870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 20
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 17
type: A, layer: 3, pos: 165
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 254
type: A, layer: 3, pos: 189
type: A, layer: 3, pos: 65
type: A, layer: 3, pos: 45
type: A, layer: 3, pos: 253
type: A, layer: 3, pos: 107
type: A, layer: 3, pos: 162
type: A, layer: 3, pos: 73
type: A, layer: 3, pos: 70
type: A, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 20

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008338, upper bound: 0.0008338
time: 1.02 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.66 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009696
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009519
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009504
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009469, upper bound: 0.0009468
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009469, upper bound: 0.0009468
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009468, upper bound: 0.0009468
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009747
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009468, upper bound: 0.0009747
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009467
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009468, upper bound: 0.0009699
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009699
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009696
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009519
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009504
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009469, upper bound: 0.0009468
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009745, upper bound: 0.0009468
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009468, upper bound: 0.0009468
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009468
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009747
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009701
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009701
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009467
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008823
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008733
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008823
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008734
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008384
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008384
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008334
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008334
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008823
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008734
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008829
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008807
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008423
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008423
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008401
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008401
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008413
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008384
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008428
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008423
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008423
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008968, upper bound: 0.0008424
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008401
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008845, upper bound: 0.0008401
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008391
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008334
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008409
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008401
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008423
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008968, upper bound: 0.0008424
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008401
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008845, upper bound: 0.0008401
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008735, upper bound: 0.0009107
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008385, upper bound: 0.0008876
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008781
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008810
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008753
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008810
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008810
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008969
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008846
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008753
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008753
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008846
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008846
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008735, upper bound: 0.0009107
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008385, upper bound: 0.0008876
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008781
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008734, upper bound: 0.0009053
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008807, upper bound: 0.0009053
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008810
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008753
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008810
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008384, upper bound: 0.0008810
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008423, upper bound: 0.0008968
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008401, upper bound: 0.0008846
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008753
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008334, upper bound: 0.0008753
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008401, upper bound: 0.0008846
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008401, upper bound: 0.0008846
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008693, upper bound: 0.0008783
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008692, upper bound: 0.0008692
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008307, upper bound: 0.0008452
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008278, upper bound: 0.0008330
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008305
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008305
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008278
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008278
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008304, upper bound: 0.0008357
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008304, upper bound: 0.0008357
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008304, upper bound: 0.0008454
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008277
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008278
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008339
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008305
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008278, upper bound: 0.0008345
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008278, upper bound: 0.0008345
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008349
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008454, upper bound: 0.0008349
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008338, upper bound: 0.0008338
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008278, upper bound: 0.0008330
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008278
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008278, upper bound: 0.0008345
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008278, upper bound: 0.0008345
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008357, upper bound: 0.0008349
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008454, upper bound: 0.0008349
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008277, upper bound: 0.0008338
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.66
Output dim: 3, lower bound: -0.0008338, upper bound: 0.0008338

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0006523, 0.0099991, -0.0100518, 0.0100518
1: -0.0035917, 0.0021417, -0.0035917, 0.0021417, -0.0056016, 0.0056016
2: 0.0070180, 0.0168748, 0.0070180, 0.0168748, -0.0098568, 0.0098568
3: 1.0058836, 1.0071558, 1.0058836, 1.0071558, -0.0012722, 0.0012722
4: -0.0043845, -0.0012404, -0.0043845, -0.0012404, -0.0031441, 0.0031441
5: 0.0034829, 0.0162374, 0.0034829, 0.0162374, -0.0122819, 0.0122819
6: -0.0122079, -0.0025360, -0.0122079, -0.0025360, -0.0096719, 0.0096719
7: -0.0172598, -0.0101350, -0.0172598, -0.0101350, -0.0070568, 0.0070568
8: -0.0151080, -0.0076373, -0.0151080, -0.0076373, -0.0074707, 0.0074707
9: -0.0051578, 0.0033023, -0.0051578, 0.0033023, -0.0084601, 0.0084601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009696, upper bound: 0.0009651
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0006523, 0.0099991, -0.0102475, 0.0100297
1: -0.0036292, 0.0021306, -0.0035917, 0.0021417, -0.0056173, 0.0055916
2: 0.0070511, 0.0168794, 0.0070180, 0.0168748, -0.0098237, 0.0098614
3: 1.0058714, 1.0071830, 1.0058836, 1.0071558, -0.0012844, 0.0012994
4: -0.0043853, -0.0012508, -0.0043845, -0.0012404, -0.0031449, 0.0031337
5: 0.0033241, 0.0162060, 0.0034829, 0.0162374, -0.0124320, 0.0122511
6: -0.0121811, -0.0025370, -0.0122079, -0.0025360, -0.0096451, 0.0096709
7: -0.0172352, -0.0098209, -0.0172598, -0.0101350, -0.0070330, 0.0073705
8: -0.0151180, -0.0076559, -0.0151080, -0.0076373, -0.0074807, 0.0074521
9: -0.0051244, 0.0033073, -0.0051578, 0.0033023, -0.0084266, 0.0084652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009519, upper bound: 0.0009468
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009468
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006213, 0.0099991, -0.0008509, 0.0099755, -0.0099928, 0.0102380
1: -0.0035823, 0.0021416, -0.0036266, 0.0021305, -0.0055781, 0.0056135
2: 0.0070180, 0.0168748, 0.0070511, 0.0168794, -0.0098613, 0.0098237
3: 1.0058852, 1.0071396, 1.0058719, 1.0071779, -0.0012927, 0.0012677
4: -0.0043845, -0.0012404, -0.0043853, -0.0012508, -0.0031337, 0.0031449
5: 0.0035069, 0.0162374, 0.0033305, 0.0162060, -0.0122223, 0.0124245
6: -0.0122079, -0.0025368, -0.0121811, -0.0025373, -0.0096706, 0.0096443
7: -0.0172597, -0.0101618, -0.0172352, -0.0098284, -0.0073628, 0.0070060
8: -0.0151080, -0.0076387, -0.0151180, -0.0076571, -0.0074509, 0.0074792
9: -0.0051578, 0.0033023, -0.0051243, 0.0033073, -0.0084651, 0.0084266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009469
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0006148, 0.0104695, -0.0008401, 0.0099756, -0.0099934, 0.0107065
1: -0.0035763, 0.0024303, -0.0036220, 0.0021305, -0.0055781, 0.0059050
2: 0.0065562, 0.0169198, 0.0070511, 0.0168794, -0.0103232, 0.0098687
3: 1.0058498, 1.0071174, 1.0058727, 1.0071660, -0.0013162, 0.0012447
4: -0.0043918, -0.0010863, -0.0043853, -0.0012508, -0.0031409, 0.0032990
5: 0.0035121, 0.0168360, 0.0033390, 0.0162060, -0.0122228, 0.0130223
6: -0.0127089, -0.0025387, -0.0121811, -0.0025380, -0.0101708, 0.0096425
7: -0.0174431, -0.0101635, -0.0172352, -0.0098346, -0.0075406, 0.0070048
8: -0.0151858, -0.0073504, -0.0151180, -0.0076587, -0.0075271, 0.0077676
9: -0.0055453, 0.0033407, -0.0051243, 0.0033073, -0.0088527, 0.0084650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009467
time: 1.27 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0006213, 0.0099991, -0.0100518, 0.0100149
1: -0.0035917, 0.0021417, -0.0035823, 0.0021416, -0.0056016, 0.0055880
2: 0.0070180, 0.0168748, 0.0070180, 0.0168748, -0.0098568, 0.0098567
3: 1.0058836, 1.0071558, 1.0058852, 1.0071396, -0.0012560, 0.0012705
4: -0.0043845, -0.0012404, -0.0043845, -0.0012404, -0.0031441, 0.0031441
5: 0.0034829, 0.0162374, 0.0035069, 0.0162374, -0.0122819, 0.0122531
6: -0.0122079, -0.0025360, -0.0122079, -0.0025368, -0.0096711, 0.0096719
7: -0.0172598, -0.0101350, -0.0172597, -0.0101618, -0.0070299, 0.0070568
8: -0.0151080, -0.0076373, -0.0151080, -0.0076387, -0.0074693, 0.0074707
9: -0.0051578, 0.0033023, -0.0051578, 0.0033023, -0.0084601, 0.0084600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009469, upper bound: 0.0009468
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009469, upper bound: 0.0009468
time: 1.07 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0006213, 0.0099991, -0.0102475, 0.0099928
1: -0.0036292, 0.0021306, -0.0035823, 0.0021416, -0.0056173, 0.0055781
2: 0.0070511, 0.0168794, 0.0070180, 0.0168748, -0.0098237, 0.0098613
3: 1.0058714, 1.0071830, 1.0058852, 1.0071396, -0.0012681, 0.0012977
4: -0.0043853, -0.0012508, -0.0043845, -0.0012404, -0.0031449, 0.0031337
5: 0.0033241, 0.0162060, 0.0035069, 0.0162374, -0.0124320, 0.0122223
6: -0.0121811, -0.0025370, -0.0122079, -0.0025368, -0.0096443, 0.0096709
7: -0.0172352, -0.0098209, -0.0172597, -0.0101618, -0.0070060, 0.0073704
8: -0.0151180, -0.0076559, -0.0151080, -0.0076387, -0.0074792, 0.0074521
9: -0.0051244, 0.0033073, -0.0051578, 0.0033023, -0.0084266, 0.0084651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009469, upper bound: 0.0009468
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009469, upper bound: 0.0009468
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0006148, 0.0104695, -0.0105267, 0.0100156
1: -0.0035917, 0.0021417, -0.0035763, 0.0024303, -0.0058952, 0.0055881
2: 0.0070180, 0.0168748, 0.0065562, 0.0169198, -0.0099018, 0.0103186
3: 1.0058836, 1.0071558, 1.0058498, 1.0071174, -0.0012338, 0.0013059
4: -0.0043845, -0.0012404, -0.0043918, -0.0010863, -0.0032982, 0.0031514
5: 0.0034829, 0.0162374, 0.0035121, 0.0168360, -0.0128847, 0.0122537
6: -0.0122079, -0.0025360, -0.0127089, -0.0025387, -0.0096693, 0.0101729
7: -0.0172598, -0.0101350, -0.0174431, -0.0101635, -0.0070287, 0.0072405
8: -0.0151080, -0.0076373, -0.0151858, -0.0073504, -0.0077576, 0.0075486
9: -0.0051578, 0.0033023, -0.0055453, 0.0033407, -0.0084986, 0.0088476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009467
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0006148, 0.0104695, -0.0107224, 0.0099935
1: -0.0036292, 0.0021306, -0.0035763, 0.0024303, -0.0059109, 0.0055782
2: 0.0070511, 0.0168794, 0.0065562, 0.0169198, -0.0098687, 0.0103232
3: 1.0058714, 1.0071830, 1.0058498, 1.0071174, -0.0012460, 0.0013331
4: -0.0043853, -0.0012508, -0.0043918, -0.0010863, -0.0032990, 0.0031410
5: 0.0033241, 0.0162060, 0.0035121, 0.0168360, -0.0130347, 0.0122229
6: -0.0121811, -0.0025370, -0.0127089, -0.0025387, -0.0096425, 0.0101719
7: -0.0172352, -0.0098209, -0.0174431, -0.0101635, -0.0070048, 0.0075542
8: -0.0151180, -0.0076559, -0.0151858, -0.0073504, -0.0077676, 0.0075300
9: -0.0051244, 0.0033073, -0.0055453, 0.0033407, -0.0084651, 0.0088527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 237

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009468, upper bound: 0.0009468
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009467
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006213, 0.0099991, -0.0006523, 0.0099991, -0.0100149, 0.0100518
1: -0.0035823, 0.0021416, -0.0035917, 0.0021417, -0.0055880, 0.0056016
2: 0.0070180, 0.0168748, 0.0070180, 0.0168748, -0.0098567, 0.0098568
3: 1.0058852, 1.0071396, 1.0058836, 1.0071558, -0.0012705, 0.0012560
4: -0.0043845, -0.0012404, -0.0043845, -0.0012404, -0.0031441, 0.0031441
5: 0.0035069, 0.0162374, 0.0034829, 0.0162374, -0.0122531, 0.0122819
6: -0.0122079, -0.0025368, -0.0122079, -0.0025360, -0.0096719, 0.0096711
7: -0.0172597, -0.0101618, -0.0172598, -0.0101350, -0.0070568, 0.0070299
8: -0.0151080, -0.0076387, -0.0151080, -0.0076373, -0.0074707, 0.0074693
9: -0.0051578, 0.0033023, -0.0051578, 0.0033023, -0.0084600, 0.0084601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009468
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008325, 0.0099755, -0.0006523, 0.0099991, -0.0102171, 0.0100297
1: -0.0036210, 0.0021305, -0.0035917, 0.0021417, -0.0056056, 0.0055916
2: 0.0070511, 0.0168794, 0.0070180, 0.0168748, -0.0098237, 0.0098614
3: 1.0058732, 1.0071677, 1.0058836, 1.0071558, -0.0012826, 0.0012841
4: -0.0043853, -0.0012508, -0.0043845, -0.0012404, -0.0031449, 0.0031337
5: 0.0033448, 0.0162060, 0.0034829, 0.0162374, -0.0124083, 0.0122511
6: -0.0121811, -0.0025379, -0.0122079, -0.0025360, -0.0096451, 0.0096700
7: -0.0172352, -0.0098452, -0.0172598, -0.0101350, -0.0070329, 0.0073458
8: -0.0151180, -0.0076594, -0.0151080, -0.0076373, -0.0074807, 0.0074486
9: -0.0051242, 0.0033073, -0.0051578, 0.0033023, -0.0084265, 0.0084652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009463
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009463
time: 1.00 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006213, 0.0099991, -0.0008592, 0.0099755, -0.0099928, 0.0102475
1: -0.0035823, 0.0021416, -0.0036292, 0.0021306, -0.0055781, 0.0056173
2: 0.0070180, 0.0168748, 0.0070511, 0.0168794, -0.0098613, 0.0098237
3: 1.0058852, 1.0071396, 1.0058714, 1.0071830, -0.0012977, 0.0012681
4: -0.0043845, -0.0012404, -0.0043853, -0.0012508, -0.0031337, 0.0031449
5: 0.0035069, 0.0162374, 0.0033241, 0.0162060, -0.0122223, 0.0124320
6: -0.0122079, -0.0025368, -0.0121811, -0.0025370, -0.0096709, 0.0096443
7: -0.0172597, -0.0101618, -0.0172352, -0.0098209, -0.0073704, 0.0070060
8: -0.0151080, -0.0076387, -0.0151180, -0.0076559, -0.0074521, 0.0074792
9: -0.0051578, 0.0033023, -0.0051244, 0.0033073, -0.0084651, 0.0084266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.35 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009747
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0008325, 0.0099755, -0.0008592, 0.0099755, -0.0100717, 0.0101063
1: -0.0036210, 0.0021305, -0.0036292, 0.0021306, -0.0055845, 0.0055980
2: 0.0070511, 0.0168794, 0.0070511, 0.0168794, -0.0098283, 0.0098283
3: 1.0058732, 1.0071677, 1.0058714, 1.0071830, -0.0013098, 0.0012963
4: -0.0043853, -0.0012508, -0.0043853, -0.0012508, -0.0031345, 0.0031345
5: 0.0033448, 0.0162060, 0.0033241, 0.0162060, -0.0122807, 0.0123078
6: -0.0121811, -0.0025379, -0.0121811, -0.0025370, -0.0096441, 0.0096432
7: -0.0172352, -0.0098452, -0.0172352, -0.0098209, -0.0073326, 0.0073079
8: -0.0151180, -0.0076594, -0.0151180, -0.0076559, -0.0074621, 0.0074586
9: -0.0051242, 0.0033073, -0.0051244, 0.0033073, -0.0084316, 0.0084317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009462, upper bound: 0.0009463
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009462, upper bound: 0.0009463
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006148, 0.0104695, -0.0006523, 0.0099991, -0.0100156, 0.0105267
1: -0.0035763, 0.0024303, -0.0035917, 0.0021417, -0.0055881, 0.0058952
2: 0.0065562, 0.0169198, 0.0070180, 0.0168748, -0.0103186, 0.0099018
3: 1.0058498, 1.0071174, 1.0058836, 1.0071558, -0.0013059, 0.0012338
4: -0.0043918, -0.0010863, -0.0043845, -0.0012404, -0.0031514, 0.0032982
5: 0.0035121, 0.0168360, 0.0034829, 0.0162374, -0.0122537, 0.0128847
6: -0.0127089, -0.0025387, -0.0122079, -0.0025360, -0.0101729, 0.0096693
7: -0.0174431, -0.0101635, -0.0172598, -0.0101350, -0.0072405, 0.0070287
8: -0.0151858, -0.0073504, -0.0151080, -0.0076373, -0.0075486, 0.0077576
9: -0.0055453, 0.0033407, -0.0051578, 0.0033023, -0.0088476, 0.0084986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009468
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008134, 0.0102705, -0.0006523, 0.0099991, -0.0102147, 0.0103217
1: -0.0036125, 0.0023114, -0.0035917, 0.0021417, -0.0056063, 0.0057718
2: 0.0067613, 0.0169073, 0.0070180, 0.0168748, -0.0101135, 0.0098893
3: 1.0058534, 1.0071440, 1.0058836, 1.0071558, -0.0013024, 0.0012604
4: -0.0043902, -0.0011543, -0.0043845, -0.0012404, -0.0031498, 0.0032303
5: 0.0033597, 0.0165816, 0.0034829, 0.0162374, -0.0124065, 0.0126253
6: -0.0124955, -0.0025395, -0.0122079, -0.0025360, -0.0099595, 0.0096684
7: -0.0173510, -0.0098539, -0.0172598, -0.0101350, -0.0071483, 0.0073387
8: -0.0151701, -0.0074798, -0.0151080, -0.0076373, -0.0075329, 0.0076282
9: -0.0053681, 0.0033333, -0.0051578, 0.0033023, -0.0086703, 0.0084911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009468
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009468
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006148, 0.0104695, -0.0008592, 0.0099755, -0.0099935, 0.0107224
1: -0.0035763, 0.0024303, -0.0036292, 0.0021306, -0.0055782, 0.0059109
2: 0.0065562, 0.0169198, 0.0070511, 0.0168794, -0.0103232, 0.0098687
3: 1.0058498, 1.0071174, 1.0058714, 1.0071830, -0.0013331, 0.0012460
4: -0.0043918, -0.0010863, -0.0043853, -0.0012508, -0.0031410, 0.0032990
5: 0.0035121, 0.0168360, 0.0033241, 0.0162060, -0.0122229, 0.0130347
6: -0.0127089, -0.0025387, -0.0121811, -0.0025370, -0.0101719, 0.0096425
7: -0.0174431, -0.0101635, -0.0172352, -0.0098209, -0.0075542, 0.0070048
8: -0.0151858, -0.0073504, -0.0151180, -0.0076559, -0.0075300, 0.0077676
9: -0.0055453, 0.0033407, -0.0051244, 0.0033073, -0.0088527, 0.0084651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009698
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0008134, 0.0102705, -0.0008592, 0.0099755, -0.0100674, 0.0104064
1: -0.0036125, 0.0023114, -0.0036292, 0.0021306, -0.0055827, 0.0057845
2: 0.0067613, 0.0169073, 0.0070511, 0.0168794, -0.0101181, 0.0098562
3: 1.0058534, 1.0071440, 1.0058714, 1.0071830, -0.0013295, 0.0012726
4: -0.0043902, -0.0011543, -0.0043853, -0.0012508, -0.0031394, 0.0032311
5: 0.0033597, 0.0165816, 0.0033241, 0.0162060, -0.0122773, 0.0126876
6: -0.0124955, -0.0025395, -0.0121811, -0.0025370, -0.0099585, 0.0096416
7: -0.0173510, -0.0098539, -0.0172352, -0.0098209, -0.0074485, 0.0073005
8: -0.0151701, -0.0074798, -0.0151180, -0.0076559, -0.0075143, 0.0076381
9: -0.0053681, 0.0033333, -0.0051244, 0.0033073, -0.0086754, 0.0084576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009462, upper bound: 0.0009468
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009462, upper bound: 0.0009468
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0006523, 0.0099991, -0.0100518, 0.0100518
1: -0.0035917, 0.0021417, -0.0035917, 0.0021417, -0.0056016, 0.0056016
2: 0.0070180, 0.0168748, 0.0070180, 0.0168748, -0.0098568, 0.0098568
3: 1.0058836, 1.0071558, 1.0058836, 1.0071558, -0.0012722, 0.0012722
4: -0.0043845, -0.0012404, -0.0043845, -0.0012404, -0.0031441, 0.0031441
5: 0.0034829, 0.0162374, 0.0034829, 0.0162374, -0.0122819, 0.0122819
6: -0.0122079, -0.0025360, -0.0122079, -0.0025360, -0.0096719, 0.0096719
7: -0.0172598, -0.0101350, -0.0172598, -0.0101350, -0.0070568, 0.0070568
8: -0.0151080, -0.0076373, -0.0151080, -0.0076373, -0.0074707, 0.0074707
9: -0.0051578, 0.0033023, -0.0051578, 0.0033023, -0.0084601, 0.0084601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009696, upper bound: 0.0009651
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009651, upper bound: 0.0009651
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0006523, 0.0099991, -0.0102475, 0.0100297
1: -0.0036292, 0.0021306, -0.0035917, 0.0021417, -0.0056173, 0.0055916
2: 0.0070511, 0.0168794, 0.0070180, 0.0168748, -0.0098237, 0.0098614
3: 1.0058714, 1.0071830, 1.0058836, 1.0071558, -0.0012844, 0.0012994
4: -0.0043853, -0.0012508, -0.0043845, -0.0012404, -0.0031449, 0.0031337
5: 0.0033241, 0.0162060, 0.0034829, 0.0162374, -0.0124320, 0.0122511
6: -0.0121811, -0.0025370, -0.0122079, -0.0025360, -0.0096451, 0.0096709
7: -0.0172352, -0.0098209, -0.0172598, -0.0101350, -0.0070330, 0.0073705
8: -0.0151180, -0.0076559, -0.0151080, -0.0076373, -0.0074807, 0.0074521
9: -0.0051244, 0.0033073, -0.0051578, 0.0033023, -0.0084266, 0.0084652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009519, upper bound: 0.0009468
time: 0.95 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009468
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006213, 0.0099991, -0.0008509, 0.0099755, -0.0099928, 0.0102380
1: -0.0035823, 0.0021416, -0.0036266, 0.0021305, -0.0055781, 0.0056135
2: 0.0070180, 0.0168748, 0.0070511, 0.0168794, -0.0098613, 0.0098237
3: 1.0058852, 1.0071396, 1.0058719, 1.0071779, -0.0012927, 0.0012677
4: -0.0043845, -0.0012404, -0.0043853, -0.0012508, -0.0031337, 0.0031449
5: 0.0035069, 0.0162374, 0.0033305, 0.0162060, -0.0122223, 0.0124245
6: -0.0122079, -0.0025368, -0.0121811, -0.0025373, -0.0096706, 0.0096443
7: -0.0172597, -0.0101618, -0.0172352, -0.0098284, -0.0073628, 0.0070060
8: -0.0151080, -0.0076387, -0.0151180, -0.0076571, -0.0074509, 0.0074792
9: -0.0051578, 0.0033023, -0.0051243, 0.0033073, -0.0084651, 0.0084266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009469
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009469
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0006148, 0.0104695, -0.0008401, 0.0099756, -0.0099934, 0.0107065
1: -0.0035763, 0.0024303, -0.0036220, 0.0021305, -0.0055781, 0.0059050
2: 0.0065562, 0.0169198, 0.0070511, 0.0168794, -0.0103232, 0.0098687
3: 1.0058498, 1.0071174, 1.0058727, 1.0071660, -0.0013162, 0.0012447
4: -0.0043918, -0.0010863, -0.0043853, -0.0012508, -0.0031409, 0.0032990
5: 0.0035121, 0.0168360, 0.0033390, 0.0162060, -0.0122228, 0.0130223
6: -0.0127089, -0.0025387, -0.0121811, -0.0025380, -0.0101708, 0.0096425
7: -0.0174431, -0.0101635, -0.0172352, -0.0098346, -0.0075406, 0.0070048
8: -0.0151858, -0.0073504, -0.0151180, -0.0076587, -0.0075271, 0.0077676
9: -0.0055453, 0.0033407, -0.0051243, 0.0033073, -0.0088527, 0.0084650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009467
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0006213, 0.0099991, -0.0100518, 0.0100149
1: -0.0035917, 0.0021417, -0.0035823, 0.0021416, -0.0056016, 0.0055880
2: 0.0070180, 0.0168748, 0.0070180, 0.0168748, -0.0098568, 0.0098567
3: 1.0058836, 1.0071558, 1.0058852, 1.0071396, -0.0012560, 0.0012705
4: -0.0043845, -0.0012404, -0.0043845, -0.0012404, -0.0031441, 0.0031441
5: 0.0034829, 0.0162374, 0.0035069, 0.0162374, -0.0122819, 0.0122531
6: -0.0122079, -0.0025360, -0.0122079, -0.0025368, -0.0096711, 0.0096719
7: -0.0172598, -0.0101350, -0.0172597, -0.0101618, -0.0070299, 0.0070568
8: -0.0151080, -0.0076373, -0.0151080, -0.0076387, -0.0074693, 0.0074707
9: -0.0051578, 0.0033023, -0.0051578, 0.0033023, -0.0084601, 0.0084600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009469, upper bound: 0.0009468
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009469, upper bound: 0.0009468
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0006213, 0.0099991, -0.0102475, 0.0099928
1: -0.0036292, 0.0021306, -0.0035823, 0.0021416, -0.0056173, 0.0055781
2: 0.0070511, 0.0168794, 0.0070180, 0.0168748, -0.0098237, 0.0098613
3: 1.0058714, 1.0071830, 1.0058852, 1.0071396, -0.0012681, 0.0012977
4: -0.0043853, -0.0012508, -0.0043845, -0.0012404, -0.0031449, 0.0031337
5: 0.0033241, 0.0162060, 0.0035069, 0.0162374, -0.0124320, 0.0122223
6: -0.0121811, -0.0025370, -0.0122079, -0.0025368, -0.0096443, 0.0096709
7: -0.0172352, -0.0098209, -0.0172597, -0.0101618, -0.0070060, 0.0073704
8: -0.0151180, -0.0076559, -0.0151080, -0.0076387, -0.0074792, 0.0074521
9: -0.0051244, 0.0033073, -0.0051578, 0.0033023, -0.0084266, 0.0084651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009745, upper bound: 0.0009468
time: 1.05 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009745, upper bound: 0.0009468
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0006148, 0.0104695, -0.0105267, 0.0100156
1: -0.0035917, 0.0021417, -0.0035763, 0.0024303, -0.0058952, 0.0055881
2: 0.0070180, 0.0168748, 0.0065562, 0.0169198, -0.0099018, 0.0103186
3: 1.0058836, 1.0071558, 1.0058498, 1.0071174, -0.0012338, 0.0013059
4: -0.0043845, -0.0012404, -0.0043918, -0.0010863, -0.0032982, 0.0031514
5: 0.0034829, 0.0162374, 0.0035121, 0.0168360, -0.0128847, 0.0122537
6: -0.0122079, -0.0025360, -0.0127089, -0.0025387, -0.0096693, 0.0101729
7: -0.0172598, -0.0101350, -0.0174431, -0.0101635, -0.0070287, 0.0072405
8: -0.0151080, -0.0076373, -0.0151858, -0.0073504, -0.0077576, 0.0075486
9: -0.0051578, 0.0033023, -0.0055453, 0.0033407, -0.0084986, 0.0088476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 237

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009467
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0006148, 0.0104695, -0.0107224, 0.0099935
1: -0.0036292, 0.0021306, -0.0035763, 0.0024303, -0.0059109, 0.0055782
2: 0.0070511, 0.0168794, 0.0065562, 0.0169198, -0.0098687, 0.0103232
3: 1.0058714, 1.0071830, 1.0058498, 1.0071174, -0.0012460, 0.0013331
4: -0.0043853, -0.0012508, -0.0043918, -0.0010863, -0.0032990, 0.0031410
5: 0.0033241, 0.0162060, 0.0035121, 0.0168360, -0.0130347, 0.0122229
6: -0.0121811, -0.0025370, -0.0127089, -0.0025387, -0.0096425, 0.0101719
7: -0.0172352, -0.0098209, -0.0174431, -0.0101635, -0.0070048, 0.0075542
8: -0.0151180, -0.0076559, -0.0151858, -0.0073504, -0.0077676, 0.0075300
9: -0.0051244, 0.0033073, -0.0055453, 0.0033407, -0.0084651, 0.0088527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009699, upper bound: 0.0009468
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009699, upper bound: 0.0009468
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006213, 0.0099991, -0.0006523, 0.0099991, -0.0100149, 0.0100518
1: -0.0035823, 0.0021416, -0.0035917, 0.0021417, -0.0055880, 0.0056016
2: 0.0070180, 0.0168748, 0.0070180, 0.0168748, -0.0098567, 0.0098568
3: 1.0058852, 1.0071396, 1.0058836, 1.0071558, -0.0012705, 0.0012560
4: -0.0043845, -0.0012404, -0.0043845, -0.0012404, -0.0031441, 0.0031441
5: 0.0035069, 0.0162374, 0.0034829, 0.0162374, -0.0122531, 0.0122819
6: -0.0122079, -0.0025368, -0.0122079, -0.0025360, -0.0096719, 0.0096711
7: -0.0172597, -0.0101618, -0.0172598, -0.0101350, -0.0070568, 0.0070299
8: -0.0151080, -0.0076387, -0.0151080, -0.0076373, -0.0074707, 0.0074693
9: -0.0051578, 0.0033023, -0.0051578, 0.0033023, -0.0084600, 0.0084601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009468
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008325, 0.0099755, -0.0006523, 0.0099991, -0.0102171, 0.0100297
1: -0.0036210, 0.0021305, -0.0035917, 0.0021417, -0.0056056, 0.0055916
2: 0.0070511, 0.0168794, 0.0070180, 0.0168748, -0.0098237, 0.0098614
3: 1.0058732, 1.0071677, 1.0058836, 1.0071558, -0.0012826, 0.0012841
4: -0.0043853, -0.0012508, -0.0043845, -0.0012404, -0.0031449, 0.0031337
5: 0.0033448, 0.0162060, 0.0034829, 0.0162374, -0.0124083, 0.0122511
6: -0.0121811, -0.0025379, -0.0122079, -0.0025360, -0.0096451, 0.0096700
7: -0.0172352, -0.0098452, -0.0172598, -0.0101350, -0.0070329, 0.0073458
8: -0.0151180, -0.0076594, -0.0151080, -0.0076373, -0.0074807, 0.0074486
9: -0.0051242, 0.0033073, -0.0051578, 0.0033023, -0.0084265, 0.0084652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009462
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009462
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0008325, 0.0099755, -0.0008509, 0.0099755, -0.0100717, 0.0100955
1: -0.0036210, 0.0021305, -0.0036266, 0.0021305, -0.0055845, 0.0055937
2: 0.0070511, 0.0168794, 0.0070511, 0.0168794, -0.0098283, 0.0098283
3: 1.0058732, 1.0071677, 1.0058719, 1.0071779, -0.0013047, 0.0012958
4: -0.0043853, -0.0012508, -0.0043853, -0.0012508, -0.0031345, 0.0031345
5: 0.0033448, 0.0162060, 0.0033305, 0.0162060, -0.0122807, 0.0122993
6: -0.0121811, -0.0025379, -0.0121811, -0.0025373, -0.0096438, 0.0096432
7: -0.0172352, -0.0098452, -0.0172352, -0.0098284, -0.0073249, 0.0073079
8: -0.0151180, -0.0076594, -0.0151180, -0.0076571, -0.0074609, 0.0074586
9: -0.0051242, 0.0033073, -0.0051243, 0.0033073, -0.0084316, 0.0084317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009468
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009701, upper bound: 0.0009747
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0008134, 0.0102705, -0.0008401, 0.0099756, -0.0100674, 0.0103934
1: -0.0036125, 0.0023114, -0.0036220, 0.0021305, -0.0055827, 0.0057797
2: 0.0067613, 0.0169073, 0.0070511, 0.0168794, -0.0101181, 0.0098562
3: 1.0058534, 1.0071440, 1.0058727, 1.0071660, -0.0013126, 0.0012712
4: -0.0043902, -0.0011543, -0.0043853, -0.0012508, -0.0031394, 0.0032311
5: 0.0033597, 0.0165816, 0.0033390, 0.0162060, -0.0122773, 0.0126776
6: -0.0124955, -0.0025395, -0.0121811, -0.0025380, -0.0099575, 0.0096416
7: -0.0173510, -0.0098539, -0.0172352, -0.0098346, -0.0074349, 0.0073005
8: -0.0151701, -0.0074798, -0.0151180, -0.0076587, -0.0075114, 0.0076381
9: -0.0053681, 0.0033333, -0.0051243, 0.0033073, -0.0086754, 0.0084576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009463
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009700, upper bound: 0.0009701
time: 0.95 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006148, 0.0104695, -0.0006523, 0.0099991, -0.0100156, 0.0105267
1: -0.0035763, 0.0024303, -0.0035917, 0.0021417, -0.0055881, 0.0058952
2: 0.0065562, 0.0169198, 0.0070180, 0.0168748, -0.0103186, 0.0099018
3: 1.0058498, 1.0071174, 1.0058836, 1.0071558, -0.0013059, 0.0012338
4: -0.0043918, -0.0010863, -0.0043845, -0.0012404, -0.0031514, 0.0032982
5: 0.0035121, 0.0168360, 0.0034829, 0.0162374, -0.0122537, 0.0128847
6: -0.0127089, -0.0025387, -0.0122079, -0.0025360, -0.0101729, 0.0096693
7: -0.0174431, -0.0101635, -0.0172598, -0.0101350, -0.0072405, 0.0070287
8: -0.0151858, -0.0073504, -0.0151080, -0.0076373, -0.0075486, 0.0077576
9: -0.0055453, 0.0033407, -0.0051578, 0.0033023, -0.0088476, 0.0084986

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009468
time: 1.00 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009467, upper bound: 0.0009468
time: 1.22 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008134, 0.0102705, -0.0006523, 0.0099991, -0.0102147, 0.0103217
1: -0.0036125, 0.0023114, -0.0035917, 0.0021417, -0.0056063, 0.0057718
2: 0.0067613, 0.0169073, 0.0070180, 0.0168748, -0.0101135, 0.0098893
3: 1.0058534, 1.0071440, 1.0058836, 1.0071558, -0.0013024, 0.0012604
4: -0.0043902, -0.0011543, -0.0043845, -0.0012404, -0.0031498, 0.0032303
5: 0.0033597, 0.0165816, 0.0034829, 0.0162374, -0.0124065, 0.0126253
6: -0.0124955, -0.0025395, -0.0122079, -0.0025360, -0.0099595, 0.0096684
7: -0.0173510, -0.0098539, -0.0172598, -0.0101350, -0.0071483, 0.0073387
8: -0.0151701, -0.0074798, -0.0151080, -0.0076373, -0.0075329, 0.0076282
9: -0.0053681, 0.0033333, -0.0051578, 0.0033023, -0.0086703, 0.0084911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009468
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009503, upper bound: 0.0009468
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0008325, 0.0099755, -0.0008509, 0.0099755, -0.0100717, 0.0100955
1: -0.0036210, 0.0021305, -0.0036266, 0.0021305, -0.0055845, 0.0055937
2: 0.0070511, 0.0168794, 0.0070511, 0.0168794, -0.0098283, 0.0098283
3: 1.0058732, 1.0071677, 1.0058719, 1.0071779, -0.0013047, 0.0012958
4: -0.0043853, -0.0012508, -0.0043853, -0.0012508, -0.0031345, 0.0031345
5: 0.0033448, 0.0162060, 0.0033305, 0.0162060, -0.0122807, 0.0122993
6: -0.0121811, -0.0025379, -0.0121811, -0.0025373, -0.0096438, 0.0096432
7: -0.0172352, -0.0098452, -0.0172352, -0.0098284, -0.0073249, 0.0073079
8: -0.0151180, -0.0076594, -0.0151180, -0.0076571, -0.0074609, 0.0074586
9: -0.0051242, 0.0033073, -0.0051243, 0.0033073, -0.0084316, 0.0084317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009468
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009701, upper bound: 0.0009747
time: 0.93 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0008134, 0.0102705, -0.0008401, 0.0099756, -0.0100674, 0.0103934
1: -0.0036125, 0.0023114, -0.0036220, 0.0021305, -0.0055827, 0.0057797
2: 0.0067613, 0.0169073, 0.0070511, 0.0168794, -0.0101181, 0.0098562
3: 1.0058534, 1.0071440, 1.0058727, 1.0071660, -0.0013126, 0.0012712
4: -0.0043902, -0.0011543, -0.0043853, -0.0012508, -0.0031394, 0.0032311
5: 0.0033597, 0.0165816, 0.0033390, 0.0162060, -0.0122773, 0.0126776
6: -0.0124955, -0.0025395, -0.0121811, -0.0025380, -0.0099575, 0.0096416
7: -0.0173510, -0.0098539, -0.0172352, -0.0098346, -0.0074349, 0.0073005
8: -0.0151701, -0.0074798, -0.0151180, -0.0076587, -0.0075114, 0.0076381
9: -0.0053681, 0.0033333, -0.0051243, 0.0033073, -0.0086754, 0.0084576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 64
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 237

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009698, upper bound: 0.0009468
time: 0.98 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009701, upper bound: 0.0009701
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0008203, 0.0080815, -0.0081549, 0.0102628
1: -0.0035917, 0.0021417, -0.0035921, 0.0009910, -0.0044458, 0.0056085
2: 0.0070180, 0.0168748, 0.0089864, 0.0167475, -0.0097295, 0.0078884
3: 1.0058836, 1.0071558, 1.0059075, 1.0071299, -0.0012463, 0.0012482
4: -0.0043845, -0.0012404, -0.0043676, -0.0018928, -0.0024917, 0.0031272
5: 0.0034829, 0.0162374, 0.0033575, 0.0137868, -0.0098461, 0.0124410
6: -0.0122079, -0.0025360, -0.0101534, -0.0025408, -0.0096671, 0.0076173
7: -0.0172598, -0.0101350, -0.0163678, -0.0096178, -0.0075805, 0.0061688
8: -0.0151080, -0.0076373, -0.0149543, -0.0089254, -0.0061826, 0.0073170
9: -0.0051578, 0.0033023, -0.0034489, 0.0032479, -0.0084058, 0.0067511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 70

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009107, upper bound: 0.0008735
time: 1.02 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008734
time: 0.94 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0008203, 0.0080815, -0.0083506, 0.0102411
1: -0.0036292, 0.0021306, -0.0035921, 0.0009910, -0.0044615, 0.0055990
2: 0.0070511, 0.0168794, 0.0089864, 0.0167475, -0.0096964, 0.0078930
3: 1.0058714, 1.0071830, 1.0059075, 1.0071299, -0.0012585, 0.0012754
4: -0.0043853, -0.0012508, -0.0043676, -0.0018928, -0.0024925, 0.0031167
5: 0.0033241, 0.0162060, 0.0033575, 0.0137868, -0.0099961, 0.0124104
6: -0.0121811, -0.0025370, -0.0101534, -0.0025408, -0.0096403, 0.0076164
7: -0.0172352, -0.0098209, -0.0163678, -0.0096178, -0.0075570, 0.0064824
8: -0.0151180, -0.0076559, -0.0149543, -0.0089254, -0.0061925, 0.0072984
9: -0.0051244, 0.0033073, -0.0034489, 0.0032479, -0.0083723, 0.0067562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 237
type: B, layer: 3, pos: 70

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008876, upper bound: 0.0008385
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008781, upper bound: 0.0008334
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0010320, 0.0081786, -0.0082542, 0.0104603
1: -0.0035917, 0.0021417, -0.0036297, 0.0010522, -0.0045092, 0.0056270
2: 0.0070180, 0.0168748, 0.0089004, 0.0167598, -0.0097418, 0.0079744
3: 1.0058836, 1.0071558, 1.0058941, 1.0071546, -0.0012710, 0.0012617
4: -0.0043845, -0.0012404, -0.0043695, -0.0018634, -0.0025211, 0.0031292
5: 0.0034829, 0.0162374, 0.0031950, 0.0139095, -0.0099703, 0.0125925
6: -0.0122079, -0.0025360, -0.0102558, -0.0025416, -0.0096664, 0.0077198
7: -0.0172598, -0.0101350, -0.0163867, -0.0092872, -0.0079089, 0.0061876
8: -0.0151080, -0.0076373, -0.0149751, -0.0088170, -0.0062910, 0.0073378
9: -0.0051578, 0.0033023, -0.0035149, 0.0032580, -0.0084158, 0.0068172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008733
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0009053, upper bound: 0.0008734
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0010320, 0.0081786, -0.0083256, 0.0103213
1: -0.0036292, 0.0021306, -0.0036297, 0.0010522, -0.0045144, 0.0056046
2: 0.0070511, 0.0168794, 0.0089004, 0.0167598, -0.0097088, 0.0079789
3: 1.0058714, 1.0071830, 1.0058941, 1.0071546, -0.0012832, 0.0012889
4: -0.0043853, -0.0012508, -0.0043695, -0.0018634, -0.0025219, 0.0031187
5: 0.0033241, 0.0162060, 0.0031950, 0.0139095, -0.0100235, 0.0124695
6: -0.0121811, -0.0025370, -0.0102558, -0.0025416, -0.0096396, 0.0077188
7: -0.0172352, -0.0098209, -0.0163867, -0.0092872, -0.0078719, 0.0064876
8: -0.0151180, -0.0076559, -0.0149751, -0.0088170, -0.0063010, 0.0073192
9: -0.0051244, 0.0033073, -0.0035149, 0.0032580, -0.0083823, 0.0068223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008384
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008334
time: 0.92 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0007882, 0.0080815, -0.0081548, 0.0102262
1: -0.0035917, 0.0021417, -0.0035823, 0.0009909, -0.0044458, 0.0055938
2: 0.0070180, 0.0168748, 0.0089866, 0.0167475, -0.0097295, 0.0078882
3: 1.0058836, 1.0071558, 1.0059092, 1.0071132, -0.0012296, 0.0012466
4: -0.0043845, -0.0012404, -0.0043676, -0.0018929, -0.0024917, 0.0031272
5: 0.0034829, 0.0162374, 0.0033824, 0.0137868, -0.0098461, 0.0124124
6: -0.0122079, -0.0025360, -0.0101534, -0.0025417, -0.0096662, 0.0076173
7: -0.0172598, -0.0101350, -0.0163677, -0.0096455, -0.0075528, 0.0061687
8: -0.0151080, -0.0076373, -0.0149543, -0.0089442, -0.0061638, 0.0073170
9: -0.0051578, 0.0033023, -0.0034487, 0.0032479, -0.0084058, 0.0067509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 20

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008384
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008384
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0008592, 0.0099755, -0.0007882, 0.0080815, -0.0083505, 0.0102045
1: -0.0036292, 0.0021306, -0.0035823, 0.0009909, -0.0044615, 0.0055843
2: 0.0070511, 0.0168794, 0.0089866, 0.0167475, -0.0096964, 0.0078928
3: 1.0058714, 1.0071830, 1.0059092, 1.0071132, -0.0012418, 0.0012738
4: -0.0043853, -0.0012508, -0.0043676, -0.0018929, -0.0024925, 0.0031167
5: 0.0033241, 0.0162060, 0.0033824, 0.0137868, -0.0099961, 0.0123818
6: -0.0121811, -0.0025370, -0.0101534, -0.0025417, -0.0096394, 0.0076164
7: -0.0172352, -0.0098209, -0.0163677, -0.0096455, -0.0075293, 0.0064824
8: -0.0151180, -0.0076559, -0.0149543, -0.0089442, -0.0061738, 0.0072984
9: -0.0051244, 0.0033073, -0.0034487, 0.0032479, -0.0083723, 0.0067560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 165
type: B, layer: 3, pos: 20
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 17
type: B, layer: 3, pos: 37
type: B, layer: 3, pos: 254
type: B, layer: 3, pos: 65
type: B, layer: 3, pos: 45
type: B, layer: 3, pos: 189
type: B, layer: 3, pos: 107
type: B, layer: 3, pos: 253
type: B, layer: 3, pos: 162
type: B, layer: 3, pos: 73
type: B, layer: 3, pos: 70
type: B, layer: 3, pos: 237

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 165

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008810, upper bound: 0.0008384
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0008753, upper bound: 0.0008334
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0006523, 0.0099991, -0.0007831, 0.0084850, -0.0085672, 0.0102290
1: -0.0035917, 0.0021417, -0.0035771, 0.0012392, -0.0046993, 0.0055944
2: 0.0070180, 0.0168748, 0.0085887, 0.0167879, -0.0097698, 0.0082861
3: 1.0058836, 1.0071558, 1.0058836, 1.0070903, -0.0012068, 0.0012722
4: -0.0043845, -0.0012404, -0.0043745, -0.0017598, -0.0026247, 0.0031341
5: 0.0034829, 0.0162374, 0.0033864, 0.0143007, -0.0103675, 0.0124147
6: -0.0122079, -0.0025360, -0.0105835, -0.0025433, -0.0096646, 0.0080475
7: -0.0172598, -0.0101350, -0.0165253, -0.0096461, -0.0075525, 0.0063267
8: -0.0151080, -0.0076373, -0.0150299, -0.0087157, -0.0063923, 0.0073926
9: -0.0051578, 0.0033023, -0.0037815, 0.0032854, -0.0084432, 0.0070837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.13 + 597.46 = 600.59 seconds
