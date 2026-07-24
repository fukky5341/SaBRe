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
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041535, -0.0041194, -0.0041535, -0.0041194, -0.0000291, 0.0000291)
1: (-0.0082052, -0.0069290, -0.0082052, -0.0069290, -0.0010884, 0.0010884)
2: (0.9666169, 0.9681484, 0.9666169, 0.9681484, -0.0013061, 0.0013061)
3: (0.0000776, 0.0113740, 0.0000776, 0.0113740, -0.0096337, 0.0096337)
4: (-0.0015581, -0.0006989, -0.0015581, -0.0006989, -0.0007327, 0.0007327)
5: (0.0156956, 0.0165639, 0.0156956, 0.0165639, -0.0007405, 0.0007405)
6: (0.0038501, 0.0042724, 0.0038501, 0.0042724, -0.0003602, 0.0003602)
7: (-0.0107259, -0.0077984, -0.0107259, -0.0077984, -0.0024966, 0.0024966)
8: (0.0082197, 0.0105423, 0.0082197, 0.0105423, -0.0019807, 0.0019807)
9: (0.0125086, 0.0166859, 0.0125086, 0.0166859, -0.0035625, 0.0035625)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.52 + 1.61 = 3.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0008934, upper bound: 0.0008934

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008393, upper bound: 0.0008203
time: 0.73 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008397, upper bound: 0.0008398
time: 0.73 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.61 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 2, lower bound: -0.0008393, upper bound: 0.0008203
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.61
Output dim: 2, lower bound: -0.0008397, upper bound: 0.0008398

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0041547, -0.0041224, -0.0041535, -0.0041200, -0.0000282, 0.0000256
1: -0.0082501, -0.0070392, -0.0082038, -0.0069510, -0.0010557, 0.0009576
2: 0.9665630, 0.9680161, 0.9666185, 0.9681219, -0.0012669, 0.0011491
3: -0.0003201, 0.0103981, 0.0000904, 0.0111789, -0.0093445, 0.0084757
4: -0.0014839, -0.0006687, -0.0015433, -0.0006999, -0.0006446, 0.0007107
5: 0.0157706, 0.0165945, 0.0157106, 0.0165630, -0.0006515, 0.0007183
6: 0.0038352, 0.0042360, 0.0038506, 0.0042651, -0.0003494, 0.0003169
7: -0.0104730, -0.0076953, -0.0106754, -0.0078017, -0.0021965, 0.0024217
8: 0.0084204, 0.0106241, 0.0082598, 0.0105397, -0.0017426, 0.0019213
9: 0.0128694, 0.0168330, 0.0125807, 0.0166812, -0.0031343, 0.0034556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008202, upper bound: 0.0008202
time: 0.76 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008202, upper bound: 0.0008202
time: 0.73 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0041535, -0.0041201, -0.0041535, -0.0041194, -0.0000290, 0.0000259
1: -0.0082038, -0.0069536, -0.0082052, -0.0069290, -0.0010864, 0.0009701
2: 0.9666185, 0.9681188, 0.9666169, 0.9681484, -0.0013037, 0.0011642
3: 0.0000898, 0.0111559, 0.0000776, 0.0113740, -0.0096157, 0.0085867
4: -0.0015415, -0.0006999, -0.0015581, -0.0006989, -0.0006531, 0.0007313
5: 0.0157124, 0.0165630, 0.0156956, 0.0165639, -0.0006600, 0.0007391
6: 0.0038505, 0.0042643, 0.0038501, 0.0042724, -0.0003595, 0.0003210
7: -0.0106694, -0.0078015, -0.0107259, -0.0077984, -0.0022253, 0.0024920
8: 0.0082645, 0.0105398, 0.0082197, 0.0105423, -0.0017655, 0.0019770
9: 0.0125892, 0.0166814, 0.0125086, 0.0166859, -0.0031753, 0.0035559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008202, upper bound: 0.0008393
time: 0.77 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0008202, upper bound: 0.0008398
time: 0.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.02 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 2, lower bound: -0.0008202, upper bound: 0.0008202
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 2, lower bound: -0.0008202, upper bound: 0.0008202
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 2, lower bound: -0.0008202, upper bound: 0.0008393
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.02
Output dim: 2, lower bound: -0.0008202, upper bound: 0.0008398

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041547, -0.0041224, -0.0041547, -0.0041224, -0.0000256, 0.0000256
1: -0.0082501, -0.0070392, -0.0082501, -0.0070392, -0.0009573, 0.0009573
2: 0.9665630, 0.9680161, 0.9665630, 0.9680161, -0.0011488, 0.0011488
3: -0.0003201, 0.0103981, -0.0003201, 0.0103981, -0.0084732, 0.0084732
4: -0.0014839, -0.0006687, -0.0014839, -0.0006687, -0.0006444, 0.0006444
5: 0.0157706, 0.0165945, 0.0157706, 0.0165945, -0.0006513, 0.0006513
6: 0.0038352, 0.0042360, 0.0038352, 0.0042360, -0.0003168, 0.0003168
7: -0.0104730, -0.0076953, -0.0104730, -0.0076953, -0.0021959, 0.0021959
8: 0.0084204, 0.0106241, 0.0084204, 0.0106241, -0.0017421, 0.0017421
9: 0.0128694, 0.0168330, 0.0128694, 0.0168330, -0.0031334, 0.0031334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0007900
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007728, upper bound: 0.0007728
time: 0.77 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041547, -0.0041224, -0.0041535, -0.0041201, -0.0000285, 0.0000256
1: -0.0082501, -0.0070392, -0.0082038, -0.0069536, -0.0010690, 0.0009570
2: 0.9665630, 0.9680161, 0.9666185, 0.9681188, -0.0012829, 0.0011485
3: -0.0003201, 0.0103981, 0.0000898, 0.0111559, -0.0094624, 0.0084711
4: -0.0014839, -0.0006687, -0.0015415, -0.0006999, -0.0006443, 0.0007197
5: 0.0157706, 0.0165945, 0.0157124, 0.0165630, -0.0006512, 0.0007274
6: 0.0038352, 0.0042360, 0.0038505, 0.0042643, -0.0003538, 0.0003167
7: -0.0104730, -0.0076953, -0.0106694, -0.0078015, -0.0021954, 0.0024523
8: 0.0084204, 0.0106241, 0.0082645, 0.0105398, -0.0017417, 0.0019455
9: 0.0128694, 0.0168330, 0.0125892, 0.0166814, -0.0031326, 0.0034992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0007900
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007728, upper bound: 0.0007727
time: 0.77 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041535, -0.0041201, -0.0041547, -0.0041224, -0.0000256, 0.0000285
1: -0.0082038, -0.0069536, -0.0082501, -0.0070392, -0.0009570, 0.0010690
2: 0.9666185, 0.9681188, 0.9665630, 0.9680161, -0.0011485, 0.0012829
3: 0.0000898, 0.0111559, -0.0003201, 0.0103981, -0.0084711, 0.0094624
4: -0.0015415, -0.0006999, -0.0014839, -0.0006687, -0.0007197, 0.0006443
5: 0.0157124, 0.0165630, 0.0157706, 0.0165945, -0.0007274, 0.0006512
6: 0.0038505, 0.0042643, 0.0038352, 0.0042360, -0.0003167, 0.0003538
7: -0.0106694, -0.0078015, -0.0104730, -0.0076953, -0.0024523, 0.0021954
8: 0.0082645, 0.0105398, 0.0084204, 0.0106241, -0.0019455, 0.0017417
9: 0.0125892, 0.0166814, 0.0128694, 0.0168330, -0.0034992, 0.0031326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0008126
time: 0.72 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007728, upper bound: 0.0008024
time: 0.78 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041535, -0.0041201, -0.0041535, -0.0041201, -0.0000259, 0.0000259
1: -0.0082038, -0.0069536, -0.0082038, -0.0069536, -0.0009682, 0.0009682
2: 0.9666185, 0.9681188, 0.9666185, 0.9681188, -0.0011618, 0.0011618
3: 0.0000898, 0.0111559, 0.0000898, 0.0111559, -0.0085695, 0.0085695
4: -0.0015415, -0.0006999, -0.0015415, -0.0006999, -0.0006518, 0.0006518
5: 0.0157124, 0.0165630, 0.0157124, 0.0165630, -0.0006587, 0.0006587
6: 0.0038505, 0.0042643, 0.0038505, 0.0042643, -0.0003204, 0.0003204
7: -0.0106694, -0.0078015, -0.0106694, -0.0078015, -0.0022209, 0.0022209
8: 0.0082645, 0.0105398, 0.0082645, 0.0105398, -0.0017619, 0.0017619
9: 0.0125892, 0.0166814, 0.0125892, 0.0166814, -0.0031690, 0.0031690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0008152
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007728, upper bound: 0.0008105
time: 0.76 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.06 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0007900
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 2, lower bound: -0.0007728, upper bound: 0.0007728
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0007900
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 2, lower bound: -0.0007728, upper bound: 0.0007727
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0008126
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 2, lower bound: -0.0007728, upper bound: 0.0008024
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0008152
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.06
Output dim: 2, lower bound: -0.0007728, upper bound: 0.0008105

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041540, -0.0041224, -0.0041547, -0.0041224, -0.0000248, 0.0000255
1: -0.0082221, -0.0070407, -0.0082501, -0.0070392, -0.0009287, 0.0009557
2: 0.9665967, 0.9680144, 0.9665630, 0.9680161, -0.0011145, 0.0011469
3: -0.0000718, 0.0103849, -0.0003201, 0.0103981, -0.0082201, 0.0084593
4: -0.0014829, -0.0006876, -0.0014839, -0.0006687, -0.0006434, 0.0006252
5: 0.0157716, 0.0165754, 0.0157706, 0.0165945, -0.0006502, 0.0006319
6: 0.0038445, 0.0042355, 0.0038352, 0.0042360, -0.0003073, 0.0003163
7: -0.0104696, -0.0077596, -0.0104730, -0.0076953, -0.0021923, 0.0021303
8: 0.0084231, 0.0105730, 0.0084204, 0.0106241, -0.0017393, 0.0016901
9: 0.0128743, 0.0167412, 0.0128694, 0.0168330, -0.0031282, 0.0030398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0007584
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0007728
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041539, -0.0041211, -0.0041545, -0.0041224, -0.0000250, 0.0000268
1: -0.0082177, -0.0069926, -0.0082420, -0.0070402, -0.0009352, 0.0010021
2: 0.9666018, 0.9680719, 0.9665727, 0.9680149, -0.0011223, 0.0012026
3: -0.0000334, 0.0108103, -0.0002480, 0.0103892, -0.0082780, 0.0088702
4: -0.0015152, -0.0006905, -0.0014832, -0.0006742, -0.0006746, 0.0006296
5: 0.0157389, 0.0165725, 0.0157713, 0.0165890, -0.0006818, 0.0006363
6: 0.0038459, 0.0042514, 0.0038379, 0.0042356, -0.0003095, 0.0003316
7: -0.0105798, -0.0077696, -0.0104707, -0.0077140, -0.0022988, 0.0021453
8: 0.0083356, 0.0105651, 0.0084222, 0.0106092, -0.0018237, 0.0017020
9: 0.0127170, 0.0167270, 0.0128727, 0.0168063, -0.0032802, 0.0030612

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007190, upper bound: 0.0007186
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007081, upper bound: 0.0007082
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041540, -0.0041224, -0.0041535, -0.0041201, -0.0000278, 0.0000255
1: -0.0082221, -0.0070407, -0.0082038, -0.0069536, -0.0010404, 0.0009555
2: 0.9665967, 0.9680144, 0.9666185, 0.9681188, -0.0012486, 0.0011466
3: -0.0000718, 0.0103849, 0.0000898, 0.0111559, -0.0092092, 0.0084571
4: -0.0014829, -0.0006876, -0.0015415, -0.0006999, -0.0006432, 0.0007004
5: 0.0157716, 0.0165754, 0.0157124, 0.0165630, -0.0006501, 0.0007079
6: 0.0038445, 0.0042355, 0.0038505, 0.0042643, -0.0003443, 0.0003162
7: -0.0104696, -0.0077596, -0.0106694, -0.0078015, -0.0021917, 0.0023866
8: 0.0084231, 0.0105730, 0.0082645, 0.0105398, -0.0017388, 0.0018935
9: 0.0128743, 0.0167412, 0.0125892, 0.0166814, -0.0031274, 0.0034056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007845, upper bound: 0.0007584
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007845, upper bound: 0.0007727
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041539, -0.0041211, -0.0041533, -0.0041201, -0.0000280, 0.0000267
1: -0.0082177, -0.0069926, -0.0081959, -0.0069545, -0.0010472, 0.0010009
2: 0.9666018, 0.9680719, 0.9666280, 0.9681178, -0.0012567, 0.0012011
3: -0.0000334, 0.0108103, 0.0001601, 0.0111479, -0.0092691, 0.0088593
4: -0.0015152, -0.0006905, -0.0015409, -0.0007052, -0.0006738, 0.0007050
5: 0.0157389, 0.0165725, 0.0157130, 0.0165576, -0.0006810, 0.0007125
6: 0.0038459, 0.0042514, 0.0038532, 0.0042640, -0.0003466, 0.0003312
7: -0.0105798, -0.0077696, -0.0106673, -0.0078197, -0.0022960, 0.0024022
8: 0.0083356, 0.0105651, 0.0082662, 0.0105253, -0.0018215, 0.0019058
9: 0.0127170, 0.0167270, 0.0125922, 0.0166554, -0.0032762, 0.0034277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007556, upper bound: 0.0007188
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007537, upper bound: 0.0007086
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041527, -0.0041201, -0.0041547, -0.0041224, -0.0000248, 0.0000285
1: -0.0081759, -0.0069552, -0.0082501, -0.0070392, -0.0009298, 0.0010676
2: 0.9666520, 0.9681171, 0.9665630, 0.9680161, -0.0011158, 0.0012811
3: 0.0003366, 0.0111422, -0.0003201, 0.0103981, -0.0082301, 0.0094495
4: -0.0015405, -0.0007186, -0.0014839, -0.0006687, -0.0007187, 0.0006259
5: 0.0157134, 0.0165440, 0.0157706, 0.0165945, -0.0007264, 0.0006326
6: 0.0038598, 0.0042638, 0.0038352, 0.0042360, -0.0003077, 0.0003533
7: -0.0106659, -0.0078655, -0.0104730, -0.0076953, -0.0024489, 0.0021329
8: 0.0082674, 0.0104890, 0.0084204, 0.0106241, -0.0019428, 0.0016921
9: 0.0125943, 0.0165902, 0.0128694, 0.0168330, -0.0034944, 0.0030435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0007845
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0008024
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041527, -0.0041188, -0.0041545, -0.0041224, -0.0000250, 0.0000296
1: -0.0081735, -0.0069047, -0.0082420, -0.0070402, -0.0009346, 0.0011088
2: 0.9666550, 0.9681776, 0.9665727, 0.9680149, -0.0011216, 0.0013307
3: 0.0003583, 0.0115891, -0.0002480, 0.0103892, -0.0082729, 0.0098147
4: -0.0015744, -0.0007203, -0.0014832, -0.0006742, -0.0007465, 0.0006292
5: 0.0156791, 0.0165424, 0.0157713, 0.0165890, -0.0007544, 0.0006359
6: 0.0038606, 0.0042805, 0.0038379, 0.0042356, -0.0003093, 0.0003670
7: -0.0107817, -0.0078711, -0.0104707, -0.0077140, -0.0025436, 0.0021440
8: 0.0081755, 0.0104846, 0.0084222, 0.0106092, -0.0020179, 0.0017009
9: 0.0124290, 0.0165821, 0.0128727, 0.0168063, -0.0036295, 0.0030593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007195, upper bound: 0.0007660
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007086, upper bound: 0.0007537
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041527, -0.0041201, -0.0041535, -0.0041201, -0.0000251, 0.0000258
1: -0.0081759, -0.0069552, -0.0082038, -0.0069536, -0.0009398, 0.0009662
2: 0.9666520, 0.9681171, 0.9666185, 0.9681188, -0.0011278, 0.0011595
3: 0.0003366, 0.0111422, 0.0000898, 0.0111559, -0.0083185, 0.0085522
4: -0.0015405, -0.0007186, -0.0015415, -0.0006999, -0.0006504, 0.0006327
5: 0.0157134, 0.0165440, 0.0157124, 0.0165630, -0.0006574, 0.0006394
6: 0.0038598, 0.0042638, 0.0038505, 0.0042643, -0.0003110, 0.0003198
7: -0.0106659, -0.0078655, -0.0106694, -0.0078015, -0.0022164, 0.0021558
8: 0.0082674, 0.0104890, 0.0082645, 0.0105398, -0.0017584, 0.0017103
9: 0.0125943, 0.0165902, 0.0125892, 0.0166814, -0.0031626, 0.0030762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007667, upper bound: 0.0007861
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007667, upper bound: 0.0008105
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041527, -0.0041188, -0.0041533, -0.0041201, -0.0000253, 0.0000271
1: -0.0081735, -0.0069047, -0.0081959, -0.0069545, -0.0009462, 0.0010146
2: 0.9666550, 0.9681776, 0.9666280, 0.9681178, -0.0011355, 0.0012176
3: 0.0003583, 0.0115891, 0.0001601, 0.0111479, -0.0083752, 0.0089807
4: -0.0015744, -0.0007203, -0.0015409, -0.0007052, -0.0006830, 0.0006370
5: 0.0156791, 0.0165424, 0.0157130, 0.0165576, -0.0006903, 0.0006438
6: 0.0038606, 0.0042805, 0.0038532, 0.0042640, -0.0003131, 0.0003358
7: -0.0107817, -0.0078711, -0.0106673, -0.0078197, -0.0023274, 0.0021705
8: 0.0081755, 0.0104846, 0.0082662, 0.0105253, -0.0018465, 0.0017220
9: 0.0124290, 0.0165821, 0.0125922, 0.0166554, -0.0033211, 0.0030971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005454, upper bound: 0.0004876
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003492, upper bound: 0.0003571
time: 0.60 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.87 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0007584
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0007728
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0007190, upper bound: 0.0007186
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0007081, upper bound: 0.0007082
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0007845, upper bound: 0.0007584
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0007845, upper bound: 0.0007727
IS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0007556, upper bound: 0.0007188
IS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0007537, upper bound: 0.0007086
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0007845
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0007584, upper bound: 0.0008024
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0007195, upper bound: 0.0007660
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0007086, upper bound: 0.0007537
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0007667, upper bound: 0.0007861
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0007667, upper bound: 0.0008105
IS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0005454, upper bound: 0.0004876
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.87
Output dim: 2, lower bound: -0.0003492, upper bound: 0.0003571

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041540, -0.0041224, -0.0041539, -0.0041211, -0.0000262, 0.0000249
1: -0.0082221, -0.0070407, -0.0082177, -0.0069926, -0.0009809, 0.0009306
2: 0.9665967, 0.9680144, 0.9666018, 0.9680719, -0.0011772, 0.0011167
3: -0.0000718, 0.0103849, -0.0000334, 0.0108103, -0.0086827, 0.0082369
4: -0.0014829, -0.0006876, -0.0015152, -0.0006905, -0.0006265, 0.0006604
5: 0.0157716, 0.0165754, 0.0157389, 0.0165725, -0.0006332, 0.0006674
6: 0.0038445, 0.0042355, 0.0038459, 0.0042514, -0.0003246, 0.0003080
7: -0.0104696, -0.0077596, -0.0105798, -0.0077696, -0.0021347, 0.0022502
8: 0.0084231, 0.0105730, 0.0083356, 0.0105651, -0.0016935, 0.0017852
9: 0.0128743, 0.0167412, 0.0127170, 0.0167270, -0.0030460, 0.0032108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007106, upper bound: 0.0007508
time: 0.76 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007014, upper bound: 0.0007480
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041540, -0.0041224, -0.0041527, -0.0041201, -0.0000277, 0.0000248
1: -0.0082221, -0.0070407, -0.0081759, -0.0069552, -0.0010390, 0.0009282
2: 0.9665967, 0.9680144, 0.9666520, 0.9681171, -0.0012468, 0.0011139
3: -0.0000718, 0.0103849, 0.0003366, 0.0111422, -0.0091963, 0.0082161
4: -0.0014829, -0.0006876, -0.0015405, -0.0007186, -0.0006249, 0.0006994
5: 0.0157716, 0.0165754, 0.0157134, 0.0165440, -0.0006316, 0.0007069
6: 0.0038445, 0.0042355, 0.0038598, 0.0042638, -0.0003438, 0.0003072
7: -0.0104696, -0.0077596, -0.0106659, -0.0078655, -0.0021293, 0.0023833
8: 0.0084231, 0.0105730, 0.0082674, 0.0104890, -0.0016893, 0.0018908
9: 0.0128743, 0.0167412, 0.0125943, 0.0165902, -0.0030383, 0.0034008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003263, upper bound: 0.0003845
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007527, upper bound: 0.0007508
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007438, upper bound: 0.0007480
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041540, -0.0041224, -0.0041527, -0.0041188, -0.0000290, 0.0000247
1: -0.0082221, -0.0070407, -0.0081735, -0.0069047, -0.0010877, 0.0009247
2: 0.9665967, 0.9680144, 0.9666550, 0.9681776, -0.0013052, 0.0011096
3: -0.0000718, 0.0103849, 0.0003583, 0.0115891, -0.0096272, 0.0081844
4: -0.0014829, -0.0006876, -0.0015744, -0.0007203, -0.0006225, 0.0007322
5: 0.0157716, 0.0165754, 0.0156791, 0.0165424, -0.0006291, 0.0007400
6: 0.0038445, 0.0042355, 0.0038606, 0.0042805, -0.0003599, 0.0003060
7: -0.0104696, -0.0077596, -0.0107817, -0.0078711, -0.0021211, 0.0024950
8: 0.0084231, 0.0105730, 0.0081755, 0.0104846, -0.0016827, 0.0019794
9: 0.0128743, 0.0167412, 0.0124290, 0.0165821, -0.0030266, 0.0035601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003263, upper bound: 0.0003845
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007527, upper bound: 0.0007508
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007438, upper bound: 0.0007480
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041527, -0.0041201, -0.0041540, -0.0041224, -0.0000248, 0.0000277
1: -0.0081759, -0.0069552, -0.0082221, -0.0070407, -0.0009282, 0.0010390
2: 0.9666520, 0.9681171, 0.9665967, 0.9680144, -0.0011139, 0.0012468
3: 0.0003366, 0.0111422, -0.0000718, 0.0103849, -0.0082161, 0.0091963
4: -0.0015405, -0.0007186, -0.0014829, -0.0006876, -0.0006994, 0.0006249
5: 0.0157134, 0.0165440, 0.0157716, 0.0165754, -0.0007069, 0.0006316
6: 0.0038598, 0.0042638, 0.0038445, 0.0042355, -0.0003072, 0.0003438
7: -0.0106659, -0.0078655, -0.0104696, -0.0077596, -0.0023833, 0.0021293
8: 0.0082674, 0.0104890, 0.0084231, 0.0105730, -0.0018908, 0.0016893
9: 0.0125943, 0.0165902, 0.0128743, 0.0167412, -0.0034008, 0.0030383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007106, upper bound: 0.0007773
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007018, upper bound: 0.0007773
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041527, -0.0041201, -0.0041539, -0.0041211, -0.0000262, 0.0000278
1: -0.0081759, -0.0069552, -0.0082177, -0.0069926, -0.0009821, 0.0010425
2: 0.9666520, 0.9681171, 0.9666018, 0.9680719, -0.0011785, 0.0012510
3: 0.0003366, 0.0111422, -0.0000334, 0.0108103, -0.0086926, 0.0092271
4: -0.0015405, -0.0007186, -0.0015152, -0.0006905, -0.0007018, 0.0006611
5: 0.0157134, 0.0165440, 0.0157389, 0.0165725, -0.0007093, 0.0006682
6: 0.0038598, 0.0042638, 0.0038459, 0.0042514, -0.0003250, 0.0003450
7: -0.0106659, -0.0078655, -0.0105798, -0.0077696, -0.0023913, 0.0022528
8: 0.0082674, 0.0104890, 0.0083356, 0.0105651, -0.0018971, 0.0017872
9: 0.0125943, 0.0165902, 0.0127170, 0.0167270, -0.0034122, 0.0032145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007106, upper bound: 0.0007774
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007018, upper bound: 0.0007774
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041527, -0.0041201, -0.0041527, -0.0041201, -0.0000250, 0.0000250
1: -0.0081759, -0.0069552, -0.0081759, -0.0069552, -0.0009378, 0.0009378
2: 0.9666520, 0.9681171, 0.9666520, 0.9681171, -0.0011255, 0.0011255
3: 0.0003366, 0.0111422, 0.0003366, 0.0111422, -0.0083012, 0.0083012
4: -0.0015405, -0.0007186, -0.0015405, -0.0007186, -0.0006314, 0.0006314
5: 0.0157134, 0.0165440, 0.0157134, 0.0165440, -0.0006381, 0.0006381
6: 0.0038598, 0.0042638, 0.0038598, 0.0042638, -0.0003104, 0.0003104
7: -0.0106659, -0.0078655, -0.0106659, -0.0078655, -0.0021513, 0.0021513
8: 0.0082674, 0.0104890, 0.0082674, 0.0104890, -0.0017068, 0.0017068
9: 0.0125943, 0.0165902, 0.0125943, 0.0165902, -0.0030698, 0.0030698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004748, upper bound: 0.0006211
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003882, upper bound: 0.0004802
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041527, -0.0041201, -0.0041527, -0.0041188, -0.0000265, 0.0000251
1: -0.0081759, -0.0069552, -0.0081735, -0.0069047, -0.0009937, 0.0009407
2: 0.9666520, 0.9681171, 0.9666550, 0.9681776, -0.0011924, 0.0011289
3: 0.0003366, 0.0111422, 0.0003583, 0.0115891, -0.0087953, 0.0083265
4: -0.0015405, -0.0007186, -0.0015744, -0.0007203, -0.0006333, 0.0006689
5: 0.0157134, 0.0165440, 0.0156791, 0.0165424, -0.0006400, 0.0006761
6: 0.0038598, 0.0042638, 0.0038606, 0.0042805, -0.0003288, 0.0003113
7: -0.0106659, -0.0078655, -0.0107817, -0.0078711, -0.0021579, 0.0022794
8: 0.0082674, 0.0104890, 0.0081755, 0.0104846, -0.0017120, 0.0018083
9: 0.0125943, 0.0165902, 0.0124290, 0.0165821, -0.0030791, 0.0032525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 96
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004748, upper bound: 0.0006211
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003882, upper bound: 0.0004802
time: 0.63 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.86 seconds
IS_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 2, lower bound: -0.0007106, upper bound: 0.0007508
IS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 2, lower bound: -0.0007014, upper bound: 0.0007480
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 2, lower bound: -0.0007527, upper bound: 0.0007508
IS_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 2, lower bound: -0.0007438, upper bound: 0.0007480
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 2, lower bound: -0.0007527, upper bound: 0.0007508
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 2, lower bound: -0.0007438, upper bound: 0.0007480
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 2, lower bound: -0.0007106, upper bound: 0.0007773
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 2, lower bound: -0.0007018, upper bound: 0.0007773
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 2, lower bound: -0.0007106, upper bound: 0.0007774
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.86
Output dim: 2, lower bound: -0.0007018, upper bound: 0.0007774
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 2, lower bound: -0.0004748, upper bound: 0.0006211
IS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 2, lower bound: -0.0003882, upper bound: 0.0004802
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 2, lower bound: -0.0004748, upper bound: 0.0006211
IS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.86
Output dim: 2, lower bound: -0.0003882, upper bound: 0.0004802

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041522, -0.0041202, -0.0041540, -0.0041224, -0.0000243, 0.0000277
1: -0.0081561, -0.0069562, -0.0082221, -0.0070407, -0.0009100, 0.0010378
2: 0.9666758, 0.9681159, 0.9665967, 0.9680144, -0.0010920, 0.0012455
3: 0.0005124, 0.0111333, -0.0000718, 0.0103849, -0.0080543, 0.0091862
4: -0.0015398, -0.0007320, -0.0014829, -0.0006876, -0.0006987, 0.0006126
5: 0.0157141, 0.0165305, 0.0157716, 0.0165754, -0.0007061, 0.0006191
6: 0.0038663, 0.0042634, 0.0038445, 0.0042355, -0.0003011, 0.0003435
7: -0.0106635, -0.0079110, -0.0104696, -0.0077596, -0.0023807, 0.0020873
8: 0.0082692, 0.0104529, 0.0084231, 0.0105730, -0.0018887, 0.0016560
9: 0.0125976, 0.0165252, 0.0128743, 0.0167412, -0.0033971, 0.0029785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005688, upper bound: 0.0005703
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007637, upper bound: 0.0007857
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007637, upper bound: 0.0007857
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041520, -0.0041198, -0.0041538, -0.0041224, -0.0000244, 0.0000280
1: -0.0081481, -0.0069438, -0.0082165, -0.0070409, -0.0009149, 0.0010486
2: 0.9666854, 0.9681306, 0.9666032, 0.9680142, -0.0010979, 0.0012584
3: 0.0005834, 0.0112430, -0.0000225, 0.0103835, -0.0080982, 0.0092817
4: -0.0015481, -0.0007374, -0.0014828, -0.0006913, -0.0007059, 0.0006159
5: 0.0157057, 0.0165251, 0.0157718, 0.0165716, -0.0007135, 0.0006225
6: 0.0038690, 0.0042675, 0.0038463, 0.0042354, -0.0003028, 0.0003470
7: -0.0106920, -0.0079294, -0.0104692, -0.0077724, -0.0024054, 0.0020987
8: 0.0082466, 0.0104383, 0.0084234, 0.0105629, -0.0019084, 0.0016650
9: 0.0125570, 0.0164989, 0.0128748, 0.0167230, -0.0034324, 0.0029947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005653, upper bound: 0.0005702
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007637, upper bound: 0.0007857
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007637, upper bound: 0.0007857
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041522, -0.0041202, -0.0041539, -0.0041211, -0.0000257, 0.0000278
1: -0.0081561, -0.0069562, -0.0082177, -0.0069926, -0.0009638, 0.0010413
2: 0.9666758, 0.9681159, 0.9666018, 0.9680719, -0.0011566, 0.0012496
3: 0.0005124, 0.0111333, -0.0000334, 0.0108103, -0.0085308, 0.0092171
4: -0.0015398, -0.0007320, -0.0015152, -0.0006905, -0.0007010, 0.0006488
5: 0.0157141, 0.0165305, 0.0157389, 0.0165725, -0.0007085, 0.0006557
6: 0.0038663, 0.0042634, 0.0038459, 0.0042514, -0.0003190, 0.0003446
7: -0.0106635, -0.0079110, -0.0105798, -0.0077696, -0.0023887, 0.0022108
8: 0.0082692, 0.0104529, 0.0083356, 0.0105651, -0.0018951, 0.0017540
9: 0.0125976, 0.0165252, 0.0127170, 0.0167270, -0.0034085, 0.0031547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003889, upper bound: 0.0004764
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007018, upper bound: 0.0007773
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007018, upper bound: 0.0007774
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041520, -0.0041198, -0.0041537, -0.0041211, -0.0000259, 0.0000281
1: -0.0081481, -0.0069438, -0.0082123, -0.0069928, -0.0009688, 0.0010518
2: 0.9666854, 0.9681306, 0.9666084, 0.9680718, -0.0011626, 0.0012622
3: 0.0005834, 0.0112430, 0.0000152, 0.0108090, -0.0085748, 0.0093096
4: -0.0015481, -0.0007374, -0.0015151, -0.0006942, -0.0007080, 0.0006522
5: 0.0157057, 0.0165251, 0.0157390, 0.0165687, -0.0007156, 0.0006591
6: 0.0038690, 0.0042675, 0.0038478, 0.0042513, -0.0003206, 0.0003481
7: -0.0106920, -0.0079294, -0.0105795, -0.0077822, -0.0024127, 0.0022222
8: 0.0082466, 0.0104383, 0.0083359, 0.0105551, -0.0019141, 0.0017630
9: 0.0125570, 0.0164989, 0.0127175, 0.0167090, -0.0034427, 0.0031710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 96
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003830, upper bound: 0.0004760
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 96

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007018, upper bound: 0.0007774
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007018, upper bound: 0.0007774
time: 0.74 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.87 seconds
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.87
Output dim: 2, lower bound: -0.0007637, upper bound: 0.0007857
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.87
Output dim: 2, lower bound: -0.0007637, upper bound: 0.0007857
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.87
Output dim: 2, lower bound: -0.0007637, upper bound: 0.0007857
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.87
Output dim: 2, lower bound: -0.0007637, upper bound: 0.0007857
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.87
Output dim: 2, lower bound: -0.0007018, upper bound: 0.0007773
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.87
Output dim: 2, lower bound: -0.0007018, upper bound: 0.0007774
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.87
Output dim: 2, lower bound: -0.0007018, upper bound: 0.0007774
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.87
Output dim: 2, lower bound: -0.0007018, upper bound: 0.0007774

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041522, -0.0041202, -0.0041534, -0.0041225, -0.0000243, 0.0000271
1: -0.0081561, -0.0069562, -0.0082006, -0.0070418, -0.0009088, 0.0010155
2: 0.9666758, 0.9681159, 0.9666225, 0.9680130, -0.0010906, 0.0012186
3: 0.0005124, 0.0111333, 0.0001188, 0.0103754, -0.0080440, 0.0089882
4: -0.0015398, -0.0007320, -0.0014821, -0.0007021, -0.0006836, 0.0006118
5: 0.0157141, 0.0165305, 0.0157724, 0.0165608, -0.0006909, 0.0006183
6: 0.0038663, 0.0042634, 0.0038516, 0.0042351, -0.0003008, 0.0003361
7: -0.0106635, -0.0079110, -0.0104671, -0.0078090, -0.0023294, 0.0020847
8: 0.0082692, 0.0104529, 0.0084250, 0.0105338, -0.0018480, 0.0016539
9: 0.0125976, 0.0165252, 0.0128778, 0.0166707, -0.0033238, 0.0029747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007475, upper bound: 0.0007624
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007540, upper bound: 0.0007747
time: 0.78 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041522, -0.0041202, -0.0041533, -0.0041220, -0.0000250, 0.0000272
1: -0.0081561, -0.0069562, -0.0081957, -0.0070233, -0.0009364, 0.0010188
2: 0.9666758, 0.9681159, 0.9666283, 0.9680352, -0.0011237, 0.0012227
3: 0.0005124, 0.0111333, 0.0001619, 0.0105388, -0.0082884, 0.0090181
4: -0.0015398, -0.0007320, -0.0014946, -0.0007053, -0.0006859, 0.0006304
5: 0.0157141, 0.0165305, 0.0157598, 0.0165575, -0.0006932, 0.0006371
6: 0.0038663, 0.0042634, 0.0038532, 0.0042412, -0.0003099, 0.0003372
7: -0.0106635, -0.0079110, -0.0105095, -0.0078202, -0.0023371, 0.0021480
8: 0.0082692, 0.0104529, 0.0083914, 0.0105250, -0.0018542, 0.0017041
9: 0.0125976, 0.0165252, 0.0128174, 0.0166548, -0.0033349, 0.0030650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007475, upper bound: 0.0007624
time: 0.85 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007540, upper bound: 0.0007747
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041520, -0.0041198, -0.0041534, -0.0041225, -0.0000241, 0.0000275
1: -0.0081481, -0.0069438, -0.0082006, -0.0070418, -0.0009014, 0.0010308
2: 0.9666854, 0.9681306, 0.9666225, 0.9680130, -0.0010817, 0.0012370
3: 0.0005834, 0.0112430, 0.0001188, 0.0103754, -0.0079786, 0.0091240
4: -0.0015481, -0.0007374, -0.0014821, -0.0007021, -0.0006939, 0.0006068
5: 0.0157057, 0.0165251, 0.0157724, 0.0165608, -0.0007013, 0.0006133
6: 0.0038690, 0.0042675, 0.0038516, 0.0042351, -0.0002983, 0.0003411
7: -0.0106920, -0.0079294, -0.0104671, -0.0078090, -0.0023646, 0.0020677
8: 0.0082466, 0.0104383, 0.0084250, 0.0105338, -0.0018759, 0.0016404
9: 0.0125570, 0.0164989, 0.0128778, 0.0166707, -0.0033741, 0.0029505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007432, upper bound: 0.0007610
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007524, upper bound: 0.0007747
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041520, -0.0041198, -0.0041533, -0.0041220, -0.0000246, 0.0000274
1: -0.0081481, -0.0069438, -0.0081957, -0.0070233, -0.0009215, 0.0010261
2: 0.9666854, 0.9681306, 0.9666283, 0.9680352, -0.0011058, 0.0012313
3: 0.0005834, 0.0112430, 0.0001619, 0.0105388, -0.0081561, 0.0090819
4: -0.0015481, -0.0007374, -0.0014946, -0.0007053, -0.0006907, 0.0006203
5: 0.0157057, 0.0165251, 0.0157598, 0.0165575, -0.0006981, 0.0006269
6: 0.0038690, 0.0042675, 0.0038532, 0.0042412, -0.0003049, 0.0003396
7: -0.0106920, -0.0079294, -0.0105095, -0.0078202, -0.0023537, 0.0021137
8: 0.0082466, 0.0104383, 0.0083914, 0.0105250, -0.0018673, 0.0016769
9: 0.0125570, 0.0164989, 0.0128174, 0.0166548, -0.0033585, 0.0030161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007432, upper bound: 0.0007610
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007524, upper bound: 0.0007747
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041522, -0.0041202, -0.0041533, -0.0041212, -0.0000257, 0.0000272
1: -0.0081561, -0.0069562, -0.0081960, -0.0069937, -0.0009627, 0.0010191
2: 0.9666758, 0.9681159, 0.9666279, 0.9680708, -0.0011552, 0.0012230
3: 0.0005124, 0.0111333, 0.0001590, 0.0108013, -0.0085209, 0.0090205
4: -0.0015398, -0.0007320, -0.0015145, -0.0007051, -0.0006861, 0.0006481
5: 0.0157141, 0.0165305, 0.0157396, 0.0165577, -0.0006934, 0.0006550
6: 0.0038663, 0.0042634, 0.0038531, 0.0042510, -0.0003186, 0.0003373
7: -0.0106635, -0.0079110, -0.0105775, -0.0078195, -0.0023377, 0.0022083
8: 0.0082692, 0.0104529, 0.0083374, 0.0105256, -0.0018547, 0.0017519
9: 0.0125976, 0.0165252, 0.0127203, 0.0166558, -0.0033358, 0.0031510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006858, upper bound: 0.0007401
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006984, upper bound: 0.0007666
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041522, -0.0041202, -0.0041531, -0.0041207, -0.0000264, 0.0000272
1: -0.0081561, -0.0069562, -0.0081909, -0.0069753, -0.0009872, 0.0010199
2: 0.9666758, 0.9681159, 0.9666341, 0.9680929, -0.0011847, 0.0012239
3: 0.0005124, 0.0111333, 0.0002044, 0.0109642, -0.0087382, 0.0090273
4: -0.0015398, -0.0007320, -0.0015269, -0.0007086, -0.0006866, 0.0006646
5: 0.0157141, 0.0165305, 0.0157271, 0.0165542, -0.0006939, 0.0006717
6: 0.0038663, 0.0042634, 0.0038548, 0.0042571, -0.0003267, 0.0003375
7: -0.0106635, -0.0079110, -0.0106197, -0.0078312, -0.0023395, 0.0022646
8: 0.0082692, 0.0104529, 0.0083040, 0.0105162, -0.0018560, 0.0017966
9: 0.0125976, 0.0165252, 0.0126601, 0.0166390, -0.0033383, 0.0032314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006858, upper bound: 0.0007401
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006984, upper bound: 0.0007666
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041520, -0.0041198, -0.0041533, -0.0041212, -0.0000255, 0.0000276
1: -0.0081481, -0.0069438, -0.0081960, -0.0069937, -0.0009553, 0.0010345
2: 0.9666854, 0.9681306, 0.9666279, 0.9680708, -0.0011464, 0.0012414
3: 0.0005834, 0.0112430, 0.0001590, 0.0108013, -0.0084555, 0.0091564
4: -0.0015481, -0.0007374, -0.0015145, -0.0007051, -0.0006964, 0.0006431
5: 0.0157057, 0.0165251, 0.0157396, 0.0165577, -0.0007038, 0.0006500
6: 0.0038690, 0.0042675, 0.0038531, 0.0042510, -0.0003161, 0.0003423
7: -0.0106920, -0.0079294, -0.0105775, -0.0078195, -0.0023729, 0.0021913
8: 0.0082466, 0.0104383, 0.0083374, 0.0105256, -0.0018826, 0.0017385
9: 0.0125570, 0.0164989, 0.0127203, 0.0166558, -0.0033860, 0.0031268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006694, upper bound: 0.0007346
time: 0.76 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006899, upper bound: 0.0007665
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041520, -0.0041198, -0.0041531, -0.0041207, -0.0000260, 0.0000275
1: -0.0081481, -0.0069438, -0.0081909, -0.0069753, -0.0009746, 0.0010290
2: 0.9666854, 0.9681306, 0.9666341, 0.9680929, -0.0011696, 0.0012349
3: 0.0005834, 0.0112430, 0.0002044, 0.0109642, -0.0086265, 0.0091081
4: -0.0015481, -0.0007374, -0.0015269, -0.0007086, -0.0006927, 0.0006561
5: 0.0157057, 0.0165251, 0.0157271, 0.0165542, -0.0007001, 0.0006631
6: 0.0038690, 0.0042675, 0.0038548, 0.0042571, -0.0003225, 0.0003405
7: -0.0106920, -0.0079294, -0.0106197, -0.0078312, -0.0023605, 0.0022356
8: 0.0082466, 0.0104383, 0.0083040, 0.0105162, -0.0018727, 0.0017737
9: 0.0125570, 0.0164989, 0.0126601, 0.0166390, -0.0033682, 0.0031901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 174
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006694, upper bound: 0.0007346
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0006899, upper bound: 0.0007666
time: 0.80 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.26 seconds
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0007475, upper bound: 0.0007624
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0007540, upper bound: 0.0007747
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0007475, upper bound: 0.0007624
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0007540, upper bound: 0.0007747
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0007432, upper bound: 0.0007610
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0007524, upper bound: 0.0007747
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0007432, upper bound: 0.0007610
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0007524, upper bound: 0.0007747
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0006858, upper bound: 0.0007401
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0006984, upper bound: 0.0007666
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0006858, upper bound: 0.0007401
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0006984, upper bound: 0.0007666
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0006694, upper bound: 0.0007346
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0006899, upper bound: 0.0007665
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0006694, upper bound: 0.0007346
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.26
Output dim: 2, lower bound: -0.0006899, upper bound: 0.0007666

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041522, -0.0041203, -0.0041534, -0.0041225, -0.0000241, 0.0000264
1: -0.0081545, -0.0069618, -0.0082006, -0.0070418, -0.0009024, 0.0009901
2: 0.9666778, 0.9681090, 0.9666225, 0.9680130, -0.0010830, 0.0011881
3: 0.0005268, 0.0110830, 0.0001188, 0.0103754, -0.0079877, 0.0087633
4: -0.0015360, -0.0007331, -0.0014821, -0.0007021, -0.0006665, 0.0006075
5: 0.0157180, 0.0165294, 0.0157724, 0.0165608, -0.0006736, 0.0006140
6: 0.0038669, 0.0042616, 0.0038516, 0.0042351, -0.0002986, 0.0003276
7: -0.0106505, -0.0079148, -0.0104671, -0.0078090, -0.0022711, 0.0020701
8: 0.0082795, 0.0104499, 0.0084250, 0.0105338, -0.0018018, 0.0016423
9: 0.0126161, 0.0165198, 0.0128778, 0.0166707, -0.0032407, 0.0029538

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005954, upper bound: 0.0005944
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0004154, upper bound: 0.0004683
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041522, -0.0041203, -0.0041533, -0.0041220, -0.0000248, 0.0000267
1: -0.0081545, -0.0069618, -0.0081957, -0.0070233, -0.0009300, 0.0009998
2: 0.9666778, 0.9681090, 0.9666283, 0.9680352, -0.0011161, 0.0011999
3: 0.0005268, 0.0110830, 0.0001619, 0.0105388, -0.0082321, 0.0088500
4: -0.0015360, -0.0007331, -0.0014946, -0.0007053, -0.0006731, 0.0006261
5: 0.0157180, 0.0165294, 0.0157598, 0.0165575, -0.0006803, 0.0006328
6: 0.0038669, 0.0042616, 0.0038532, 0.0042412, -0.0003078, 0.0003309
7: -0.0106505, -0.0079148, -0.0105095, -0.0078202, -0.0022935, 0.0021334
8: 0.0082795, 0.0104499, 0.0083914, 0.0105250, -0.0018196, 0.0016926
9: 0.0126161, 0.0165198, 0.0128174, 0.0166548, -0.0032727, 0.0030442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005551, upper bound: 0.0005607
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007381, upper bound: 0.0007709
time: 0.77 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007381, upper bound: 0.0007747
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041519, -0.0041200, -0.0041534, -0.0041225, -0.0000239, 0.0000269
1: -0.0081463, -0.0069496, -0.0082006, -0.0070418, -0.0008953, 0.0010071
2: 0.9666875, 0.9681237, 0.9666225, 0.9680130, -0.0010744, 0.0012085
3: 0.0005989, 0.0111912, 0.0001188, 0.0103754, -0.0079249, 0.0089137
4: -0.0015442, -0.0007386, -0.0014821, -0.0007021, -0.0006779, 0.0006027
5: 0.0157097, 0.0165239, 0.0157724, 0.0165608, -0.0006852, 0.0006092
6: 0.0038696, 0.0042656, 0.0038516, 0.0042351, -0.0002963, 0.0003333
7: -0.0106786, -0.0079335, -0.0104671, -0.0078090, -0.0023101, 0.0020538
8: 0.0082573, 0.0104351, 0.0084250, 0.0105338, -0.0018327, 0.0016294
9: 0.0125761, 0.0164932, 0.0128778, 0.0166707, -0.0032963, 0.0029306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005696, upper bound: 0.0005844
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0003714, upper bound: 0.0004480
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041519, -0.0041200, -0.0041533, -0.0041220, -0.0000244, 0.0000267
1: -0.0081463, -0.0069496, -0.0081957, -0.0070233, -0.0009152, 0.0010008
2: 0.9666875, 0.9681237, 0.9666283, 0.9680352, -0.0010982, 0.0012010
3: 0.0005989, 0.0111912, 0.0001619, 0.0105388, -0.0081005, 0.0088584
4: -0.0015442, -0.0007386, -0.0014946, -0.0007053, -0.0006737, 0.0006161
5: 0.0157097, 0.0165239, 0.0157598, 0.0165575, -0.0006809, 0.0006227
6: 0.0038696, 0.0042656, 0.0038532, 0.0042412, -0.0003029, 0.0003312
7: -0.0106786, -0.0079335, -0.0105095, -0.0078202, -0.0022957, 0.0020993
8: 0.0082573, 0.0104351, 0.0083914, 0.0105250, -0.0018213, 0.0016655
9: 0.0125761, 0.0164932, 0.0128174, 0.0166548, -0.0032758, 0.0029956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 174
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005518, upper bound: 0.0005607
time: 0.71 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 174

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007331, upper bound: 0.0007709
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0007331, upper bound: 0.0007747
time: 0.89 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 6.12 seconds
IS_A2_B1_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 6.12
Output dim: 2, lower bound: -0.0005954, upper bound: 0.0005944
IS_A2_B1_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 6.12
Output dim: 2, lower bound: -0.0004154, upper bound: 0.0004683
IS_A2_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.12
Output dim: 2, lower bound: -0.0007381, upper bound: 0.0007709
IS_A2_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.12
Output dim: 2, lower bound: -0.0007381, upper bound: 0.0007747
IS_A2_B1_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 6.12
Output dim: 2, lower bound: -0.0005696, upper bound: 0.0005844
IS_A2_B1_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 6.12
Output dim: 2, lower bound: -0.0003714, upper bound: 0.0004480
IS_A2_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 6.12
Output dim: 2, lower bound: -0.0007331, upper bound: 0.0007709
IS_A2_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 6.12
Output dim: 2, lower bound: -0.0007331, upper bound: 0.0007747

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041522, -0.0041203, -0.0041535, -0.0041223, -0.0000243, 0.0000270
1: -0.0081545, -0.0069618, -0.0082032, -0.0070367, -0.0009102, 0.0010105
2: 0.9666778, 0.9681090, 0.9666192, 0.9680191, -0.0010923, 0.0012126
3: 0.0005268, 0.0110830, 0.0000949, 0.0104201, -0.0080564, 0.0089441
4: -0.0015360, -0.0007331, -0.0014855, -0.0007002, -0.0006803, 0.0006127
5: 0.0157180, 0.0165294, 0.0157689, 0.0165626, -0.0006875, 0.0006193
6: 0.0038669, 0.0042616, 0.0038507, 0.0042368, -0.0003012, 0.0003344
7: -0.0106505, -0.0079148, -0.0104787, -0.0078028, -0.0023180, 0.0020879
8: 0.0082795, 0.0104499, 0.0084158, 0.0105387, -0.0018390, 0.0016564
9: 0.0126161, 0.0165198, 0.0128613, 0.0166795, -0.0033075, 0.0029792

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005979, upper bound: 0.0006762
time: 0.73 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007309, upper bound: 0.0007638
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041522, -0.0041203, -0.0041532, -0.0041221, -0.0000243, 0.0000266
1: -0.0081545, -0.0069618, -0.0081936, -0.0070297, -0.0009090, 0.0009951
2: 0.9666778, 0.9681090, 0.9666308, 0.9680275, -0.0010909, 0.0011942
3: 0.0005268, 0.0110830, 0.0001801, 0.0104822, -0.0080459, 0.0088080
4: -0.0015360, -0.0007331, -0.0014903, -0.0007067, -0.0006699, 0.0006119
5: 0.0157180, 0.0165294, 0.0157642, 0.0165561, -0.0006771, 0.0006185
6: 0.0038669, 0.0042616, 0.0038539, 0.0042391, -0.0003008, 0.0003293
7: -0.0106505, -0.0079148, -0.0104948, -0.0078249, -0.0022827, 0.0020852
8: 0.0082795, 0.0104499, 0.0084031, 0.0105212, -0.0018110, 0.0016543
9: 0.0126161, 0.0165198, 0.0128383, 0.0166480, -0.0032572, 0.0029754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005979, upper bound: 0.0006916
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007309, upper bound: 0.0007675
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041519, -0.0041200, -0.0041535, -0.0041223, -0.0000239, 0.0000270
1: -0.0081463, -0.0069496, -0.0082032, -0.0070367, -0.0008951, 0.0010119
2: 0.9666875, 0.9681237, 0.9666192, 0.9680191, -0.0010741, 0.0012143
3: 0.0005989, 0.0111912, 0.0000949, 0.0104201, -0.0079225, 0.0089564
4: -0.0015442, -0.0007386, -0.0014855, -0.0007002, -0.0006812, 0.0006026
5: 0.0157097, 0.0165239, 0.0157689, 0.0165626, -0.0006885, 0.0006090
6: 0.0038696, 0.0042656, 0.0038507, 0.0042368, -0.0002962, 0.0003349
7: -0.0106786, -0.0079335, -0.0104787, -0.0078028, -0.0023211, 0.0020532
8: 0.0082573, 0.0104351, 0.0084158, 0.0105387, -0.0018415, 0.0016289
9: 0.0125761, 0.0164932, 0.0128613, 0.0166795, -0.0033121, 0.0029297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005936, upper bound: 0.0006756
time: 0.75 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007257, upper bound: 0.0007639
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041519, -0.0041200, -0.0041532, -0.0041221, -0.0000238, 0.0000266
1: -0.0081463, -0.0069496, -0.0081936, -0.0070297, -0.0008931, 0.0009958
2: 0.9666875, 0.9681237, 0.9666308, 0.9680275, -0.0010718, 0.0011951
3: 0.0005989, 0.0111912, 0.0001801, 0.0104822, -0.0079051, 0.0088145
4: -0.0015442, -0.0007386, -0.0014903, -0.0007067, -0.0006704, 0.0006012
5: 0.0157097, 0.0165239, 0.0157642, 0.0165561, -0.0006776, 0.0006076
6: 0.0038696, 0.0042656, 0.0038539, 0.0042391, -0.0002956, 0.0003296
7: -0.0106786, -0.0079335, -0.0104948, -0.0078249, -0.0022844, 0.0020487
8: 0.0082573, 0.0104351, 0.0084031, 0.0105212, -0.0018123, 0.0016253
9: 0.0125761, 0.0164932, 0.0128383, 0.0166480, -0.0032596, 0.0029233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 1

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0005936, upper bound: 0.0006911
time: 0.88 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.0007257, upper bound: 0.0007675
time: 0.85 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 5.26 seconds
IS_A2_B1_A1_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 2, lower bound: -0.0005979, upper bound: 0.0006762
IS_A2_B1_A1_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 2, lower bound: -0.0007309, upper bound: 0.0007638
IS_A2_B1_A1_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 2, lower bound: -0.0005979, upper bound: 0.0006916
IS_A2_B1_A1_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 2, lower bound: -0.0007309, upper bound: 0.0007675
IS_A2_B1_A1_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 2, lower bound: -0.0005936, upper bound: 0.0006756
IS_A2_B1_A1_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 2, lower bound: -0.0007257, upper bound: 0.0007639
IS_A2_B1_A1_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 2, lower bound: -0.0005936, upper bound: 0.0006911
IS_A2_B1_A1_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 5.26
Output dim: 2, lower bound: -0.0007257, upper bound: 0.0007675

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 3.13 + 185.88 = 189.02 seconds
