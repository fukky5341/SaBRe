## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00018056


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0068738, 0.0071593, 0.0068738, 0.0071593, -0.0001543, 0.0001543)
1: (0.0013961, 0.0019491, 0.0013961, 0.0019491, -0.0002989, 0.0002989)
2: (0.0012069, 0.0056672, 0.0012069, 0.0056672, -0.0024110, 0.0024110)
3: (-0.0030187, -0.0026204, -0.0030187, -0.0026204, -0.0002153, 0.0002153)
4: (0.0066898, 0.0086226, 0.0066898, 0.0086226, -0.0010448, 0.0010448)
5: (-0.0018069, -0.0015184, -0.0018069, -0.0015184, -0.0001560, 0.0001560)
6: (0.9930379, 0.9935671, 0.9930379, 0.9935671, -0.0002861, 0.0002861)
7: (-0.0012732, 0.0022256, -0.0012732, 0.0022256, -0.0018913, 0.0018913)
8: (0.0005895, 0.0016856, 0.0005895, 0.0016856, -0.0005925, 0.0005925)
9: (-0.0106934, -0.0085057, -0.0106934, -0.0085057, -0.0011826, 0.0011826)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 1.43 = 2.93 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.0002082, upper bound: 0.0002082

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001996, upper bound: 0.0001826
time: 0.63 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001996, upper bound: 0.0001996
time: 0.57 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.33 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 6, lower bound: -0.0001996, upper bound: 0.0001826
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.33
Output dim: 6, lower bound: -0.0001996, upper bound: 0.0001996

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0068742, 0.0071409, 0.0068739, 0.0071547, -0.0001493, 0.0001369
1: 0.0013967, 0.0019133, 0.0013962, 0.0019400, -0.0002891, 0.0002652
2: 0.0014951, 0.0056619, 0.0012799, 0.0056659, -0.0021393, 0.0023321
3: -0.0030183, -0.0026461, -0.0030186, -0.0026269, -0.0002083, 0.0001911
4: 0.0066921, 0.0084978, 0.0066904, 0.0085910, -0.0010106, 0.0009270
5: -0.0017883, -0.0015187, -0.0018022, -0.0015185, -0.0001384, 0.0001509
6: 0.9930386, 0.9935329, 0.9930381, 0.9935584, -0.0002767, 0.0002538
7: -0.0012690, 0.0019996, -0.0012721, 0.0021683, -0.0018293, 0.0016781
8: 0.0005908, 0.0016148, 0.0005898, 0.0016677, -0.0005731, 0.0005257
9: -0.0105520, -0.0085083, -0.0106576, -0.0085063, -0.0010493, 0.0011439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 97

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001826, upper bound: 0.0001826
time: 0.64 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001826, upper bound: 0.0001826
time: 0.63 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0068583, 0.0071507, 0.0068740, 0.0071567, -0.0001745, 0.0001433
1: 0.0013661, 0.0019324, 0.0013964, 0.0019440, -0.0003380, 0.0002775
2: 0.0013411, 0.0059091, 0.0012477, 0.0056650, -0.0022386, 0.0027264
3: -0.0030403, -0.0026323, -0.0030185, -0.0026240, -0.0002435, 0.0001999
4: 0.0065850, 0.0085645, 0.0066908, 0.0086050, -0.0011815, 0.0009701
5: -0.0017982, -0.0015027, -0.0018043, -0.0015185, -0.0001448, 0.0001764
6: 0.9930092, 0.9935512, 0.9930382, 0.9935623, -0.0003235, 0.0002656
7: -0.0014629, 0.0021204, -0.0012714, 0.0021936, -0.0021387, 0.0017560
8: 0.0005300, 0.0016526, 0.0005900, 0.0016756, -0.0006700, 0.0005501
9: -0.0106276, -0.0083870, -0.0106734, -0.0085068, -0.0010980, 0.0013373

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001826, upper bound: 0.0001996
time: 0.61 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001826, upper bound: 0.0001996
time: 0.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.74 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 6, lower bound: -0.0001826, upper bound: 0.0001826
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 6, lower bound: -0.0001826, upper bound: 0.0001826
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 6, lower bound: -0.0001826, upper bound: 0.0001996
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.74
Output dim: 6, lower bound: -0.0001826, upper bound: 0.0001996

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0068742, 0.0071409, 0.0068742, 0.0071409, -0.0001368, 0.0001368
1: 0.0013967, 0.0019133, 0.0013967, 0.0019133, -0.0002649, 0.0002649
2: 0.0014951, 0.0056619, 0.0014951, 0.0056619, -0.0021365, 0.0021365
3: -0.0030183, -0.0026461, -0.0030183, -0.0026461, -0.0001908, 0.0001908
4: 0.0066921, 0.0084978, 0.0066921, 0.0084978, -0.0009258, 0.0009258
5: -0.0017883, -0.0015187, -0.0017883, -0.0015187, -0.0001382, 0.0001382
6: 0.9930386, 0.9935329, 0.9930386, 0.9935329, -0.0002535, 0.0002535
7: -0.0012690, 0.0019996, -0.0012690, 0.0019996, -0.0016759, 0.0016759
8: 0.0005908, 0.0016148, 0.0005908, 0.0016148, -0.0005251, 0.0005251
9: -0.0105520, -0.0085083, -0.0105520, -0.0085083, -0.0010479, 0.0010479

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001794, upper bound: 0.0001643
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001795, upper bound: 0.0001753
time: 0.62 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0068742, 0.0071409, 0.0068583, 0.0071507, -0.0001494, 0.0001591
1: 0.0013967, 0.0019133, 0.0013661, 0.0019324, -0.0002893, 0.0003081
2: 0.0014951, 0.0056619, 0.0013411, 0.0059091, -0.0024850, 0.0023334
3: -0.0030183, -0.0026461, -0.0030403, -0.0026323, -0.0002084, 0.0002220
4: 0.0066921, 0.0084978, 0.0065850, 0.0085645, -0.0010111, 0.0010769
5: -0.0017883, -0.0015187, -0.0017982, -0.0015027, -0.0001608, 0.0001509
6: 0.9930386, 0.9935329, 0.9930092, 0.9935512, -0.0002768, 0.0002948
7: -0.0012690, 0.0019996, -0.0014629, 0.0021204, -0.0018304, 0.0019493
8: 0.0005908, 0.0016148, 0.0005300, 0.0016526, -0.0005734, 0.0006107
9: -0.0105520, -0.0085083, -0.0106276, -0.0083870, -0.0012189, 0.0011445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001794, upper bound: 0.0001643
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001795, upper bound: 0.0001752
time: 0.61 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0068583, 0.0071507, 0.0068742, 0.0071409, -0.0001591, 0.0001494
1: 0.0013661, 0.0019324, 0.0013967, 0.0019133, -0.0003081, 0.0002893
2: 0.0013411, 0.0059091, 0.0014951, 0.0056619, -0.0023334, 0.0024850
3: -0.0030403, -0.0026323, -0.0030183, -0.0026461, -0.0002220, 0.0002084
4: 0.0065850, 0.0085645, 0.0066921, 0.0084978, -0.0010769, 0.0010111
5: -0.0017982, -0.0015027, -0.0017883, -0.0015187, -0.0001509, 0.0001608
6: 0.9930092, 0.9935512, 0.9930386, 0.9935329, -0.0002948, 0.0002768
7: -0.0014629, 0.0021204, -0.0012690, 0.0019996, -0.0019493, 0.0018304
8: 0.0005300, 0.0016526, 0.0005908, 0.0016148, -0.0006107, 0.0005734
9: -0.0106276, -0.0083870, -0.0105520, -0.0085083, -0.0011445, 0.0012189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001752, upper bound: 0.0001801
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001752, upper bound: 0.0001926
time: 0.65 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0068583, 0.0071507, 0.0068583, 0.0071507, -0.0001431, 0.0001431
1: 0.0013661, 0.0019324, 0.0013661, 0.0019324, -0.0002772, 0.0002772
2: 0.0013411, 0.0059091, 0.0013411, 0.0059091, -0.0022362, 0.0022362
3: -0.0030403, -0.0026323, -0.0030403, -0.0026323, -0.0001997, 0.0001997
4: 0.0065850, 0.0085645, 0.0065850, 0.0085645, -0.0009690, 0.0009690
5: -0.0017982, -0.0015027, -0.0017982, -0.0015027, -0.0001447, 0.0001447
6: 0.9930092, 0.9935512, 0.9930092, 0.9935512, -0.0002653, 0.0002653
7: -0.0014629, 0.0021204, -0.0014629, 0.0021204, -0.0017541, 0.0017541
8: 0.0005300, 0.0016526, 0.0005300, 0.0016526, -0.0005495, 0.0005495
9: -0.0106276, -0.0083870, -0.0106276, -0.0083870, -0.0010968, 0.0010968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 97

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 68

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001752, upper bound: 0.0001801
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001752, upper bound: 0.0001926
time: 0.80 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.82 seconds
IS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001794, upper bound: 0.0001643
IS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001795, upper bound: 0.0001753
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001794, upper bound: 0.0001643
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001795, upper bound: 0.0001752
IS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001752, upper bound: 0.0001801
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001752, upper bound: 0.0001926
IS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001752, upper bound: 0.0001801
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.82
Output dim: 6, lower bound: -0.0001752, upper bound: 0.0001926

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0068431, 0.0071423, 0.0068743, 0.0071383, -0.0001702, 0.0001461
1: 0.0013365, 0.0019161, 0.0013971, 0.0019084, -0.0003297, 0.0002831
2: 0.0014730, 0.0061479, 0.0015350, 0.0056593, -0.0022832, 0.0026595
3: -0.0030617, -0.0026441, -0.0030180, -0.0026497, -0.0002375, 0.0002039
4: 0.0064815, 0.0085073, 0.0066932, 0.0084804, -0.0011525, 0.0009894
5: -0.0017897, -0.0014873, -0.0017857, -0.0015189, -0.0001477, 0.0001720
6: 0.9929810, 0.9935356, 0.9930388, 0.9935282, -0.0003155, 0.0002709
7: -0.0016502, 0.0020169, -0.0012670, 0.0019682, -0.0020862, 0.0017910
8: 0.0004713, 0.0016202, 0.0005914, 0.0016050, -0.0006536, 0.0005611
9: -0.0105629, -0.0082699, -0.0105325, -0.0085095, -0.0011199, 0.0013045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001842
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001804
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0068431, 0.0071423, 0.0068585, 0.0071482, -0.0001596, 0.0001392
1: 0.0013365, 0.0019161, 0.0013664, 0.0019275, -0.0003091, 0.0002696
2: 0.0014730, 0.0061479, 0.0013807, 0.0059066, -0.0021743, 0.0024931
3: -0.0030617, -0.0026441, -0.0030401, -0.0026359, -0.0002227, 0.0001942
4: 0.0064815, 0.0085073, 0.0065860, 0.0085473, -0.0010804, 0.0009422
5: -0.0017897, -0.0014873, -0.0017957, -0.0015029, -0.0001407, 0.0001613
6: 0.9929810, 0.9935356, 0.9930095, 0.9935465, -0.0002958, 0.0002580
7: -0.0016502, 0.0020169, -0.0014610, 0.0020893, -0.0019556, 0.0017056
8: 0.0004713, 0.0016202, 0.0005306, 0.0016429, -0.0006127, 0.0005343
9: -0.0105629, -0.0082699, -0.0106082, -0.0083882, -0.0010665, 0.0012228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 240
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 108
type: B, layer: 1, pos: 160
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 237
type: B, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 240

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001842
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001804
time: 0.65 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.66 seconds
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.66
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001842
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001804
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.66
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001842
IS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 2.66
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001804

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0068431, 0.0071423, 0.0068780, 0.0071383, -0.0001702, 0.0001421
1: 0.0013365, 0.0019161, 0.0014042, 0.0019084, -0.0003297, 0.0002752
2: 0.0014730, 0.0061479, 0.0015350, 0.0056021, -0.0022194, 0.0026595
3: -0.0030617, -0.0026441, -0.0030129, -0.0026497, -0.0002375, 0.0001982
4: 0.0064815, 0.0085073, 0.0067180, 0.0084804, -0.0011525, 0.0009618
5: -0.0017897, -0.0014873, -0.0017857, -0.0015226, -0.0001436, 0.0001720
6: 0.9929810, 0.9935356, 0.9930456, 0.9935282, -0.0003155, 0.0002633
7: -0.0016502, 0.0020169, -0.0012221, 0.0019682, -0.0020862, 0.0017409
8: 0.0004713, 0.0016202, 0.0006055, 0.0016050, -0.0006536, 0.0005454
9: -0.0105629, -0.0082699, -0.0105325, -0.0085376, -0.0010886, 0.0013045

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 97

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001804
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001804
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0068431, 0.0071423, 0.0068622, 0.0071482, -0.0001596, 0.0001351
1: 0.0013365, 0.0019161, 0.0013735, 0.0019275, -0.0003091, 0.0002616
2: 0.0014730, 0.0061479, 0.0013807, 0.0058496, -0.0021102, 0.0024931
3: -0.0030617, -0.0026441, -0.0030350, -0.0026359, -0.0002227, 0.0001885
4: 0.0064815, 0.0085073, 0.0066107, 0.0085473, -0.0010804, 0.0009144
5: -0.0017897, -0.0014873, -0.0017957, -0.0015066, -0.0001365, 0.0001613
6: 0.9929810, 0.9935356, 0.9930162, 0.9935465, -0.0002958, 0.0002504
7: -0.0016502, 0.0020169, -0.0014163, 0.0020893, -0.0019556, 0.0016553
8: 0.0004713, 0.0016202, 0.0005446, 0.0016429, -0.0006127, 0.0005186
9: -0.0105629, -0.0082699, -0.0106082, -0.0084162, -0.0010350, 0.0012228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=8, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 108
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 160
type: A, layer: 1, pos: 237
type: A, layer: 1, pos: 97

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 240

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001804
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001804
time: 0.63 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.70 seconds
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001804
IS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001804
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001804
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.70
Output dim: 6, lower bound: -0.0001639, upper bound: 0.0001804

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.93 + 28.27 = 31.21 seconds
